#include <QStringBuilder>

#include "gcode/writers/cincinnati_writer.h"
#include "utilities/enums.h"
#include "utilities/mathutils.h"

namespace ORNL
{
    CincinnatiWriter::CincinnatiWriter(GcodeMeta meta, const QSharedPointer<SettingsBase>& sb) : WriterBase(meta, sb) {
        m_laser_prefix = "G104 P60000 ";
        m_laser_delimiter = '|';
        m_M10 = "M10";
        m_M11 = "M11";
        m_M64 = "M64";
        m_M65 = "M65";
    }

    QString CincinnatiWriter::writeInitialSetup(Distance minimum_x, Distance minimum_y, Distance maximum_x, Distance maximum_y, int num_layers)
    {
        m_current_z = m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kZOffset);
        m_current_w = m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kWMax);
        m_current_rpm = 0;       
        for (int ext = 0, end = m_extruders_on.size(); ext < end; ++ext) //all extruders off initially
            m_extruders_on[ext] = false;
        m_first_travel = true;
        m_z_travel = false;
        m_w_travel = false;
        m_first_print = true;
        m_layer_start = true;
        m_wire_feed = false;
        m_need_wirecut = false;
        m_min_z = 0.0f;
        m_material_number = -1;
        QString rv;
        if (m_sb->setting< int >(Constants::PrinterSettings::GCode::kEnableStartupCode))
        {
            rv += commentLine("SAFETY BLOCK - ESTABLISH OPERATIONAL MODES");
            rv += m_M11 % commentSpaceLine("EXIT SERVOING MODE");
            if (m_sb->setting< int >(Constants::PrinterSettings::Acceleration::kEnableDynamic))
            {
                rv += writeAcceleration(m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kDefault));
            }
            rv += "M16 " % commentLine("DEFAULT SPINDLE ADJUSTMENT");
            if (m_sb->setting< int >(Constants::MaterialSettings::Cooling::kForceMinLayerTime) &&
                    m_sb->setting< int >(Constants::MaterialSettings::Cooling::kForceMinLayerTimeMethod) == (int)ForceMinimumLayerTime::kSlow_Feedrate)
            {
                rv += "M270 L1 " % commentLine("SET FEEDRATE MULTIPLIER TO 100%");
            }
            rv += "G1 F120 " % commentLine("SET INITIAL FEEDRATE");
            if (m_sb->setting< int >(Constants::PrinterSettings::GCode::kEnableWaitForUser))
            {
                rv += "M0" % commentSpaceLine("WAIT FOR USER");
            }
            rv += writeDwell(0.25);

            if (m_sb->setting< int >(Constants::PrinterSettings::Dimensions::kLayerChangeAxis) != static_cast<int>(LayerChange::kW_only))
            {
                //can't output in Z if there is no Z-axis
                rv += "G0 Z0 " % commentLine("LIFT Z FOR SAFETY");
                m_current_z = 0;
            }
            if (m_sb->setting< int >(Constants::PrinterSettings::Dimensions::kLayerChangeAxis) != static_cast<int>(LayerChange::kZ_only))
            {
                //can't output in W if there is no W-axis
                m_current_w = m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kWMax);
                rv += m_G0 % m_w % QString::number(m_current_w.to(m_meta.m_distance_unit), 'f', 4)
                    % commentSpaceLine("SET INITIAL W HEIGHT");
            }
        }

        if(m_sb->setting< int >(Constants::PrinterSettings::GCode::kEnableBoundingBox))
        {
            rv += "G0 Z0" % commentSpaceLine("RAISE Z TO DEMO BOUNDING BOX")
                % m_G0 % m_x % QString::number(minimum_x.to(m_meta.m_distance_unit), 'f', 4) % " Y" % QString::number(minimum_y.to(m_meta.m_distance_unit), 'f', 4) % commentSpaceLine("BOUNDING BOX")
                % m_G0 % m_x % QString::number(maximum_x.to(m_meta.m_distance_unit), 'f', 4) % " Y" % QString::number(minimum_y.to(m_meta.m_distance_unit), 'f', 4) % commentSpaceLine("BOUNDING BOX")
                % m_G0 % m_x % QString::number(maximum_x.to(m_meta.m_distance_unit), 'f', 4) % " Y" % QString::number(maximum_y.to(m_meta.m_distance_unit), 'f', 4) % commentSpaceLine("BOUNDING BOX")
                % m_G0 % m_x % QString::number(minimum_x.to(m_meta.m_distance_unit), 'f', 4) % " Y" % QString::number(maximum_y.to(m_meta.m_distance_unit), 'f', 4) % commentSpaceLine("BOUNDING BOX")
                % m_G0 % m_x % QString::number(minimum_x.to(m_meta.m_distance_unit), 'f', 4) % " Y" % QString::number(minimum_y.to(m_meta.m_distance_unit), 'f', 4) % commentSpaceLine("BOUNDING BOX")
                % "M0" % commentSpaceLine("WAIT FOR USER");

            m_start_point = Point(minimum_x, minimum_y, 0);
        }

        if (m_sb->setting< int >(Constants::PrinterSettings::GCode::kEnableMaterialLoad))
        {
            rv += writePurge(m_sb->setting< int >(Constants::MaterialSettings::Purge::kInitialScrewRPM),
                             m_sb->setting< int >(Constants::MaterialSettings::Purge::kInitialDuration),
                             m_sb->setting< int >(Constants::MaterialSettings::Purge::kInitialTipWipeDelay));
            if (m_sb->setting< int >(Constants::PrinterSettings::GCode::kEnableWaitForUser))
            {
                rv += "M0" % commentSpaceLine("WAIT FOR USER");
            }
        }

        if(m_sb->setting<bool>(Constants::ProfileSettings::LaserScanner::kLaserScanner))
        {
            Distance scannerZOffset = m_sb->setting<Distance>(Constants::ProfileSettings::LaserScanner::kLaserScannerHeight)
                    - m_sb->setting<Distance>(Constants::ProfileSettings::LaserScanner::kLaserScannerHeightOffset);

            rv += m_laser_prefix % commentLine(Constants::RegionTypeStrings::kLaserScan % " - TRANSMIT NECESSARY BUILD PARAMETERS|" %
                     QString::number(m_sb->setting<Distance>(Constants::ProfileSettings::Layer::kBeadWidth).to(mm), 'f', 4) % m_laser_delimiter %
                     QString::number(m_sb->setting<Distance>(Constants::ProfileSettings::Layer::kLayerHeight).to(mm), 'f', 4) % m_laser_delimiter %
                     QString::number(m_sb->setting<Distance>(Constants::ProfileSettings::LaserScanner::kLaserScannerHeightOffset).to(mm), 'f', 4) % m_laser_delimiter %
                     QString::number(m_sb->setting<Distance>(Constants::ProfileSettings::LaserScanner::kLaserScannerHeightOffset).to(mm), 'f', 4) % m_laser_delimiter %
                     QString::number(m_sb->setting<Distance>(Constants::ProfileSettings::LaserScanner::kLaserScannerWidth).to(mm), 'f', 4) % m_laser_delimiter %
                     QString::number(m_sb->setting<Distance>(Constants::ProfileSettings::LaserScanner::kLaserScanLineResolution).to(mm), 'f', 4) % m_laser_delimiter %
                     QString::number((int)m_sb->setting<bool>(Constants::ProfileSettings::LaserScanner::kInvertLaserScannerHead)) % m_laser_delimiter %
                     QString::number((int)m_sb->setting<bool>(Constants::ProfileSettings::LaserScanner::kEnableBedScan)) % m_laser_delimiter %
                     QString::number(m_sb->setting<int>(Constants::ProfileSettings::LaserScanner::kLaserScannerAxis)) % m_laser_delimiter %
                     QString::number(scannerZOffset.to(mm), 'f', 4) % m_laser_delimiter %
                     QString::number((int)m_sb->setting<bool>(Constants::ProfileSettings::LaserScanner::kTransmitHeightMap))) %
                     m_laser_prefix % commentLine(Constants::RegionTypeStrings::kLaserScan % " - TRANSMIT FILENAME|#TEMPFILENAME#");
        }
        else if(m_sb->setting<bool>(Constants::ProfileSettings::ThermalScanner::kThermalScanner))
            rv += m_laser_prefix % commentLine(Constants::RegionTypeStrings::kThermalScan % " - TRANSMIT FILENAME|#TEMPFILENAME#");

        if(m_sb->setting< QString >(Constants::PrinterSettings::GCode::kStartCode) != "")
            rv += m_sb->setting< QString >(Constants::PrinterSettings::GCode::kStartCode);

        rv += m_newline;

        rv += commentLine("LAYER COUNT: " % QString::number(num_layers));

        return rv;
    }

    QString CincinnatiWriter::writeBeforeLayer(float new_min_z, QSharedPointer<SettingsBase> sb)
    {
        QString rv;

        m_layer_start = true;

        Distance layer_height = sb->setting<Distance>(Constants::ProfileSettings::Layer::kLayerHeight);
        m_spiral_layer = sb->setting<bool>(Constants::ProfileSettings::SpecialModes::kEnableSpiralize);
        //if printer is Z_Only or W_Only, the vertical change is completely handled by the path, no need for change between layers

        /*
         *  If the printer has both Z and W axes(extruder and table move) and we're slicing along the z-axis, move the table
         *  between layers
         *  Move the table down between layers only if the minimum z value on this layer is greater than the layer before and
         *  the difference in the minimum z values is (nearly) a layer height
        */
        if (m_sb->setting< int >(Constants::PrinterSettings::Dimensions::kLayerChangeAxis) == static_cast<int>(LayerChange::kBoth_Z_and_W)
                && static_cast<Axis>(m_sb->setting<int>(Constants::ExperimentalSettings::SlicingAngle::kSlicingAxis)) == Axis::kZ &&
                !m_spiral_layer)
        {
            if (new_min_z == std::numeric_limits<float>::max()) //means that a layer doesn't have geometry on it, didn't find a lower point
                new_min_z = m_min_z;                            //don't want this to affect table height

           // Distance layer_height = m_sb->setting< Distance >(Constants::ProfileSettings::Layer::kLayerHeight);
            float z_change = (new_min_z - m_min_z);
            if (z_change > 0 && (layer_height - z_change) < 10) //if the z_min moved up by a layer height or within 10 microns
            {
                m_min_z = new_min_z; //update min for next call

                //move table down by layer height, if possible
                if (m_current_w - layer_height > m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kWMin))
                {
                    m_current_w -= layer_height;
                    rv += m_G0 % m_w % QString::number(m_current_w.to(m_meta.m_distance_unit), 'f', 4) %
                            commentSpaceLine("MOVE W - LAYER CHANGE");
                    setFeedrate(m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kWTableSpeed));
                }
                //if table is above min, but can not move down the entire layer height, move to min
                else if( m_current_w > m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kWMin))
                {
                    m_current_w = m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kWMin);
                    rv += m_G0 % m_w % QString::number(m_current_w.to(m_meta.m_distance_unit), 'f', 4) %
                            commentSpaceLine("MOVE W TO BOTTOM");
                    setFeedrate(m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kWTableSpeed));
                    //no longer need to raise Z to compensate for rest of layer height
                }
            }
        }

        rv += "M1 " % commentLine("OPTIONAL STOP - LAYER CHANGE");
        return rv;
    }

    QString CincinnatiWriter::writeBeforePart(QVector3D normal)
    {
        QString rv;
        return rv;
    }

    QString CincinnatiWriter::writeBeforeIsland()
    {
        QString rv;
        return rv;
    }

    QString CincinnatiWriter::writeBeforeScan(Point min, Point max, int layer, int boundingBox, Axis axis, Angle angle)
    {
        QString rv;

        rv += m_laser_prefix % commentLine(Constants::RegionTypeStrings::kLaserScan % " - TRANSMIT BOUND BOX AND LAYER|" %
            QString::number(Distance(min.x()).to(mm), 'f', 4) % m_laser_delimiter %
            QString::number(Distance(max.x()).to(mm), 'f', 4) % m_laser_delimiter %
            QString::number(Distance(min.y()).to(mm), 'f', 4) % m_laser_delimiter %
            QString::number(Distance(max.y()).to(mm), 'f', 4) % m_laser_delimiter %
            QString::number(layer) % m_laser_delimiter % QString::number(boundingBox) % m_laser_delimiter %
            QString::number((int)axis) % m_laser_delimiter % QString::number(round(angle.to(deg))));

        return rv;
    }

    QString CincinnatiWriter::writeBeforeRegion(RegionType type, int pathSize)
    {
        QString rv;
        if(!m_spiral_layer || m_first_print)
        {
            if (type == RegionType::kPerimeter)
            {
                rv += "M12 (PERIMETER SPINDLE ADJUSTMENT)\n";
            }
            else if (type == RegionType::kInset || type == RegionType::kSkeleton)
            {
                rv += "M13 (INSET SPINDLE ADJUSTMENT)\n";
            }
            else if (type == RegionType::kSkin)
            {
                rv += "M15 (SKIN SPINDLE ADJUSTMENT)\n";
            }
            else if (type == RegionType::kInfill)
            {
                rv += "M14 (INFILL SPINDLE ADJUSTMENT)\n";
            }
        }
        return rv;
    }

    QString CincinnatiWriter::writeBeforePath(RegionType type)
    {
        QString rv;
        if(!m_spiral_layer || m_first_print)
        {
            if (type == RegionType::kPerimeter)
            {
                if (!m_sb->setting< QString >(Constants::ProfileSettings::GCode::kPerimeterStart).isEmpty())
                    rv += m_sb->setting< QString >(Constants::ProfileSettings::GCode::kPerimeterStart) % m_newline;
                rv += writeAcceleration(m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kPerimeter));
            }
            else if (type == RegionType::kInset)
            {
                if (!m_sb->setting< QString >(Constants::ProfileSettings::GCode::kInsetStart).isEmpty())
                    rv += m_sb->setting< QString >(Constants::ProfileSettings::GCode::kInsetStart) % m_newline;
                rv += writeAcceleration(m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kInset));
            }
            else if(type == RegionType::kSkeleton)
            {
                if (!m_sb->setting< QString >(Constants::ProfileSettings::GCode::kSkeletonStart).isEmpty())
                    rv += m_sb->setting< QString >(Constants::ProfileSettings::GCode::kSkeletonStart) % m_newline;
                rv += writeAcceleration(m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kSkeleton));
            }
            else if (type == RegionType::kSkin)
            {
                if (!m_sb->setting< QString >(Constants::ProfileSettings::GCode::kSkinStart).isEmpty())
                    rv += m_sb->setting< QString >(Constants::ProfileSettings::GCode::kSkinStart) % m_newline;
                rv += writeAcceleration(m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kSkin));
            }
            else if (type == RegionType::kInfill)
            {
                if (!m_sb->setting< QString >(Constants::ProfileSettings::GCode::kInfillStart).isEmpty())
                    rv += m_sb->setting< QString >(Constants::ProfileSettings::GCode::kInfillStart) % m_newline;
                rv += writeAcceleration(m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kInfill));
            }
            else if (type == RegionType::kSupport)
            {
                if (!m_sb->setting< QString >(Constants::ProfileSettings::GCode::kSupportStart).isEmpty())
                    rv += m_sb->setting< QString >(Constants::ProfileSettings::GCode::kSupportStart) % m_newline;
                rv += writeAcceleration(m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kSupport));
            }
            else
            {
                rv += writeAcceleration(m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kDefault));
            }
        }
        return rv;
    }

    QString CincinnatiWriter::writeTravel(Point start_location, Point target_location, TravelLiftType lType,
                                          QSharedPointer<SettingsBase> params)
    {
        QString rv;

        Point new_start_location;
        RegionType rType = params->setting<RegionType>(Constants::SegmentSettings::kRegionType);
        bool w_active_first_travel = false;
        if(m_first_travel)
             w_active_first_travel = true;

        //Use updated start location if this is the first travel
        if(m_first_travel)
            new_start_location = m_start_point;
        else
            new_start_location = start_location;

        Distance liftDist;
        if(rType == RegionType::kLaserScan)
             liftDist =  m_sb->setting< Distance >(Constants::ProfileSettings::LaserScanner::kLaserScannerHeight) -
                     m_sb->setting<Distance>(Constants::ProfileSettings::LaserScanner::kLaserScannerHeightOffset);
        else
             liftDist = m_sb->setting< Distance >(Constants::ProfileSettings::Travel::kLiftHeight);

        bool travel_lift_required = liftDist > 0;// && !m_first_travel; //do not write a lift on first travel

        //Don't lift for short travel moves
        if(start_location.distance(target_location) < m_sb->setting< Distance >(Constants::ProfileSettings::Travel::kMinTravelForLift))
        {
            travel_lift_required = false;
        }

        //travel_lift vector in direction normal to the layer
        //with length = lift height as defined in settings
        QVector3D travel_lift = getTravelLift();

        //write the lift
        if (travel_lift_required && !m_first_travel && (lType == TravelLiftType::kBoth || lType == TravelLiftType::kLiftUpOnly))
        {
            Point lift_destination = new_start_location + travel_lift; //lift destination is above start location
            rv += m_G0 % writeCoordinates(lift_destination);
            if(m_w_travel)
            {
                rv += commentSpaceLine("TRAVEL LOWER W");
                setFeedrate(m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kWTableSpeed));
            }
            else
            {
               rv += commentSpaceLine("TRAVEL LIFT Z");
               setFeedrate(m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kZSpeed));
            }
        }
        /*else if(travel_lift_required && !m_first_travel && (lType == TravelLiftType::kBoth || lType == TravelLiftType::kLiftUpOnly))
        {
            Point lift_destination = new_start_location + travel_lift; //lift destination is above start location
            rv += m_G0 % writeCoordinates(lift_destination) % commentSpaceLine("TRAVEL LOWER W");
            setFeedrate(m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kZSpeed));
        }*/

        //write the travel
        Point travel_destination = target_location;
        if(m_first_travel)
            travel_destination.z(qAbs(m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kZOffset)()));
        else if (travel_lift_required)
            travel_destination = travel_destination + travel_lift; //travel destination is above the target point

        rv += m_G0 % writeCoordinates(travel_destination) % commentSpaceLine("TRAVEL");
        setFeedrate(m_sb->setting< Velocity >(Constants::ProfileSettings::Travel::kSpeed));

        if (m_first_travel) //if this is the first travel
            m_first_travel = false; //update for next one

        //write the travel lower (undo the lift)
        if (travel_lift_required && (lType == TravelLiftType::kBoth || lType == TravelLiftType::kLiftLowerOnly))
        {
            rv += m_G0 % writeCoordinates(target_location);//
            if(m_w_travel)
            {
                rv += commentSpaceLine("TRAVEL LIFT W");
                setFeedrate(m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kWTableSpeed));
            }
            else
            {
               rv += commentSpaceLine("TRAVEL LOWER Z");
               setFeedrate(m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kZSpeed));
            }


        }
        /*else if(travel_lift_required && (lType == TravelLiftType::kBoth || lType == TravelLiftType::kLiftLowerOnly))
        {
            rv += m_G0 % writeCoordinates(target_location) % commentSpaceLine("TRAVEL LIFT W");
            setFeedrate(m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kWTableSpeed));
        }*/

        if(w_active_first_travel && m_sb->setting<int>(Constants::PrinterSettings::Dimensions::kLayerChangeAxis) == static_cast<int>(LayerChange::kW_only))
        {
            //If using W only, a first travel to position the Z is required
            rv += m_G0 % m_z % QString::number(m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kZOffset).to(m_meta.m_distance_unit), 'f', 4)
                % commentSpaceLine("TRAVEL SET PRINTING Z HEIGHT");
            w_active_first_travel = false;
        }
        else if(w_active_first_travel && m_sb->setting<int>(Constants::PrinterSettings::Dimensions::kLayerChangeAxis) == static_cast<int>(LayerChange::kBoth_Z_and_W) && m_spiral_layer)
        {
            //If using ZW and Spiralize, a first travel to position the Z is required
            rv += m_G0 % m_z % QString::number(m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kZOffset).to(m_meta.m_distance_unit), 'f', 4)
                % commentSpaceLine("TRAVEL SET PRINTING Z HEIGHT");
            w_active_first_travel = false;
        }

        return rv;
    }

    QString CincinnatiWriter::writeLine(const Point& start_point, const Point& target_point, const QSharedPointer<SettingsBase> params)
    {
        Velocity speed = params->setting<Velocity>(Constants::SegmentSettings::kSpeed);
        int rpm = params->setting<int>(Constants::SegmentSettings::kExtruderSpeed);
        int material_number = params->setting<int>(Constants::SegmentSettings::kMaterialNumber);
        RegionType region_type = params->setting<RegionType>(Constants::SegmentSettings::kRegionType);
        PathModifiers path_modifiers = params->setting<PathModifiers>(Constants::SegmentSettings::kPathModifiers);
        float output_rpm = rpm * m_sb->setting< float >(Constants::PrinterSettings::MachineSpeed::kGearRatio);

        QString rv;        

        // Update the material number if needed
        if(material_number != m_material_number && m_sb->setting<int>(Constants::MaterialSettings::MultiMaterial::kEnable)
                && !m_sb->setting<bool>(Constants::ExperimentalSettings::MultiNozzle::kEnableMultiNozzleMultiMaterial))
        {
            if(m_sb->setting<int>(Constants::MaterialSettings::MultiMaterial::kUseM222))
                rv += "M222 P" % QString::number(material_number) % commentSpaceLine("CHANGE MATERIAL");
            else
                rv += "M237 L" % QString::number(material_number) % commentSpaceLine("CHANGE MATERIAL");
            m_material_number = material_number;
        }

        if(params->contains(Constants::SegmentSettings::kRecipe))
        {
            int recipe = params->setting<int>(Constants::SegmentSettings::kRecipe);
            if(m_current_recipe != recipe)
            {
                rv += "M222 P" % QString::number(recipe) % commentSpaceLine("CHANGE MATERIAL");
                m_current_recipe = recipe;
            }
        }

        for (int extruder : params->setting<QVector<int>>(Constants::SegmentSettings::kExtruders))
        {
            // Turn on the extruder if it isn't already on
            if (m_extruders_on[0] == false && rpm > 0) //only check first extruder
            {
                rv += writeExtruderOn(region_type, rpm, extruder);
                // Set Feedrate to 0 if turning extruder on so that an F parameter
                // is issued with the first G1 of the path
                setFeedrate(0);

            }
        }

        rv += m_G1;
        //update feedrate and speed if needed | forces first motion of layer to issue speed (needed for spiralize mode so that feedrate is scaled properly)
        if (getFeedrate() != speed || m_layer_start)
        {
            setFeedrate(speed);
            rv += m_f % QString::number(speed.to(m_meta.m_velocity_unit));
            m_layer_start = false;
        }

        if (rpm != m_current_rpm)
        {
            rv += m_s % QString::number(output_rpm);
            m_current_rpm = rpm;
        }

        //writes WXYZ to destination
        rv += writeCoordinates(target_point);

        //add comment for gcode parser
        if (path_modifiers != PathModifiers::kNone)
            rv += commentSpaceLine(toString(region_type) % m_space % toString(path_modifiers));
        else
            rv += commentSpaceLine(toString(region_type));

        m_first_print = false;

        return rv;
    }

    QString CincinnatiWriter::writeArc(const Point &start_point,
                                           const Point &end_point,
                                           const Point &center_point,
                                           const Angle &angle,
                                           const bool &ccw,
                                           const QSharedPointer<SettingsBase> params)
    {
        QString rv;

        Velocity speed = params->setting<Velocity>(Constants::SegmentSettings::kSpeed);
        int rpm = params->setting<int>(Constants::SegmentSettings::kExtruderSpeed);
        int material_number = params->setting<int>(Constants::SegmentSettings::kMaterialNumber);
        auto region_type = params->setting<RegionType>(Constants::SegmentSettings::kRegionType);
        auto path_modifiers = params->setting<PathModifiers>(Constants::SegmentSettings::kPathModifiers);
        float output_rpm = rpm * m_sb->setting< float >(Constants::PrinterSettings::MachineSpeed::kGearRatio);

        //update the material number if needed
        if(material_number != m_material_number && m_sb->setting<int>(Constants::MaterialSettings::MultiMaterial::kEnable)
                && !m_sb->setting<bool>(Constants::ExperimentalSettings::MultiNozzle::kEnableMultiNozzleMultiMaterial))
        {
            if(m_sb->setting<int>(Constants::MaterialSettings::MultiMaterial::kUseM222))
                rv += "M222 P" % QString::number(material_number) % commentSpaceLine("CHANGE MATERIAL");
            else
                rv += "M237 L" % QString::number(material_number) % commentSpaceLine("CHANGE MATERIAL");
            m_material_number = material_number;
        }

        if(params->contains(Constants::SegmentSettings::kRecipe))
        {
            int recipe = params->setting<int>(Constants::SegmentSettings::kRecipe);
            if(m_current_recipe != recipe)
            {
                rv += "M222 P" % QString::number(recipe) % commentSpaceLine("CHANGE MATERIAL");
                m_current_recipe = recipe;
            }
        }

        for (int extruder : params->setting<QVector<int>>(Constants::SegmentSettings::kExtruders))
        {
            //turn on the extruder if it isn't already on
            if (m_extruders_on[0] == false && rpm > 0) //only check first extruder
            {
                rv += writeExtruderOn(region_type, rpm, extruder);
            }
        }

        rv += ((ccw) ? m_G3 : m_G2);

        if (getFeedrate() != speed)
        {
            setFeedrate(speed);
            rv += m_f % QString::number(speed.to(m_meta.m_velocity_unit));
        }

        if (rpm != m_current_rpm)
        {
            rv += m_s % QString::number(output_rpm);
            m_current_rpm = rpm;
        }

        rv += m_i % QString::number(Distance(center_point.x() - start_point.x()).to(m_meta.m_distance_unit), 'f', 4) %
              m_j % QString::number(Distance(center_point.y() - start_point.y()).to(m_meta.m_distance_unit), 'f', 4) %
              m_x % QString::number(Distance(end_point.x()).to(m_meta.m_distance_unit), 'f', 4) %
              m_y % QString::number(Distance(end_point.y()).to(m_meta.m_distance_unit), 'f', 4) %
              getZWValue(end_point);

        // Add comment for gcode parser
        if (path_modifiers != PathModifiers::kNone)
            rv += commentSpaceLine(toString(region_type) % m_space % toString(path_modifiers));
        else
            rv += commentSpaceLine(toString(region_type));

        return rv;
    }

    QString CincinnatiWriter::writeSpline(const Point &start_point,
                                  const Point &a_control_point,
                                  const Point &b_control_point,
                                  const Point &end_point,
                                  const QSharedPointer<SettingsBase> params)
    {
        QString rv;


        Velocity speed = params->setting<Velocity>(Constants::SegmentSettings::kSpeed);
        int rpm = params->setting<int>(Constants::SegmentSettings::kExtruderSpeed);
        int material_number = params->setting<int>(Constants::SegmentSettings::kMaterialNumber);
        auto region_type = params->setting<RegionType>(Constants::SegmentSettings::kRegionType);
        auto path_modifiers = params->setting<PathModifiers>(Constants::SegmentSettings::kPathModifiers);
        float output_rpm = rpm * m_sb->setting< float >(Constants::PrinterSettings::MachineSpeed::kGearRatio);

        //update the material number if needed
        if(material_number != m_material_number && m_sb->setting<int>(Constants::MaterialSettings::MultiMaterial::kEnable)
                && !m_sb->setting<bool>(Constants::ExperimentalSettings::MultiNozzle::kEnableMultiNozzleMultiMaterial))
        {
            if(m_sb->setting<int>(Constants::MaterialSettings::MultiMaterial::kUseM222))
                rv += "M222 P" % QString::number(material_number) % commentSpaceLine("CHANGE MATERIAL");
            else
                rv += "M237 L" % QString::number(material_number) % commentSpaceLine("CHANGE MATERIAL");
            m_material_number = material_number;
        }

        if(params->contains(Constants::SegmentSettings::kRecipe))
        {
            int recipe = params->setting<int>(Constants::SegmentSettings::kRecipe);
            if(m_current_recipe != recipe)
            {
                rv += "M222 P" % QString::number(recipe) % commentSpaceLine("CHANGE MATERIAL");
                m_current_recipe = recipe;
            }
        }

        for (int extruder : params->setting<QVector<int>>(Constants::SegmentSettings::kExtruders))
        {
            //turn on the extruder if it isn't already on
            if (m_extruders_on[0] == false && rpm > 0) //only check first extruder
            {
                rv += writeExtruderOn(region_type, rpm, extruder);
            }
        }

        rv += m_G5;

        if (getFeedrate() != speed)
        {
            setFeedrate(speed);
            rv += m_f % QString::number(speed.to(m_meta.m_velocity_unit));
        }

        if (rpm != m_current_rpm)
        {
            rv += m_s % QString::number(output_rpm);
            m_current_rpm = rpm;
        }

        rv += m_i % QString::number(Distance(a_control_point.x() - start_point.x()).to(m_meta.m_distance_unit), 'f', 4) %
              m_j % QString::number(Distance(a_control_point.y() - start_point.y()).to(m_meta.m_distance_unit), 'f', 4) %
              m_p % QString::number(Distance(b_control_point.x() - end_point.x()).to(m_meta.m_distance_unit), 'f', 4) %
              m_q % QString::number(Distance(b_control_point.y() - end_point.y()).to(m_meta.m_distance_unit), 'f', 4) %
              m_x % QString::number(Distance(end_point.x()).to(m_meta.m_distance_unit), 'f', 4) %
              m_y % QString::number(Distance(end_point.y()).to(m_meta.m_distance_unit), 'f', 4) %
              getZWValue(end_point);

        // Add comment for gcode parser
        if (path_modifiers != PathModifiers::kNone)
            rv += commentSpaceLine(toString(region_type) % m_space % toString(path_modifiers));
        else
            rv += commentSpaceLine(toString(region_type));

        return rv;
    }

    QString CincinnatiWriter::writeScan(Point target_point, Velocity speed, bool on_off)
    {
        QString rv;
        //on
        if(on_off)
        {
            rv += m_laser_prefix % commentLine(Constants::RegionTypeStrings::kLaserScan % " - START|" %
                                               QString::number(m_sb->setting<Distance>(Constants::ProfileSettings::LaserScanner::kLaserScannerStepDistance).to(mm), 'f', 4));
        }
        else
        {
            rv += m_laser_prefix % commentLine(Constants::RegionTypeStrings::kLaserScan % " - STOP");
        }

        rv += m_G1 % m_x % QString::number(Distance(target_point.x()).to(m_meta.m_distance_unit), 'f', 4) %
                m_y % QString::number(Distance(target_point.y()).to(m_meta.m_distance_unit), 'f', 4) %
                commentSpaceLine(Constants::RegionTypeStrings::kLaserScan);

        return rv;
    }

    QString CincinnatiWriter::writeAfterPath(RegionType type)
    {
        QString rv;
        if(!m_spiral_layer)
        {
            rv += writeExtruderOff(0); //update to turn off appropriate extruders
            if (type == RegionType::kPerimeter)
            {
                if (!m_sb->setting< QString >(Constants::ProfileSettings::GCode::kPerimeterEnd).isEmpty())
                    rv += m_sb->setting< QString >(Constants::ProfileSettings::GCode::kPerimeterEnd) % m_newline;
            }
            else if (type == RegionType::kInset)
            {
                if (!m_sb->setting< QString >(Constants::ProfileSettings::GCode::kInsetEnd).isEmpty())
                    rv += m_sb->setting< QString >(Constants::ProfileSettings::GCode::kInsetEnd) % m_newline;
            }
            else if (type == RegionType::kSkeleton)
            {
                if (!m_sb->setting< QString >(Constants::ProfileSettings::GCode::kSkeletonEnd).isEmpty())
                    rv += m_sb->setting< QString >(Constants::ProfileSettings::GCode::kSkeletonEnd) % m_newline;
            }
            else if (type == RegionType::kSkin)
            {
                if (!m_sb->setting< QString >(Constants::ProfileSettings::GCode::kSkinEnd).isEmpty())
                    rv += m_sb->setting< QString >(Constants::ProfileSettings::GCode::kSkinEnd) % m_newline;
            }
            else if (type == RegionType::kInfill)
            {
                if (!m_sb->setting< QString >(Constants::ProfileSettings::GCode::kInfillEnd).isEmpty())
                    rv += m_sb->setting< QString >(Constants::ProfileSettings::GCode::kInfillEnd) % m_newline;
            }
            else if (type == RegionType::kSupport)
            {
                if (!m_sb->setting< QString >(Constants::ProfileSettings::GCode::kSupportEnd).isEmpty())
                    rv += m_sb->setting< QString >(Constants::ProfileSettings::GCode::kSupportEnd) % m_newline;
            }
        }
        return rv;
    }

    QString CincinnatiWriter::writeAfterRegion(RegionType type)
    {
        QString rv;
        return rv;
    }

    QString CincinnatiWriter::writeAfterScan(Distance beadWidth, Distance laserStep, Distance laserResolution)
    {
        int xDistance = round((beadWidth.to(mm) / laserStep.to(mm)));
        int yDistance = round((beadWidth.to(mm) / laserResolution.to(mm)));

        return m_laser_prefix % commentLine(Constants::RegionTypeStrings::kLaserScan % " - STOP") %
               m_laser_prefix % commentLine(Constants::RegionTypeStrings::kLaserScan % " - PROCESS PATCH|" %
                                           QString::number(xDistance) % m_laser_delimiter % QString::number(yDistance));
    }

    QString CincinnatiWriter::writeAfterIsland()
    {
        QString rv;
        return rv;
    }

    QString CincinnatiWriter::writeAfterPart()
    {
        QString rv;
        return rv;
    }

    QString CincinnatiWriter::writeAfterLayer()
    {
        QString rv;
        if(m_sb->setting<bool>(Constants::ProfileSettings::LaserScanner::kLaserScanner))
            rv += m_laser_prefix % commentLine(Constants::RegionTypeStrings::kLaserScan % " - PROCESS LAYER");

        rv += m_sb->setting< QString >(Constants::PrinterSettings::GCode::kLayerCodeChange) % m_newline;
        return rv;
    }

    QString CincinnatiWriter::writeShutdown()
    {
        QString rv;
        rv += m_M5 % commentSpaceLine("TURN EXTRUDER OFF END OF PRINT") %
              writeTamperOff() %
              "M68 (PARK)\n";

        if (m_sb->setting< int >(Constants::MaterialSettings::Extruder::kServoToTravelSpeed)){
            rv += m_M11 % commentSpaceLine("TURN OFF EXTRUDER SERVOING");
        }
        if (m_sb->setting< int >(Constants::PrinterSettings::Dimensions::kEnableDoffing) && m_current_w > m_sb->setting< int >(Constants::PrinterSettings::Dimensions::kDoffingHeight)){
            rv += m_G1 % m_w %
                  QString::number(m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kDoffingHeight).to(m_meta.m_distance_unit))
                  % commentSpaceLine("JOG W TO DOFFING LOCATION");
        }
        rv += m_sb->setting< QString >(Constants::PrinterSettings::GCode::kEndCode) % m_newline %
              "M30" % commentSpaceLine("END OF G-CODE");
        return rv;
    }

    QString CincinnatiWriter::writePurge(int RPM, int duration, int delay)
    {
        return "M69 F" % QString::number(RPM) % m_p % QString::number(duration) % m_s % QString::number(delay) % commentSpaceLine("PURGE");
    }

    QString CincinnatiWriter::writeDwell(Time time)
    {
        if (time > 0)
            return m_G4 % m_p % QString::number(time.to(m_meta.m_time_unit), 'f', 4) % commentSpaceLine("DWELL");
        else
            return {};
    }

    QString CincinnatiWriter::writeTamperOn()
    {
        if (m_sb->setting< int >(Constants::PrinterSettings::Auxiliary::kEnableTamper))
        {
            return m_M64 % m_l % QString::number(m_sb->setting< Voltage >(Constants::PrinterSettings::Auxiliary::kTamperVoltage).to(V))
                % commentSpaceLine("TURN TAMPER ON");
        }
        else
        {
            return {};
        }
    }

    QString CincinnatiWriter::writeTamperOff()
    {
        if (m_sb->setting< int >(Constants::PrinterSettings::Auxiliary::kEnableTamper))
        {
            return m_M65 % commentSpaceLine("TURN TAMPER OFF");
        }
        else
        {
            return {};
        }
    }

    QString CincinnatiWriter::writeExtruderOn(RegionType type, int rpm, int extruder_number)
    {
        QString rv;
        m_extruders_on[extruder_number] = true;
        float output_rpm;

        rv += writeTamperOn();

        if (m_sb->setting< int >(Constants::MaterialSettings::Extruder::kInitialSpeed) > 0)
        {
            // Check settings, turn off extruder servoing, will turn back on at end
            if (m_sb->setting< int >(Constants::MaterialSettings::Extruder::kServoToTravelSpeed))
                rv += m_M11 % m_space % commentLine("TURN OFF EXTRUDER SERVOING");

            output_rpm = m_sb->setting< float >(Constants::PrinterSettings::MachineSpeed::kGearRatio) * m_sb->setting< int >(Constants::MaterialSettings::Extruder::kInitialSpeed);

            // Only update the current rpm if not using feedrate scaling. An updated rpm value here could prevent the S parameter
            // from being issued during the first G1 motion of the path and thus the extruder rate won't properly scale
            if(!(m_sb->setting< int >(Constants::MaterialSettings::Cooling::kForceMinLayerTime) &&
                 m_sb->setting< int >(Constants::MaterialSettings::Cooling::kForceMinLayerTimeMethod) == (int)ForceMinimumLayerTime::kSlow_Feedrate))
                m_current_rpm = m_sb->setting< int >(Constants::MaterialSettings::Extruder::kInitialSpeed);

            rv += m_M3 % m_s % QString::number(output_rpm) % commentSpaceLine("TURN EXTRUDER ON");

            if (type == RegionType::kInset)
            {
                if(m_sb->setting< Time >(Constants::MaterialSettings::Extruder::kOnDelayInset) > 0)
                    rv += writeDwell(m_sb->setting< Time >(Constants::MaterialSettings::Extruder::kOnDelayInset));
            }
            else if (type == RegionType::kSkin)
            {
                if(m_sb->setting< Time >(Constants::MaterialSettings::Extruder::kOnDelaySkin) > 0)
                    rv += writeDwell(m_sb->setting< Time >(Constants::MaterialSettings::Extruder::kOnDelaySkin));
            }
            else if (type == RegionType::kInfill)
            {
                if(m_sb->setting< Time >(Constants::MaterialSettings::Extruder::kOnDelayInfill) > 0)
                    rv += writeDwell(m_sb->setting< Time >(Constants::MaterialSettings::Extruder::kOnDelayInfill));
            }
            else if (type == RegionType::kSkeleton)
            {
                if(m_sb->setting< Time >(Constants::MaterialSettings::Extruder::kOnDelaySkeleton) > 0)
                    rv += writeDwell(m_sb->setting< Time >(Constants::MaterialSettings::Extruder::kOnDelaySkeleton));
            }
            else
            {
                if(m_sb->setting< Time >(Constants::MaterialSettings::Extruder::kOnDelayPerimeter) > 0)
                    rv += writeDwell(m_sb->setting< Time >(Constants::MaterialSettings::Extruder::kOnDelayPerimeter));
            }
        }
        else
        {
            output_rpm = m_sb->setting< float >(Constants::PrinterSettings::MachineSpeed::kGearRatio) * rpm;
            rv += m_M3 % m_s % QString::number(output_rpm) % commentSpaceLine("TURN EXTRUDER ON");

            // Only update the current rpm if not using feedrate scaling. An updated rpm value here could prevent the S parameter
            // from being issued during the first G1 motion of the path and thus the extruder rate won't properly scale
            if(!(m_sb->setting< int >(Constants::MaterialSettings::Cooling::kForceMinLayerTime) &&
                 m_sb->setting< int >(Constants::MaterialSettings::Cooling::kForceMinLayerTimeMethod) == (int)ForceMinimumLayerTime::kSlow_Feedrate))
                m_current_rpm = rpm;
        }

        // Check settings, turn on servoing if it was turned off at beginning
        if (m_sb->setting< int >(Constants::MaterialSettings::Extruder::kServoToTravelSpeed))
            rv += m_M10 % m_space % commentLine("TURN ON EXTRUDER SERVOING");

        return rv;
    }

    QString CincinnatiWriter::writeExtruderOff(int extruder_number)
    {
        //update to use extruder number

        QString rv;
        m_extruders_on[extruder_number] = false;
        if(m_sb->setting< Time >(Constants::MaterialSettings::Extruder::kOffDelay) > 0)
        {
            rv += writeDwell(m_sb->setting< Time >(Constants::MaterialSettings::Extruder::kOffDelay));
        }
        rv += writeTamperOff() % m_M5 % commentSpaceLine("TURN EXTRUDER OFF");
        m_current_rpm = 0;
        return rv;
    }

    QString CincinnatiWriter::writeAcceleration(Acceleration acc)
    {
        float ci_acc = acc.to(m_meta.m_acceleration_unit) / 386.08858; // Convert to units of G
        ci_acc = (1 / ci_acc * 1000000 * 25.4 / 162560) / (1000 * 9.81);
        return "M66 L" % QString::number(ci_acc, 'g', 4) % commentSpaceLine("SET ACCELERATION");
        return "";
    }

    QString CincinnatiWriter::writeCoordinates(Point destination)
    {
        QString rv;
        m_z_travel = false;
        m_w_travel = false;

        //always specify X and Y
        rv += m_x % QString::number(Distance(destination.x()).to(m_meta.m_distance_unit), 'f', 4) %
              m_y % QString::number(Distance(destination.y()).to(m_meta.m_distance_unit), 'f', 4) %
              getZWValue(destination);

        return rv;
    }

    QString CincinnatiWriter::getZWValue(const Point& destination)
    {
        QString rv;
        //write vertical coordinate along the correct axis (Z or W) according to printer settings
        //only output Z/W coordinate if there was a change in Z/W
        Distance z_offset = m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kZOffset);
        if(m_sb->setting< int >(Constants::PrinterSettings::Dimensions::kLayerChangeAxis) == static_cast<int>(LayerChange::kZ_only))
        {
            //move in Z only
            Distance target_z = destination.z() + z_offset;
            if(qAbs(target_z - m_last_z) > 10)
            {
                rv += m_z % QString::number(Distance(target_z).to(m_meta.m_distance_unit), 'f', 4);
                m_current_z = target_z;
                m_last_z = target_z;
                m_z_travel = true;
            }
        }
        else if (m_sb->setting< int >(Constants::PrinterSettings::Dimensions::kLayerChangeAxis) == static_cast<int>(LayerChange::kW_only))
        {
            //move in W only
            Distance target_w = destination.z() * -1.0;
            if(qAbs(target_w - m_last_w) > 10 && !m_first_travel)
            {
                rv += m_w % QString::number(Distance(target_w).to(m_meta.m_distance_unit), 'f', 4);
                m_current_w = target_w;
                m_last_w = target_w;
                m_w_travel = true;
            }
        }
        else //Both Z and W
        {
            if(m_spiral_layer)
            {
                if(m_first_travel)
                {
                    //use Z for first travels to lower extruder down - bed is raised to top by the initial setup
                    Distance target_z = destination.z() + z_offset + m_current_w;
                    if(qAbs(target_z - m_last_z) > 10)
                    {
                        rv += m_z % QString::number(Distance(target_z).to(m_meta.m_distance_unit), 'f', 4);
                        m_current_z = target_z;
                        m_last_z = target_z;
                        m_z_travel = true;
                    }
                }
                else
                {
                    //when in spiralize mode, use W for all print paths until W is maxed out then transition to Z
                    Distance target_w = m_sb->setting<Distance>(Constants::PrinterSettings::Dimensions::kWMax) - destination.z();
                    if(target_w < m_sb->setting<Distance>(Constants::PrinterSettings::Dimensions::kWMin))
                    {
                        Distance target_z = destination.z() + z_offset + m_current_w;
                        rv += m_z % QString::number(Distance(target_z).to(m_meta.m_distance_unit), 'f', 4);
                        m_current_z = target_z;
                        m_last_z = target_z;
                        m_z_travel = true;
                    }
                    else
                    {
                        m_current_w = target_w;
                        m_last_w = target_w;
                        rv += m_w % QString::number(Distance(target_w).to(m_meta.m_distance_unit), 'f', 4);
                        m_w_travel = true;
                    }
                }
            }
            else if(m_first_travel)
            {
                rv += m_z % QString::number(m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kZMax).to(m_meta.m_distance_unit), 'f', 4);
            }
            else
            {
                //when a printer has both Z and W axis, use Z for a layer's segments and travels. W is used only for shifts between layers
                Distance target_z = destination.z() + z_offset + m_current_w;
                if(qAbs(target_z - m_last_z) > 10)
                {
                    rv += m_z % QString::number(Distance(target_z).to(m_meta.m_distance_unit), 'f', 4);
                    m_current_z = target_z;
                    m_last_z = target_z;
                    m_z_travel = true;
                }
            }
        }
        return rv;
    }

}  // namespace ORNL
