#include "gcode/writers/mach4_writer.h"
#include <QStringBuilder>
#include "utilities/enums.h"
#include "utilities/mathutils.h"

namespace ORNL
{
    Mach4Writer::Mach4Writer(GcodeMeta meta, const QSharedPointer<SettingsBase>& sb) : WriterBase(meta, sb) {

    }

    QString Mach4Writer::writeInitialSetup(Distance minimum_x, Distance minimum_y, Distance maximum_x, Distance maximum_y, int num_layers)
    {
        m_current_z = m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kZOffset);
        m_filament_location = 0.0;
        for (int i = 0, end = m_extruders_on.size(); i < end; ++i) //all extruders initially off
            m_extruders_on[i] = false;
        m_first_print = true;
        m_first_travel = true;
        m_layer_start = true;
        m_min_z = 0.0f;
        QString rv;
        if (m_sb->setting< int >(Constants::PrinterSettings::GCode::kEnableStartupCode))
        {
            rv += "M140 P" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kBed).to(degC))
                    % commentSpaceLine("SET BED TEMPERATURE");
            if (m_sb->setting< int >(Constants::MaterialSettings::Temperatures::kBed) > 0)
            {
                rv += "M190 P" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kBed).to(degC))
                        % commentSpaceLine("SET BED TEMPERATURE AND WAIT");
            }

            rv += "M104 P" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder0).to(degC))
                        % commentSpaceLine("SET EXTRUDER TEMPERATURE");
            if (m_sb->setting< int >(Constants::MaterialSettings::Temperatures::kBed) > 0)
            {
                rv += "M109 P" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder0).to(degC))
                        % commentSpaceLine("SET EXTRUDER 0 TEMPERATURE AND WAIT");
            }

            rv += "G21" % commentSpaceLine("SET UNITS TO MILLIMETERS");

            if (m_sb->setting< int >(Constants::MaterialSettings::Filament::kRelative) > 0)
            {
                rv += "G91" % commentSpaceLine("USE RELATIVE EXTRUSION DISTANCES");
            }
            else
            {
                rv += "G90" % commentSpaceLine("USE ABSOLUTE EXTRUSION DISTANCES");
            }

            rv += "G92 A0" % commentSpaceLine("RESET FILAMENT AXIS TO 0");

            // Cooling fan
            if(m_sb->setting< bool >(Constants::MaterialSettings::Cooling::kEnable))
                // Marlin index the fan from 0 to 255
                rv += "M106 S" % QString::number((m_sb->setting<int>(Constants::MaterialSettings::Cooling::kMaxSpeed) / 100) * 255) % commentSpaceLine("ENABLE FAN");
            else
                rv += "M107" % commentSpaceLine("DISABLE COOLING FAN");

            rv += m_newline;
        }

        if(m_sb->setting< int >(Constants::PrinterSettings::GCode::kEnableBoundingBox))
        {
            rv += m_G1 % m_x % QString::number(minimum_x.to(m_meta.m_distance_unit), 'f', 4) % " Y" % QString::number(minimum_y.to(m_meta.m_distance_unit), 'f', 4) % commentSpaceLine("BOUNDING BOX")
                % m_G1 % m_x % QString::number(maximum_x.to(m_meta.m_distance_unit), 'f', 4) % " Y" % QString::number(minimum_y.to(m_meta.m_distance_unit), 'f', 4) % commentSpaceLine("BOUNDING BOX")
                % m_G1 % m_x % QString::number(maximum_x.to(m_meta.m_distance_unit), 'f', 4) % " Y" % QString::number(maximum_y.to(m_meta.m_distance_unit), 'f', 4) % commentSpaceLine("BOUNDING BOX")
                % m_G1 % m_x % QString::number(minimum_x.to(m_meta.m_distance_unit), 'f', 4) % " Y" % QString::number(maximum_y.to(m_meta.m_distance_unit), 'f', 4) % commentSpaceLine("BOUNDING BOX")
                % m_G1 % m_x % QString::number(minimum_x.to(m_meta.m_distance_unit), 'f', 4) % " Y" % QString::number(minimum_y.to(m_meta.m_distance_unit), 'f', 4) % commentSpaceLine("BOUNDING BOX");

            m_start_point = Point(minimum_x, minimum_y, 0);
        }

        if(m_sb->setting< QString >(Constants::PrinterSettings::GCode::kStartCode) != "")
            rv += m_sb->setting< QString >(Constants::PrinterSettings::GCode::kStartCode);

        rv += m_newline;

        rv += commentLine(m_meta.m_layer_count_delimiter % ":" % QString::number(num_layers));

        return rv;
    }

    QString Mach4Writer::writeBeforeLayer(float new_min_z, QSharedPointer<SettingsBase> sb)
    {
        m_spiral_layer = sb->setting<bool>(Constants::ProfileSettings::SpecialModes::kEnableSpiralize);
        m_layer_start = true;
        QString rv;
        if(m_sb->setting< bool >(Constants::MaterialSettings::Retraction::kEnable) &&
                m_sb->setting< bool >(Constants::MaterialSettings::Retraction::kLayerChange) &&
                new_min_z > sb->setting< Distance >(Constants::ProfileSettings::Layer::kLayerHeight))
        {
            rv += writeRetraction();
        }
        return rv;
    }

    QString Mach4Writer::writeBeforePart(QVector3D normal)
    {
        QString rv;
        return rv;
    }

    QString Mach4Writer::writeBeforeIsland()
    {
        QString rv;
        return rv;
    }

    QString Mach4Writer::writeBeforeRegion(RegionType type, int pathSize)
    {
        QString rv;
        return rv;
    }

    QString Mach4Writer::writeBeforePath(RegionType type)
    {
        QString rv;
        if(!m_spiral_layer || m_first_print)
        {
            if (type == RegionType::kPerimeter)
            {
                if (!m_sb->setting< QString >(Constants::ProfileSettings::GCode::kPerimeterStart).isEmpty())
                    rv += m_sb->setting< QString >(Constants::ProfileSettings::GCode::kPerimeterStart) % m_newline;
            }
            else if (type == RegionType::kInset)
            {
                if (!m_sb->setting< QString >(Constants::ProfileSettings::GCode::kInsetStart).isEmpty())
                    rv += m_sb->setting< QString >(Constants::ProfileSettings::GCode::kInsetStart) % m_newline;
            }
            else if(type == RegionType::kSkeleton)
            {
                if (!m_sb->setting< QString >(Constants::ProfileSettings::GCode::kSkeletonStart).isEmpty())
                    rv += m_sb->setting< QString >(Constants::ProfileSettings::GCode::kSkeletonStart) % m_newline;
            }
            else if (type == RegionType::kSkin)
            {
                if (!m_sb->setting< QString >(Constants::ProfileSettings::GCode::kSkinStart).isEmpty())
                    rv += m_sb->setting< QString >(Constants::ProfileSettings::GCode::kSkinStart) % m_newline;
            }
            else if (type == RegionType::kInfill)
            {
                if (!m_sb->setting< QString >(Constants::ProfileSettings::GCode::kInfillStart).isEmpty())
                    rv += m_sb->setting< QString >(Constants::ProfileSettings::GCode::kInfillStart) % m_newline;
            }
            else if (type == RegionType::kSupport)
            {
                if (!m_sb->setting< QString >(Constants::ProfileSettings::GCode::kSupportStart).isEmpty())
                    rv += m_sb->setting< QString >(Constants::ProfileSettings::GCode::kSupportStart) % m_newline;
            }
        }
        if (m_filament_location > 0)
        {
            rv += "G92 A0" % commentSpaceLine("RESET FILAMENT TO 0");
            m_filament_location = 0.0;
        }

        return rv;
    }

    QString Mach4Writer::writeTravel(Point start_location, Point target_location, TravelLiftType lType,
                                      QSharedPointer<SettingsBase> params)
    {
        QString rv;
        Velocity speed = params->setting<Velocity>(Constants::SegmentSettings::kSpeed);
        Point new_start_location;

        setTools(QVector<int>()); //turn off all extruders

        //Update Acceleration
        if(m_sb->setting< bool >(Constants::PrinterSettings::Acceleration::kEnableDynamic))
        {
            rv += "M204 S" % QString::number(m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kDefault).to(m_meta.m_acceleration_unit)) %
                    commentSpaceLine("UPDATE ACCELERATION");
        }

        //Use updated start location if this is the first travel
        if(m_first_travel)
            new_start_location = m_start_point;
        else
            new_start_location = start_location;

        Distance liftDist = m_sb->setting< Distance >(Constants::ProfileSettings::Travel::kLiftHeight);

        bool travel_lift_required = liftDist > 0;// && !m_first_travel; //do not write a lift on first travel

        //Don't lift for short travel moves
        if(start_location.distance(target_location) < m_sb->setting< Distance >(Constants::ProfileSettings::Travel::kMinTravelForLift))
        {
            travel_lift_required = false;
        }

        //Retract on long travel moves
        if(start_location.distance(target_location) > m_sb->setting< Distance >(Constants::MaterialSettings::Retraction::kMinTravel) &&
                (params->setting<bool>(Constants::SegmentSettings::kIsRegionStartSegment) ||
                 !m_sb->setting<bool>(Constants::MaterialSettings::Retraction::kOpenSpacesOnly)))
        {
            rv += writeRetraction();
        }

        //travel_lift vector in direction normal to the layer
        //with length = lift height as defined in settings
        QVector3D travel_lift = getTravelLift();

        //write the lift
        if (travel_lift_required && (lType == TravelLiftType::kBoth || lType == TravelLiftType::kLiftUpOnly))
        {
            Point lift_destination = new_start_location + travel_lift; //lift destination is above start location
            rv += m_G1 % m_f % QString::number(m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kZSpeed).to(m_meta.m_velocity_unit))
                    % writeCoordinates(lift_destination) % commentSpaceLine("TRAVEL LIFT Z");
            setFeedrate(m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kZSpeed));
        }

        //write the travel
        Point travel_destination = target_location;
        if (travel_lift_required)
            travel_destination = travel_destination + travel_lift; //travel destination is above the target point

        rv += m_G1 % m_f % QString::number(speed.to(m_meta.m_velocity_unit))
                % writeCoordinates(travel_destination) % commentSpaceLine("TRAVEL");
        setFeedrate(speed);

        //write the travel lower (undo the lift)
        if (travel_lift_required && (lType == TravelLiftType::kBoth || lType == TravelLiftType::kLiftLowerOnly))
        {
            rv += m_G1 % m_f % QString::number(m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kZSpeed).to(m_meta.m_velocity_unit))
                    % writeCoordinates(target_location) % commentSpaceLine("TRAVEL LOWER Z");
            setFeedrate(m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kZSpeed));
        }

        m_first_travel = false;
        return rv;
    }

    QString Mach4Writer::writeLine(const Point& start_point, const Point& target_point, const QSharedPointer<SettingsBase> params)
    {
        Velocity speed = params->setting<Velocity>(Constants::SegmentSettings::kSpeed);
        RegionType region_type = params->setting<RegionType>(Constants::SegmentSettings::kRegionType);
        PathModifiers path_modifiers = params->setting<PathModifiers>(Constants::SegmentSettings::kPathModifiers);

        QString rv;

        //check if any extruders need priming
        bool needs_prime = false;
        for (int ext : params->setting<QVector<int>>(Constants::SegmentSettings::kExtruders))
            needs_prime = !m_extruders_on[ext] || needs_prime;

        //set the tools/extruders before priming so that correct extruders get primed
        rv += setTools(params->setting<QVector<int>>(Constants::SegmentSettings::kExtruders));

        //If first printing segment, prime extruder, or at least undo any retraction, and update acceleration
        //First segment of the path is signified by extruder being off and the modifier isn't one of five ending modifiers
        if(needs_prime && path_modifiers != PathModifiers::kSlowDown &&
                path_modifiers != PathModifiers::kForwardTipWipe && path_modifiers != PathModifiers::kReverseTipWipe &&
                path_modifiers != PathModifiers::kCoasting && path_modifiers != PathModifiers::kSpiralLift)
        {
            if(m_sb->setting< bool >(Constants::PrinterSettings::Acceleration::kEnableDynamic))
            {
                if (region_type == RegionType::kPerimeter)
                {
                    rv += "M204 S" % QString::number(m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kPerimeter).to(m_meta.m_acceleration_unit)) %
                            commentSpaceLine("UPDATE ACCELERATION");
                }
                else if (region_type == RegionType::kInset)
                {
                    rv += "M204 S" % QString::number(m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kInset).to(m_meta.m_acceleration_unit)) %
                            commentSpaceLine("UPDATE ACCELERATION");
                }
                else if(region_type == RegionType::kSkeleton)
                {
                    rv += "M204 S" % QString::number(m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kSkeleton).to(m_meta.m_acceleration_unit)) %
                            commentSpaceLine("UPDATE ACCELERATION");
                }
                else if (region_type == RegionType::kSkin)
                {
                    rv += "M204 S" % QString::number(m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kSkin).to(m_meta.m_acceleration_unit)) %
                            commentSpaceLine("UPDATE ACCELERATION");
                }
                else if (region_type == RegionType::kInfill)
                {
                    rv += "M204 S" % QString::number(m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kInfill).to(m_meta.m_acceleration_unit)) %
                            commentSpaceLine("UPDATE ACCELERATION");
                }
                else if (region_type == RegionType::kSupport)
                {
                    rv += "M204 S" % QString::number(m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kSupport).to(m_meta.m_acceleration_unit)) %
                            commentSpaceLine("UPDATE ACCELERATION");
                }
                else
                {
                    rv += "M204 S" % QString::number(m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kDefault).to(m_meta.m_acceleration_unit)) %
                            commentSpaceLine("UPDATE ACCELERATION");
                }
            }

            if(m_filament_location < 0)
                rv += writePrime();
        }

        rv += m_G1;
        //update feedrate if needed
        if (getFeedrate() != speed || m_layer_start)
        {
            setFeedrate(speed);
            rv += m_f % QString::number(speed.to(m_meta.m_velocity_unit));
            m_layer_start = false;
        }

        //writes XYZ to destination
        rv += writeCoordinates(target_point);

        //calculate and write E value if path is an extrusion path
        if (path_modifiers != PathModifiers::kCoasting && path_modifiers != PathModifiers::kForwardTipWipe
                && path_modifiers != PathModifiers::kPerimeterTipWipe && path_modifiers != PathModifiers::kReverseTipWipe
                && path_modifiers != PathModifiers::kSpiralLift)
        {
            //Set extrusion multiplier, or use default value of 1.0
            double current_multiplier;
            if(region_type == RegionType::kPerimeter)
                current_multiplier = m_sb->setting<double>(Constants::ProfileSettings::Perimeter::kExtrusionMultiplier);
            else if(region_type == RegionType::kInset)
                current_multiplier = m_sb->setting<double>(Constants::ProfileSettings::Inset::kExtrusionMultiplier);
            else if(region_type == RegionType::kSkeleton)
                current_multiplier = m_sb->setting<double>(Constants::ProfileSettings::Skeleton::kExtrusionMultiplier);
            else if(region_type == RegionType::kSkin)
                current_multiplier = m_sb->setting<double>(Constants::ProfileSettings::Skin::kExtrusionMultiplier);
            else if(region_type == RegionType::kInfill)
                current_multiplier = m_sb->setting<double>(Constants::ProfileSettings::Infill::kExtrusionMultiplier);
            else
                current_multiplier = m_sb->setting<double>(Constants::ProfileSettings::Perimeter::kExtrusionMultiplier);

            Distance segment_length = start_point.distance(target_point);
            Distance width = params->setting<Distance>(Constants::SegmentSettings::kWidth);
            Distance height = params->setting<Distance>(Constants::SegmentSettings::kHeight);
            Distance filament_diameter = m_sb->setting<Distance>(Constants::MaterialSettings::Filament::kDiameter);
            Distance segment_filament_length = (segment_length * width * height) / ((filament_diameter / 2) * (filament_diameter / 2) * 3.14159);
            segment_filament_length *= current_multiplier;

            //if using relative E, write out segment filament length and be done
            if (m_sb->setting< bool >(Constants::MaterialSettings::Filament::kRelative))
            {
                m_filament_location = segment_filament_length;
                rv += m_a % QString::number(segment_filament_length.to(m_meta.m_distance_unit));
            }
            //if using absolute E, update total length and write value
            else
            {
                m_filament_location += segment_filament_length;
                rv += m_a % QString::number(m_filament_location.to(m_meta.m_distance_unit));
            }

        }

        //add comment for gcode parser
        if (path_modifiers != PathModifiers::kNone)
            rv += commentSpaceLine(toString(region_type) % m_space % toString(path_modifiers));
        else
            rv += commentSpaceLine(toString(region_type));

        m_first_print = false;

        return rv;
    }

    QString Mach4Writer::writeArc(const Point &start_point,
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

        //check if any extruders need priming
        bool needs_prime = false;
        for (int ext : params->setting<QVector<int>>(Constants::SegmentSettings::kExtruders))
            needs_prime = !m_extruders_on[ext] || needs_prime;

        //set the tools/extruders before priming so that correct extruders get primed
        rv += setTools(params->setting<QVector<int>>(Constants::SegmentSettings::kExtruders));

        //If first printing segment, prime extruder, or at least undo any retraction, and update acceleration
        //First segment of the path is signified by extruder being off and the modifier isn't one of five ending modifiers
        if(needs_prime && path_modifiers != PathModifiers::kSlowDown &&
                path_modifiers != PathModifiers::kForwardTipWipe && path_modifiers != PathModifiers::kReverseTipWipe &&
                path_modifiers != PathModifiers::kCoasting && path_modifiers != PathModifiers::kSpiralLift)
        {
            if(m_sb->setting< bool >(Constants::PrinterSettings::Acceleration::kEnableDynamic))
            {
                if (region_type == RegionType::kPerimeter)
                {
                    rv += "M204 S" % QString::number(m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kPerimeter).to(m_meta.m_acceleration_unit)) %
                            commentSpaceLine("UPDATE ACCELERATION");
                }
                else if (region_type == RegionType::kInset)
                {
                    rv += "M204 S" % QString::number(m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kInset).to(m_meta.m_acceleration_unit)) %
                            commentSpaceLine("UPDATE ACCELERATION");
                }
                else if(region_type == RegionType::kSkeleton)
                {
                    rv += "M204 S" % QString::number(m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kSkeleton).to(m_meta.m_acceleration_unit)) %
                            commentSpaceLine("UPDATE ACCELERATION");
                }
                else if (region_type == RegionType::kSkin)
                {
                    rv += "M204 S" % QString::number(m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kSkin).to(m_meta.m_acceleration_unit)) %
                            commentSpaceLine("UPDATE ACCELERATION");
                }
                else if (region_type == RegionType::kInfill)
                {
                    rv += "M204 S" % QString::number(m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kInfill).to(m_meta.m_acceleration_unit)) %
                            commentSpaceLine("UPDATE ACCELERATION");
                }
                else if (region_type == RegionType::kSupport)
                {
                    rv += "M204 S" % QString::number(m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kSupport).to(m_meta.m_acceleration_unit)) %
                            commentSpaceLine("UPDATE ACCELERATION");
                }
                else
                {
                    rv += "M204 S" % QString::number(m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kDefault).to(m_meta.m_acceleration_unit)) %
                            commentSpaceLine("UPDATE ACCELERATION");
                }
            }

            if(m_filament_location < 0)
                rv += writePrime();
        }

        rv += ((ccw) ? m_G3 : m_G2);

        //update feedrate and speed if needed
        if (getFeedrate() != speed)
        {
            setFeedrate(speed);
            rv += m_f % QString::number(speed.to(m_meta.m_velocity_unit));
        }

        rv += m_i % QString::number(Distance(center_point.x() - start_point.x()).to(m_meta.m_distance_unit), 'f', 4) %
              m_j % QString::number(Distance(center_point.y() - start_point.y()).to(m_meta.m_distance_unit), 'f', 4) %
              m_x % QString::number(Distance(end_point.x()).to(m_meta.m_distance_unit), 'f', 4) %
              m_y % QString::number(Distance(end_point.y()).to(m_meta.m_distance_unit), 'f', 4);

        // write vertical coordinate along the correct axis (Z or W) according to printer settings
        // only output Z/W coordinate if there was a change in Z/W
        Distance z_offset = m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kZOffset);

        Distance target_z = end_point.z() + z_offset;
        if(qAbs(target_z - m_last_z) > 10)
        {
            rv += m_z % QString::number(Distance(target_z).to(m_meta.m_distance_unit), 'f', 4);
            m_current_z = target_z;
            m_last_z = target_z;
        }

        //calculate and write E value if path is an extrusion path
        if (path_modifiers != PathModifiers::kCoasting && path_modifiers != PathModifiers::kForwardTipWipe
                && path_modifiers != PathModifiers::kPerimeterTipWipe && path_modifiers != PathModifiers::kReverseTipWipe
                && path_modifiers != PathModifiers::kSpiralLift)
        {
            //Set extrusion multiplier, or use default value of 1.0
            double current_multiplier;
            if(region_type == RegionType::kPerimeter)
                current_multiplier = m_sb->setting<double>(Constants::ProfileSettings::Perimeter::kExtrusionMultiplier);
            else if(region_type == RegionType::kInset)
                current_multiplier = m_sb->setting<double>(Constants::ProfileSettings::Inset::kExtrusionMultiplier);
            else if(region_type == RegionType::kSkeleton)
                current_multiplier = m_sb->setting<double>(Constants::ProfileSettings::Skeleton::kExtrusionMultiplier);
            else if(region_type == RegionType::kSkin)
                current_multiplier = m_sb->setting<double>(Constants::ProfileSettings::Skin::kExtrusionMultiplier);
            else if(region_type == RegionType::kInfill)
                current_multiplier = m_sb->setting<double>(Constants::ProfileSettings::Infill::kExtrusionMultiplier);
            else
                current_multiplier = m_sb->setting<double>(Constants::ProfileSettings::Perimeter::kExtrusionMultiplier);

            Distance length = angle() * center_point.distance(start_point);
            Distance width = params->setting<Distance>(Constants::SegmentSettings::kWidth);
            Distance height = params->setting<Distance>(Constants::SegmentSettings::kHeight);
            Distance filament_diameter = m_sb->setting<Distance>(Constants::MaterialSettings::Filament::kDiameter);
            Distance segment_filament_length = (length * width * height) / ((filament_diameter / 2) * (filament_diameter / 2) * 3.14159);
            segment_filament_length *= current_multiplier;

            //if using relative E, write out segment filament length and be done
            if (m_sb->setting< bool >(Constants::MaterialSettings::Filament::kRelative))
            {
                m_filament_location = segment_filament_length;
                rv += m_e % QString::number(segment_filament_length.to(m_meta.m_distance_unit));
            }
            //if using absolute E, update total length and write value
            else
            {
                m_filament_location += segment_filament_length;
                rv += m_e % QString::number(m_filament_location.to(m_meta.m_distance_unit));
            }

        }

        // Add comment for gcode parser
        if (path_modifiers != PathModifiers::kNone)
            rv += commentSpaceLine(toString(region_type) % m_space % toString(path_modifiers));
        else
            rv += commentSpaceLine(toString(region_type));

        return rv;
    }

    QString Mach4Writer::writeAfterPath(RegionType type)
    {
        QString rv;
        if(!m_spiral_layer)
        {
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

    QString Mach4Writer::writeAfterRegion(RegionType type)
    {
        QString rv;
        return rv;
    }

    QString Mach4Writer::writeAfterIsland()
    {
        QString rv;
        return rv;
    }

    QString Mach4Writer::writeAfterPart()
    {
        QString rv;
        return rv;
    }

    QString Mach4Writer::writeAfterLayer()
    {
        QString rv;
        rv += m_sb->setting< QString >(Constants::PrinterSettings::GCode::kLayerCodeChange) % m_newline;
        return rv;
    }

    QString Mach4Writer::writeShutdown()
    {
        QString rv;
        rv += "G28" % commentSpaceLine("TRAVEL HOME ALL AXES");

        rv += m_sb->setting< QString >(Constants::PrinterSettings::GCode::kEndCode) % m_newline %
              "M104 S0" % commentSpaceLine("TURN EXTRUDER OFF") %
                "M140 S0" % commentSpaceLine("TURN BED OFF") %
              "M84" % commentSpaceLine("DISABLE MOTORS");
        return rv;
    }

    QString Mach4Writer::writePurge(int RPM, int duration, int delay)
    {
        return {};
    }

    QString Mach4Writer::writeDwell(Time time)
    {
        if (time > 0)
            return m_G4 % m_p % QString::number(time.to(m_meta.m_time_unit), 'f', 4) % commentSpaceLine("DWELL");
        else
            return {};
    }

    QString Mach4Writer::writeCoordinates(Point destination)
    {
        QString rv;

        //always specify X and Y
        rv += m_x % QString::number(Distance(destination.x()).to(m_meta.m_distance_unit), 'f', 4) %
              m_y % QString::number(Distance(destination.y()).to(m_meta.m_distance_unit), 'f', 4);

        //write vertical coordinate only if there was a change in Z
        Distance z_offset = m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kZOffset);

        Distance target_z = destination.z() + z_offset;
        if(qAbs(target_z - m_last_z) > 10)
        {
            rv += m_z % QString::number(Distance(target_z).to(m_meta.m_distance_unit), 'f', 4);
            m_current_z = target_z;
            m_last_z = target_z;
        }
        return rv;
    }

    QString Mach4Writer::writeRetraction()
    {
        QString rv;

        if(m_filament_location >= 0 && m_sb->setting< bool >(Constants::MaterialSettings::Retraction::kEnable))
        {
            if(m_filament_location != 0)
                rv += "G92 E0" % commentSpaceLine("RESET FILAMENT TO 0");
            m_filament_location = 0.0;

            rv += m_G1 % m_f %
                    QString::number(m_sb->setting< Velocity >(Constants::MaterialSettings::Retraction::kSpeed).to(m_meta.m_velocity_unit)) %
                    m_e % QString::number(m_sb->setting< Distance >(Constants::MaterialSettings::Retraction::kLength).to(m_meta.m_distance_unit) * -1) %
                    commentSpaceLine("RETRACT FILAMENT");

            if(m_sb->setting< bool >(Constants::MaterialSettings::Filament::kRelative))
                m_filament_location = -1 * m_sb->setting< Distance >(Constants::MaterialSettings::Retraction::kLength);
            else
                m_filament_location -= m_sb->setting< Distance >(Constants::MaterialSettings::Retraction::kLength);
        }

        return rv;
    }

    QString Mach4Writer::writePrime()
    {
        QString rv;

        if (m_sb->setting< bool >(Constants::MaterialSettings::Filament::kRelative))
        {
            m_filament_location = m_sb->setting< Distance >(Constants::MaterialSettings::Retraction::kLength) +
                    m_sb->setting< Distance >(Constants::MaterialSettings::Retraction::kPrimeAdditionalLength);
            rv += m_G1 % m_f %
                    QString::number(m_sb->setting< Velocity >(Constants::MaterialSettings::Retraction::kPrimeSpeed).to(m_meta.m_velocity_unit)) %
                    m_e % QString::number(m_filament_location.to(m_meta.m_distance_unit)) %
                    commentSpaceLine("PRIME FILAMENT");
        }
        else
        {
            m_filament_location += (m_sb->setting< Distance >(Constants::MaterialSettings::Retraction::kLength) +
                    m_sb->setting< Distance >(Constants::MaterialSettings::Retraction::kPrimeAdditionalLength));
            rv += m_G1 % m_f %
                    QString::number(m_sb->setting< Velocity >(Constants::MaterialSettings::Retraction::kPrimeSpeed).to(m_meta.m_velocity_unit)) %
                    m_e % QString::number(m_filament_location.to(m_meta.m_distance_unit)) %
                    commentSpaceLine("PRIME FILAMENT");
        }
        return rv;
    }

    QString Mach4Writer::setTools(QVector<int> extruders)
    {

        QString rv = "";
        if (m_extruders_on.size() > 2) //assumes at most two extruders for simultaneuos extrusion
        {
            if ( m_extruders_on[0] && m_extruders_on[1] && extruders.length() < 2) //currently both on, need single ext
            {
                //turn off both
                rv += "M605 S0" % m_newline;
                m_extruders_on[0] = false;
                m_extruders_on[1] = false;
            }

            if (extruders.length() > 1 && !(m_extruders_on[0] && m_extruders_on[1])) // need both exts. on & not already on
            {
                // turn on both
                rv += "M605 S2" % m_newline;
                m_extruders_on[0] = true;
                m_extruders_on[1] = true;
            }
        }

        // zero or one ext. need to be on
        for (int i = 0, end = m_extruders_on.size(); i < end; ++i )
        {
            if (!m_extruders_on[i] && extruders.contains(i)) //ext0 should be on but isn't
            {
                //write tool change to zero
                //rv += "T" % QString::number(i) % m_newline;
                m_extruders_on[i] = true;
            }
            else if (m_extruders_on[i] && !extruders.contains(i))
            {
                m_extruders_on[i] = false;
            }
            // else - was already corrrect
        }
        return rv;
    }
}
