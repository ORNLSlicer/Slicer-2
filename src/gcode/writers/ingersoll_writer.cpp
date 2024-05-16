#include <QStringBuilder>

#include "gcode/writers/ingersoll_writer.h"
#include "utilities/enums.h"
#include "utilities/mathutils.h"

namespace ORNL
{
    IngersollWriter::IngersollWriter(GcodeMeta meta, const QSharedPointer<SettingsBase>& sb) : WriterBase(meta, sb) {

    }

    QString IngersollWriter::writeInitialSetup(Distance minimum_x, Distance minimum_y, Distance maximum_x, Distance maximum_y, int num_layers)
    {
        m_current_z = m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kZOffset);
        m_current_rpm = 0;
        for (int ext = 0, end = m_extruders_on.size(); ext < end; ++ext) //all extruders off initially
            m_extruders_on[ext] = false;
        m_first_travel = true;
        m_first_print = true;
        m_layer_start = true;
        m_min_z = 0.0f;
        m_material_number = -1;
        m_wire_feed = false;
        m_wire_feed_total = 0;

        QString rv;
        if (m_sb->setting< int >(Constants::PrinterSettings::GCode::kEnableStartupCode))
        {
            //rv += "G0 X0 Y0 Z-3179.816 A3=0 B3=0 C3=1 AN3=0 BN3=1 CN3=0 F6000" % commentSpaceLine("PLANE CHANGE");
        }

        if(m_sb->setting< int >(Constants::PrinterSettings::GCode::kEnableBoundingBox))
        {
            rv += "G0 Z0" % commentSpaceLine("RAISE Z TO DEMO BOUNDING BOX")
                % m_G0 % m_x % QString::number(minimum_x.to(m_meta.m_distance_unit), 'f', 4) % " Y" % QString::number(minimum_y.to(m_meta.m_distance_unit), 'f', 4) % commentSpaceLine("BOUNDING BOX")
                % m_G0 % m_x % QString::number(maximum_x.to(m_meta.m_distance_unit), 'f', 4) % " Y" % QString::number(minimum_y.to(m_meta.m_distance_unit), 'f', 4) % commentSpaceLine("BOUNDING BOX")
                % m_G0 % m_x % QString::number(maximum_x.to(m_meta.m_distance_unit), 'f', 4) % " Y" % QString::number(maximum_y.to(m_meta.m_distance_unit), 'f', 4) % commentSpaceLine("BOUNDING BOX")
                % m_G0 % m_x % QString::number(minimum_x.to(m_meta.m_distance_unit), 'f', 4) % " Y" % QString::number(maximum_y.to(m_meta.m_distance_unit), 'f', 4) % commentSpaceLine("BOUNDING BOX")
                % m_G0 % m_x % QString::number(minimum_x.to(m_meta.m_distance_unit), 'f', 4) % " Y" % QString::number(minimum_y.to(m_meta.m_distance_unit), 'f', 4) % commentSpaceLine("BOUNDING BOX");

            m_start_point = Point(minimum_x, minimum_y, 0);
        }

        if (m_sb->setting< int >(Constants::PrinterSettings::GCode::kEnableMaterialLoad))
        {

        }

        if(m_sb->setting< QString >(Constants::PrinterSettings::GCode::kStartCode) != "")
            rv += m_sb->setting< QString >(Constants::PrinterSettings::GCode::kStartCode);

        rv += m_newline;

        rv += commentLine("LAYER COUNT: " % QString::number(num_layers));

        return rv;
    }

    QString IngersollWriter::writeBeforeLayer(float new_min_z, QSharedPointer<SettingsBase> sb)
    {
        m_spiral_layer = sb->setting<bool>(Constants::ProfileSettings::SpecialModes::kEnableSpiralize);
        m_layer_start = true;
        QString rv;
        return rv;
    }

    QString IngersollWriter::writeBeforePart(QVector3D normal)
    {
        QString rv;
        return rv;
    }

    QString IngersollWriter::writeBeforeIsland()
    {
        QString rv;
        return rv;
    }

    QString IngersollWriter::writeBeforeScan(Point min, Point max, int layer, int boundingBox, Axis axis, Angle angle)
    {
        QString rv;
        return rv;
    }

    QString IngersollWriter::writeBeforeRegion(RegionType type, int pathSize)
    {
        QString rv;
        return rv;
    }

    QString IngersollWriter::writeBeforePath(RegionType type)
    {
        QString rv;
        if(!m_spiral_layer || m_first_print)
        {
            if (type == RegionType::kPerimeter)
            {
                if (!m_sb->setting< QString >(Constants::ProfileSettings::GCode::kPerimeterStart).isEmpty())
                    rv += m_sb->setting< QString >(Constants::ProfileSettings::GCode::kPerimeterStart) % m_newline;            }
            else if (type == RegionType::kInset)
            {
                if (!m_sb->setting< QString >(Constants::ProfileSettings::GCode::kInsetStart).isEmpty())
                    rv += m_sb->setting< QString >(Constants::ProfileSettings::GCode::kInsetStart) % m_newline;            }
            else if(type == RegionType::kSkeleton)
            {
                if (!m_sb->setting< QString >(Constants::ProfileSettings::GCode::kSkeletonStart).isEmpty())
                    rv += m_sb->setting< QString >(Constants::ProfileSettings::GCode::kSkeletonStart) % m_newline;            }
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
            else
            {
            }
        }
        return rv;
    }

    QString IngersollWriter::writeTravel(Point start_location, Point target_location, TravelLiftType lType,
                                          QSharedPointer<SettingsBase> params)
    {
        QString rv;
        Point new_start_location;

        bool isWireFeed = false;
        if(params->contains(Constants::SegmentSettings::kWireFeed))
            isWireFeed = true;

        if(isWireFeed)
            m_wire_feed_total = 0;

        //Use updated start location if this is the first travel
        if(m_first_travel)
            new_start_location = m_start_point;
        else
            new_start_location = start_location;

        Distance liftDist;
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
            if(isWireFeed)
            {
                rv += m_G1 % writeCoordinates(lift_destination);
                Velocity speed = m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kZSpeed);
                if (getFeedrate() != speed)
                {
                    setFeedrate(speed);
                    rv += m_f % QString::number(speed.to(m_meta.m_velocity_unit));
                }
                 rv += commentSpaceLine("TRAVEL LIFT Z");
            }
            else
            {
                rv += m_G0 % writeCoordinates(lift_destination) % commentSpaceLine("TRAVEL LIFT Z");
                setFeedrate(m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kZSpeed));
            }
        }

        //write the travel
        Point travel_destination = target_location;
        if(m_first_travel)
            travel_destination.z(qAbs(m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kZOffset)()));
        else if (travel_lift_required)
            travel_destination = travel_destination + travel_lift; //travel destination is above the target point

        if(isWireFeed)
        {
            rv += "H54" % commentSpaceLine("Zero UA and UT");
            rv += "H122" % commentSpaceLine("Engage add roller");
            rv += m_G1 % writeCoordinates(travel_destination);
            rv += " UA=" + QString::number(params->setting<Distance>(Constants::SegmentSettings::kWireFeed).to(m_meta.m_distance_unit));
            m_wire_feed_total += params->setting<Distance>(Constants::SegmentSettings::kWireFeed);
            Velocity speed = params->setting<Velocity>(Constants::SegmentSettings::kSpeed);
            if (getFeedrate() != speed)
            {
                setFeedrate(speed);
                rv += m_f % QString::number(speed.to(m_meta.m_velocity_unit));
            }
        }
        else
        {
            rv += m_G0 % writeCoordinates(travel_destination);
        }

        rv += commentSpaceLine("TRAVEL");
        setFeedrate(m_sb->setting< Velocity >(Constants::ProfileSettings::Travel::kSpeed));

        if(isWireFeed)
            rv += "H123" % commentSpaceLine("Disengage add roller");

        //write the travel lower (undo the lift)
        if (travel_lift_required && (lType == TravelLiftType::kBoth || lType == TravelLiftType::kLiftLowerOnly))
        {
            if(isWireFeed)
            {
                rv += m_G1 % writeCoordinates(target_location);
                Velocity speed = m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kZSpeed);
                if (getFeedrate() != speed)
                {
                    setFeedrate(speed);
                    rv += m_f % QString::number(speed.to(m_meta.m_velocity_unit));
                }
                rv += commentSpaceLine("TRAVEL LOWER Z");
            }
            else
            {
                rv += m_G0 % writeCoordinates(target_location) % commentSpaceLine("TRAVEL LOWER Z");
                setFeedrate(m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kZSpeed));
            }
        }

        if (m_first_travel) //if this is the first travel
            m_first_travel = false; //update for next one

        return rv;
    }

    QString IngersollWriter::writeLine(const Point& start_point, const Point& target_point, const QSharedPointer<SettingsBase> params)
    {
        Velocity speed = params->setting<Velocity>(Constants::SegmentSettings::kSpeed);
        float rpm = params->setting<float>(Constants::SegmentSettings::kExtruderSpeed);
        int material_number = params->setting<int>(Constants::SegmentSettings::kMaterialNumber);
        RegionType region_type = params->setting<RegionType>(Constants::SegmentSettings::kRegionType);
        PathModifiers path_modifiers = params->setting<PathModifiers>(Constants::SegmentSettings::kPathModifiers);
        float output_rpm = rpm * m_sb->setting< float >(Constants::PrinterSettings::MachineSpeed::kGearRatio);

        QString rv;

        for (int extruder : params->setting<QVector<int>>(Constants::SegmentSettings::kExtruders))
        {
            // Turn on the extruder if it isn't already on
            if (m_extruders_on[0] == false && rpm > 0) //only check first extruder
            {
                rv += writeExtruderOn(region_type, rpm, extruder);
                m_current_rpm = rpm;
            }
            // Update extruder speed if needed
            if (m_extruders_on[0] == true && rpm != m_current_rpm) //only check first extruder
            {
                rv += "EXTRUDER(" % QString::number(output_rpm) % ")" % commentSpaceLine("UPDATE EXTRUDER RPM");
                m_current_rpm = rpm;
            }
        }

        if(!m_wire_feed && params->contains(Constants::SegmentSettings::kWireFeed))
        {
            rv += "H54" % commentSpaceLine("Zero UA and UT");
            rv += "H122" % commentSpaceLine("Engage add roller");
            m_wire_feed = true;
        }

        rv += m_G1;
        // Update feedrate if needed
        if (getFeedrate() != speed || m_layer_start)
        {
            setFeedrate(speed);
            rv += m_f % QString::number(speed.to(m_meta.m_velocity_unit));
            m_layer_start = false;
        }

        // Writes WXYZ to destination
        rv += writeCoordinates(target_point);

        if(m_wire_feed)
        {
            m_wire_feed_total += params->setting<Distance>(Constants::SegmentSettings::kWireFeed);
            rv += " UA=" % QString::number(m_wire_feed_total.to(m_meta.m_distance_unit));

            if(params->setting<bool>(Constants::SegmentSettings::kFinalWireCoast))
            {
                rv += " H120";
                m_wire_feed = false;
            }

            if(params->setting<bool>(Constants::SegmentSettings::kFinalWireFeed))
            {
                rv += " H123";
            }
        }

        // Add comment for gcode parser
        if (path_modifiers != PathModifiers::kNone)
            rv += commentSpaceLine(toString(region_type) % m_space % toString(path_modifiers));
        else
            rv += commentSpaceLine(toString(region_type));

        m_first_print = false;

        return rv;
    }

    QString IngersollWriter::writeArc(const Point &start_point,
                                           const Point &end_point,
                                           const Point &center_point,
                                           const Angle &angle,
                                           const bool &ccw,
                                           const QSharedPointer<SettingsBase> params)
    {
        QString rv;

        Velocity speed = params->setting<Velocity>(Constants::SegmentSettings::kSpeed);
        int rpm = params->setting<int>(Constants::SegmentSettings::kExtruderSpeed);
        float output_rpm = rpm * m_sb->setting< float >(Constants::PrinterSettings::MachineSpeed::kGearRatio);
        int material_number = params->setting<int>(Constants::SegmentSettings::kMaterialNumber);
        auto region_type = params->setting<RegionType>(Constants::SegmentSettings::kRegionType);
        auto path_modifiers = params->setting<PathModifiers>(Constants::SegmentSettings::kPathModifiers);

        for (int extruder : params->setting<QVector<int>>(Constants::SegmentSettings::kExtruders))
        {
            //turn on the extruder if it isn't already on
            if (m_extruders_on[0] == false && rpm > 0) //only check first extruder
            {
                rv += writeExtruderOn(region_type, rpm, extruder);
            }
            // Update extruder speed if needed
            if (m_extruders_on[0] == true && rpm != m_current_rpm) //only check first extruder
            {
                rv += "EXTRUDER(" % QString::number(output_rpm) % ")" % commentSpaceLine("UPDATE EXTRUDER RPM");
                m_current_rpm = rpm;
            }
        }

        rv += ((ccw) ? m_G3 : m_G2);

        if (getFeedrate() != speed)
        {
            setFeedrate(speed);
            rv += m_f % QString::number(speed.to(m_meta.m_velocity_unit));
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

    QString IngersollWriter::writeSpline(const Point &start_point,
                                  const Point &a_control_point,
                                  const Point &b_control_point,
                                  const Point &end_point,
                                  const QSharedPointer<SettingsBase> params)
    {
        QString rv;


        Velocity speed = params->setting<Velocity>(Constants::SegmentSettings::kSpeed);
        int rpm = params->setting<int>(Constants::SegmentSettings::kExtruderSpeed);
        float output_rpm = rpm * m_sb->setting< float >(Constants::PrinterSettings::MachineSpeed::kGearRatio);
        int material_number = params->setting<int>(Constants::SegmentSettings::kMaterialNumber);
        auto region_type = params->setting<RegionType>(Constants::SegmentSettings::kRegionType);
        auto path_modifiers = params->setting<PathModifiers>(Constants::SegmentSettings::kPathModifiers);

        for (int extruder : params->setting<QVector<int>>(Constants::SegmentSettings::kExtruders))
        {
            //turn on the extruder if it isn't already on
            if (m_extruders_on[0] == false && rpm > 0) //only check first extruder
            {
                rv += writeExtruderOn(region_type, rpm, extruder);
            }
            // Update extruder speed if needed
            if (m_extruders_on[0] == true && rpm != m_current_rpm) //only check first extruder
            {
                rv += "EXTRUDER(" % QString::number(output_rpm) % ")" % commentSpaceLine("UPDATE EXTRUDER RPM");
                m_current_rpm = rpm;
            }

        }

        rv += m_G5;

        if (getFeedrate() != speed)
        {
            setFeedrate(speed);
            rv += m_f % QString::number(speed.to(m_meta.m_velocity_unit));
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

    QString IngersollWriter::writeScan(Point target_point, Velocity speed, bool on_off)
    {
        QString rv;
        return rv;
    }

    QString IngersollWriter::writeAfterPath(RegionType type)
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
        rv += commentLine("Bead End");
        return rv;
    }

    QString IngersollWriter::writeAfterRegion(RegionType type)
    {
        QString rv;
        return rv;
    }

    QString IngersollWriter::writeAfterScan(Distance beadWidth, Distance laserStep, Distance laserResolution)
    {
        return {};
    }

    QString IngersollWriter::writeAfterIsland()
    {
        QString rv;
        return rv;
    }

    QString IngersollWriter::writeAfterPart()
    {
        QString rv;
        return rv;
    }

    QString IngersollWriter::writeAfterLayer()
    {
        QString rv;
        rv += m_sb->setting< QString >(Constants::PrinterSettings::GCode::kLayerCodeChange) % m_newline;
        return rv;
    }

    QString IngersollWriter::writeShutdown()
    {
        QString rv;
        rv += writeExtruderOff(0); //update to turn off appropriate extruders
        rv += m_sb->setting< QString >(Constants::PrinterSettings::GCode::kEndCode);
        return rv;
    }

    QString IngersollWriter::writePurge(int RPM, int duration, int delay)
    {
        return {};
    }

    QString IngersollWriter::writeDwell(Time time)
    {
        if (time > 0)
            return m_G4 % m_f % QString::number(time.to(m_meta.m_time_unit), 'f', 4) % commentSpaceLine("DWELL");
        else
            return {};
    }

    QString IngersollWriter::writeExtruderOn(RegionType type, float rpm, int extruder_number)
    {
        QString rv;
        float output_rpm;

        rv += commentLine("Bead Start");

        m_extruders_on[extruder_number] = true;


        //write dwell and initial extruder turn on depending on region type
        if (m_sb->setting< int >(Constants::MaterialSettings::Extruder::kInitialSpeed) > 0)
        {
            output_rpm = m_sb->setting< float >(Constants::PrinterSettings::MachineSpeed::kGearRatio) * m_sb->setting< float >(Constants::MaterialSettings::Extruder::kInitialSpeed);

            rv += "EXTRUDER(" % QString::number(output_rpm) % ")" % commentSpaceLine("TURN EXTRUDER ON");

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

        // Update Extruder Speed
        output_rpm = m_sb->setting< float >(Constants::PrinterSettings::MachineSpeed::kGearRatio) * rpm;
        rv += "EXTRUDER(" % QString::number(output_rpm) % ")" % commentSpaceLine("UPDATE EXTRUDER RPM");
        m_current_rpm = rpm;

        return rv;
    }

    QString IngersollWriter::writeExtruderOff(int extruder_number)
    {
        //update to use extruder number

        QString rv;
        m_extruders_on[extruder_number] = false;
        if(m_sb->setting< Time >(Constants::MaterialSettings::Extruder::kOffDelay) > 0)
        {
            rv += writeDwell(m_sb->setting< Time >(Constants::MaterialSettings::Extruder::kOffDelay));
        }
        rv += "EXTRUDER(0)" % commentSpaceLine("TURN EXTRUDER OFF");
        return rv;
    }

    QString IngersollWriter::writeCoordinates(Point destination)
    {
        QString rv;

        //always specify X and Y
        rv += m_x % QString::number(Distance(destination.x()).to(m_meta.m_distance_unit), 'f', 4) %
              m_y % QString::number(Distance(destination.y()).to(m_meta.m_distance_unit), 'f', 4) %
              getZWValue(destination);

        return rv;
    }

    QString IngersollWriter::getZWValue(const Point& destination)
    {
        QString rv;
        //write vertical coordinate along the Z axis
        //only output Z coordinate if there was a change in Z
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

}  // namespace ORNL
