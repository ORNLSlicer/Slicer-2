#include <QStringBuilder>

#include "gcode/writers/skybaam_writer.h"
#include "utilities/enums.h"
#include "utilities/mathutils.h"

namespace ORNL
{
    SkyBaamWriter::SkyBaamWriter(GcodeMeta meta, const QSharedPointer<SettingsBase>& sb) : WriterBase(meta, sb) {
        //NOP
    }

    QString SkyBaamWriter::writeInitialSetup(Distance minimum_x, Distance minimum_y, Distance maximum_x, Distance maximum_y, int num_layers)
    {
        m_current_z = m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kZOffset);
        m_current_rpm = 0;
        m_extruders_on[0] = false;
        m_first_print = true;
        m_first_travel = true;
        m_layer_start = true;
        QString rv;
        if (m_sb->setting< int >(Constants::PrinterSettings::GCode::kEnableStartupCode))
        {
            //Nothing here right now
        }

        if(m_sb->setting< int >(Constants::PrinterSettings::GCode::kEnableBoundingBox))
        {
            rv += "G0 Z0" % commentSpaceLine("RAISE Z TO DEMO BOUNDING BOX") %
                  m_G0 % m_x % QString::number(minimum_x.to(in), 'f', 4) % m_y % QString::number(minimum_y.to(in), 'f', 4) % commentSpaceLine("BOUNDING BOX") %
                  m_G0 % m_x % QString::number(maximum_x.to(in), 'f', 4) % m_y % QString::number(minimum_y.to(in), 'f', 4) % commentSpaceLine("BOUNDING BOX") %
                  m_G0 % m_x % QString::number(maximum_x.to(in), 'f', 4) % m_y % QString::number(maximum_y.to(in), 'f', 4) % commentSpaceLine("BOUNDING BOX") %
                  m_G0 % m_x % QString::number(minimum_x.to(in), 'f', 4) % m_y % QString::number(maximum_y.to(in), 'f', 4) % commentSpaceLine("BOUNDING BOX") %
                  m_G0 % m_x % QString::number(minimum_x.to(in), 'f', 4) % m_y % QString::number(minimum_y.to(in), 'f', 4) % commentSpaceLine("BOUNDING BOX") %
                  "M0" % commentSpaceLine("WAIT FOR USER");

            m_start_point = Point(minimum_x, minimum_y, 0);
        }

        if(m_sb->setting< QString >(Constants::PrinterSettings::GCode::kStartCode) != "")
            rv += m_sb->setting< QString >(Constants::PrinterSettings::GCode::kStartCode);

        rv += m_newline;

        rv += commentLine("LAYER COUNT: " % QString::number(num_layers));

        return rv;
    }

    QString SkyBaamWriter::writeBeforeLayer(float min_z, QSharedPointer<SettingsBase> sb)
    {
        m_spiral_layer = sb->setting<bool>(Constants::ProfileSettings::SpecialModes::kEnableSpiralize);
        m_layer_start = true;
        QString rv;
        return rv;
    }

    QString SkyBaamWriter::writeBeforePart(QVector3D normal)
    {
        QString rv;
        return rv;
    }

    QString SkyBaamWriter::writeBeforeIsland()
    {
        QString rv;
        return rv;
    }

    QString SkyBaamWriter::writeBeforeRegion(RegionType type, int pathSize)
    {
        QString rv;
        return rv;
    }

    QString SkyBaamWriter::writeBeforePath(RegionType type)
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
        return rv;
    }

    QString SkyBaamWriter::writeTravel(Point start_location, Point target_location, TravelLiftType lType,
                                       QSharedPointer<SettingsBase> params)
    {
        QString rv;
        Point new_start_location;
        Velocity speed = params->setting<Velocity>(Constants::SegmentSettings::kSpeed);

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

        //travel_lift vector in direction normal to the layer
        //with length = lift height as defined in settings
        QVector3D travel_lift = getTravelLift();

        //write the lift
        if (travel_lift_required && (lType == TravelLiftType::kBoth || lType == TravelLiftType::kLiftUpOnly))
        {
            Point lift_destination = new_start_location + travel_lift; //lift destination is above start location
            rv += m_G1 % " F" % QString::number(m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kZSpeed).to(m_meta.m_velocity_unit))
                    % writeCoordinates(lift_destination) % commentSpaceLine("TRAVEL LIFT Z");
            setFeedrate(m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kZSpeed));
        }

        //write the travel
        Point travel_destination = target_location;
        if (travel_lift_required)
        {
            travel_destination = travel_destination + travel_lift; //travel destination is above the target point
        }

        rv += m_G1 % " F" % QString::number(speed.to(m_meta.m_velocity_unit))
                % writeCoordinates(travel_destination) % commentSpaceLine("TRAVEL");
        setFeedrate(speed);

        //write the travel lower (undo the lift)
        if (travel_lift_required && (lType == TravelLiftType::kBoth || lType == TravelLiftType::kLiftLowerOnly))
        {
            rv += m_G1 % " F" % QString::number(m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kZSpeed).to(m_meta.m_velocity_unit))
                    % writeCoordinates(target_location) % commentSpaceLine("TRAVEL LOWER Z");
            setFeedrate(m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kZSpeed));
        }

        if (m_first_travel) //if this is the first travel
            m_first_travel = false; //update for next one

        return rv;
    }

    QString SkyBaamWriter::writeLine(const Point& start_point, const Point& target_point, const QSharedPointer<SettingsBase> params)
    {
        Velocity speed = params->setting<Velocity>(Constants::SegmentSettings::kSpeed);
        int rpm = params->setting<int>(Constants::SegmentSettings::kExtruderSpeed);
        RegionType region_type = params->setting<RegionType>(Constants::SegmentSettings::kRegionType);
        PathModifiers path_modifiers = params->setting<PathModifiers>(Constants::SegmentSettings::kPathModifiers);
        float output_rpm = rpm * m_sb->setting< float >(Constants::PrinterSettings::MachineSpeed::kGearRatio);

        QString rv;

        //turn on the extruder if it isn't already on
        if ( !m_extruders_on[0] && rpm > 0)
        {
            m_current_rpm = rpm;
            rv += writeExtruderOn(region_type, rpm);
        }
        if (rpm == 0 && m_extruders_on[0])
        {
            m_current_rpm = rpm;
            rv += writeExtruderOff();
        }

        rv += m_G1;
        if (getFeedrate() != speed || m_layer_start)
        {
            setFeedrate(speed);
            rv += m_f % QString::number(speed.to(GcodeMetaList::SkyBaamMeta.m_velocity_unit));
            m_layer_start = false;
        }

        if (rpm != m_current_rpm)
        {
            rv += m_s % QString::number(output_rpm);
            m_current_rpm = rpm;
        }

        //writes XYZ to destination
        rv += writeCoordinates(target_point);

        //add comment for gcode parser
        if (path_modifiers != PathModifiers::kNone)
            rv += commentSpaceLine(toString(region_type) % m_space % toString(path_modifiers));
        else
            rv += commentSpaceLine(toString(region_type));

        m_first_print = false;

        return rv;
    }

    QString SkyBaamWriter::writeAfterPath(RegionType type)
    {
        QString rv;
        if(!m_spiral_layer)
        {
            rv += writeExtruderOff();
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

    QString SkyBaamWriter::writeAfterRegion(RegionType type)
    {
        return {};
    }

    QString SkyBaamWriter::writeAfterIsland()
    {
        return {};
    }

    QString SkyBaamWriter::writeAfterPart()
    {
        return {};
    }

    QString SkyBaamWriter::writeAfterLayer()
    {
        QString rv;
        rv += m_sb->setting< QString >(Constants::PrinterSettings::GCode::kLayerCodeChange) % m_newline;
        return rv;
    }

    QString SkyBaamWriter::writeShutdown()
    {
        QString rv;
        rv += "M5 " % commentLine("TURN EXTRUDER OFF END OF PRINT");
        rv += "M68 (PARK)\n";
        rv += m_sb->setting< QString >(Constants::PrinterSettings::GCode::kEndCode) % m_newline;
        rv += "M30 (END OF G-CODE)\n";
        return rv;
    }

    QString SkyBaamWriter::writeDwell(Time time)
    {
        if (time > 0)
            return m_G4 % m_s % QString::number(time.to(m_meta.m_time_unit), 'f', 4) % commentSpaceLine("DWELL");
        else
            return {};
    }

    QString SkyBaamWriter::writeExtruderOn(RegionType type, int rpm)
    {
        QString rv;
        m_extruders_on[0] = true;
        float output_rpm;

        if (m_sb->setting< int >(Constants::MaterialSettings::Extruder::kInitialSpeed) > 0)
        {
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

        return rv;
    }

    QString SkyBaamWriter::writeExtruderOff()
    {
        QString rv;
        m_extruders_on[0] = false;
        if(m_sb->setting< Time >(Constants::MaterialSettings::Extruder::kOffDelay) > 0)
        {
            rv += writeDwell(m_sb->setting< Time >(Constants::MaterialSettings::Extruder::kOffDelay));
        }
        rv += m_M5 % commentSpaceLine("TURN EXTRUDER OFF");
        return rv;
    }

    QString SkyBaamWriter::writePurge(int RPM, int duration, int delay)
    {
        return {};
    }

    QString SkyBaamWriter::writeCoordinates(Point destination)
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

}  // namespace ORNL
