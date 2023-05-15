#include <QStringBuilder>

#include "gcode/writers/mvp_writer.h"
#include "utilities/enums.h"

namespace ORNL
{
    MVPWriter::MVPWriter(GcodeMeta meta, const QSharedPointer<SettingsBase>& sb) : WriterBase(meta, sb) {
        // NOP
    }

    QString MVPWriter::writeInitialSetup(Distance minimum_x, Distance minimum_y, Distance maximum_x, Distance maximum_y, int num_layers)
    {
        m_current_z = m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kZOffset);
        m_current_rpm = 0;
        m_extruders_on[0] = false;
        m_first_print = true;
        m_layer_start = true;
        QString rv;
        if (m_sb->setting< int >(Constants::PrinterSettings::GCode::kEnableStartupCode))
        {
            rv += "G54" % commentSpaceLine("ENABLE W01") %
                  m_G1 % m_f % QString::number(m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kZSpeed).to(m_meta.m_velocity_unit), 'f', 4)
                    % " Z0" % commentSpaceLine("SET INITIAL TABLE HEIGHT");
            m_current_z = 0;
        }

        if(m_sb->setting< int >(Constants::PrinterSettings::GCode::kEnableBoundingBox))
        {
            rv += m_G0 % m_x % QString::number(minimum_x.to(in), 'f', 4) % m_y % QString::number(minimum_y.to(in), 'f', 4) % commentSpaceLine("BOUNDING BOX") %
                  m_G0 % m_x  % QString::number(maximum_x.to(in), 'f', 4) % m_y % QString::number(minimum_y.to(in), 'f', 4) % commentSpaceLine("BOUNDING BOX") %
                  m_G0 % m_x  % QString::number(maximum_x.to(in), 'f', 4) % m_y % QString::number(maximum_y.to(in), 'f', 4) % commentSpaceLine("BOUNDING BOX") %
                  m_G0 % m_x  % QString::number(minimum_x.to(in), 'f', 4) % m_y % QString::number(maximum_y.to(in), 'f', 4) % commentSpaceLine("BOUNDING BOX") %
                  m_G0 % m_x  % QString::number(minimum_x.to(in), 'f', 4) % m_y % QString::number(minimum_y.to(in), 'f', 4) % commentSpaceLine("BOUNDING BOX") %
                  "M0" % commentSpaceLine("WAIT FOR USER");
        }

        if(!m_sb->setting< QString >(Constants::PrinterSettings::GCode::kStartCode).isEmpty())
            rv += m_sb->setting< QString >(Constants::PrinterSettings::GCode::kStartCode);

        rv += m_newline;

        rv += commentLine("LAYER COUNT: " % QString::number(num_layers));

        return rv;
    }

    QString MVPWriter::writeBeforeLayer(float min_z,QSharedPointer<SettingsBase> sb)
    {
        m_spiral_layer = sb->setting<bool>(Constants::ProfileSettings::SpecialModes::kEnableSpiralize);
        m_layer_start = true;
        QString rv;
        rv += "M01 " % commentLine("OPTIONAL STOP - LAYER CHANGE");
        return rv;
    }

    QString MVPWriter::writeBeforePart(QVector3D normal)
    {
        QString rv;
        return rv;
    }

    QString MVPWriter::writeBeforeIsland()
    {
        QString rv;
        return rv;
    }

    QString MVPWriter::writeBeforeRegion(RegionType type, int pathSize)
    {
        return {};
    }

    QString MVPWriter::writeBeforePath(RegionType type)
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

    QString MVPWriter::writeTravelLift(bool lift)
    {
        if (lift)
            return "M51 ; RAISE EXTRUDER\n";
        else
            return "M50 ; LOWER EXTRUDER\n";
    }

    QString MVPWriter::writeTravel(Point start_location, Point target_location, TravelLiftType lType,
                                   QSharedPointer<SettingsBase> params)
    {
        QString rv;

        bool travel_lift_required = true;

        //Don't lift for short travel moves
        if(start_location.distance(target_location) < m_sb->setting< Distance >(Constants::ProfileSettings::Travel::kMinTravelForLift))
        {
            travel_lift_required = false;
        }

        if(travel_lift_required)
            rv += writeTravelLift(true);

        rv += m_G0 % writeCoordinates(target_location) % commentSpaceLine("TRAVEL");
        setFeedrate(params->setting<Velocity>(Constants::SegmentSettings::kSpeed));

        if(travel_lift_required)
            rv += writeTravelLift(false);

        return rv;
    }

    QString MVPWriter::writeLine(const Point& start_point, const Point& target_point, const QSharedPointer<SettingsBase> params)
    {
        Velocity speed = params->setting<Velocity>(Constants::SegmentSettings::kSpeed);
        int rpm = params->setting<int>(Constants::SegmentSettings::kExtruderSpeed);
        RegionType region_type = params->setting<RegionType>(Constants::SegmentSettings::kRegionType);
        PathModifiers path_modifiers = params->setting<PathModifiers>(Constants::SegmentSettings::kPathModifiers);
        float output_rpm = rpm * m_sb->setting< float >(Constants::PrinterSettings::MachineSpeed::kGearRatio);

        QString rv;

        //turn on the extruder if it isn't already on
        if (m_extruders_on[0] == false && rpm > 0)
        {
            rv += writeExtruderOn(region_type, rpm);
        }

        if (rpm != m_current_rpm)
        {
            rv += m_M3 % m_s % QString::number(output_rpm) % commentSpaceLine("UPDATE EXTRUDER RPM");
            m_current_rpm = rpm;
        }

        rv += m_G1;
        if (getFeedrate() != speed || m_layer_start)
        {
            setFeedrate(speed);
            rv += m_f % QString::number(speed.to(m_meta.m_velocity_unit));
            m_layer_start = false;
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

    QString MVPWriter::writeAfterPath(RegionType type)
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

    QString MVPWriter::writeAfterRegion(RegionType type)
    {
        QString rv;
        return rv;
    }

    QString MVPWriter::writeAfterIsland()
    {
        QString rv;
        return rv;
    }

    QString MVPWriter::writeAfterPart()
    {
        QString rv;
        return rv;
    }

    QString MVPWriter::writeAfterLayer()
    {
        QString rv;
        rv += m_sb->setting< QString >(Constants::PrinterSettings::GCode::kLayerCodeChange) % m_newline;
        return rv;
    }

    QString MVPWriter::writeShutdown()
    {
        QString rv;
        rv += "M5" % commentSpaceLine("TURN PUMP OFF END OF PRINT");
        rv += "M53" % commentSpaceLine("TURN GUN OFF END OF PRINT");
        rv += m_sb->setting< QString >(Constants::PrinterSettings::GCode::kEndCode) % m_newline;
        rv += commentLine("PARK");
        //rv += writeTravelLift(true);
        rv += "G0 X0 Y0" % commentSpaceLine("TRAVEL");
        rv += "M30" % commentSpaceLine("END OF G-CODE");
        return rv;
    }

    QString MVPWriter::writePurge(int RPM, int duration, int delay)
    {
        return {};
    }

    QString MVPWriter::writeDwell(Time time)
    {
        if (time > 0)
            return m_G4 % m_f % QString::number(time.to(m_meta.m_time_unit), 'f', 4) % commentSpaceLine("DWELL");
        else
            return {};
    }

    QString MVPWriter::writeExtruderOn(RegionType type, int rpm)
    {
        QString rv;
        m_extruders_on[0] = true;
        float output_rpm;

        rv += "M52" % commentSpaceLine("TURN GUN ON");

        if (m_sb->setting< int >(Constants::MaterialSettings::Extruder::kInitialSpeed) > 0)
        {
            output_rpm = m_sb->setting< float >(Constants::PrinterSettings::MachineSpeed::kGearRatio) * m_sb->setting< int >(Constants::MaterialSettings::Extruder::kInitialSpeed);

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
            m_current_rpm = rpm;
        }

        return rv;
    }

    QString MVPWriter::writeExtruderOff()
    {
        QString rv;
        m_extruders_on[0] = false;
        if(m_sb->setting< Time >(Constants::MaterialSettings::Extruder::kOffDelay) > 0)
        {
            rv += writeDwell(m_sb->setting< Time >(Constants::MaterialSettings::Extruder::kOffDelay));
        }
        rv += "M53" % commentSpaceLine("TURN GUN OFF");
        return rv;
    }

    QString MVPWriter::writeCoordinates(Point destination)
    {
        QString rv;

        //always specify X and Y
        rv += m_x % QString::number(Distance(destination.x()).to(m_meta.m_distance_unit), 'f', 4) %
              m_y % QString::number(Distance(destination.y()).to(m_meta.m_distance_unit), 'f', 4);

        //write vertical coordinate only if there was a change in Z
        Distance z_offset = m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kZOffset);

        Distance target_z = (-1 * destination.z()) + z_offset;
        if(qAbs(target_z - m_last_z) > 10)
        {
            rv += m_z % QString::number(Distance(target_z).to(m_meta.m_distance_unit), 'f', 4);
            m_current_z = target_z;
            m_last_z = target_z;
        }
        return rv;
    }

}  // namespace ORNL
