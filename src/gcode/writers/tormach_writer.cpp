#include <QStringBuilder>

#include "gcode/writers/tormach_writer.h"
#include "utilities/enums.h"
#include "utilities/mathutils.h"

namespace ORNL
{
TormachWriter::TormachWriter(GcodeMeta meta, const QSharedPointer<SettingsBase>& sb) : WriterBase(meta, sb) {
}

QString TormachWriter::writeSettingsHeader(GcodeSyntax syntax)
{
    QString text = "";
    text += commentLine("Slicing Parameters");
    WriterBase::writeSettingsHeader(syntax);

    text += m_newline;
    return text;
}

QString TormachWriter::writeInitialSetup(Distance minimum_x, Distance minimum_y, Distance maximum_x, Distance maximum_y, int num_layers)
{
    m_current_z = m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kZOffset);
    m_current_rpm = 0;
    for (int ext = 0, end = m_extruders_on.size(); ext < end; ++ext) //all extruders off initially
        m_extruders_on[ext] = false;
    m_first_travel = true;
    m_first_print = true;
    m_layer_start = true;
    m_tip_wipe = false;
    m_min_z = 0.0f;
    QString rv;
    if (m_sb->setting< int >(Constants::PrinterSettings::GCode::kEnableStartupCode))
    {
        rv += commentLine("SAFETY BLOCK - ESTABLISH OPERATIONAL MODES");
        rv += "G1 F120 " % commentLine("SET INITIAL FEEDRATE");
        if (m_sb->setting< int >(Constants::PrinterSettings::GCode::kEnableWaitForUser))
        {
            rv += "M0" % commentSpaceLine("WAIT FOR USER");
        }
        rv += writeDwell(0.25);
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

    if(m_sb->setting< QString >(Constants::PrinterSettings::GCode::kStartCode) != "")
        rv += m_sb->setting< QString >(Constants::PrinterSettings::GCode::kStartCode);

    rv += m_newline;

    rv += commentLine("LAYER COUNT: " % QString::number(num_layers));

    return rv;
}

QString TormachWriter::writeBeforeLayer(float new_min_z, QSharedPointer<SettingsBase> sb)
{
    m_spiral_layer = sb->setting<bool>(Constants::ProfileSettings::SpecialModes::kEnableSpiralize);
    m_layer_start = true;
    QString rv;
    rv += "M1 " % commentLine("OPTIONAL STOP - LAYER CHANGE");
    return rv;
}

QString TormachWriter::writeBeforePart(QVector3D normal)
{
    QString rv;
    return rv;
}

QString TormachWriter::writeBeforeIsland()
{
    QString rv;
    return rv;
}

QString TormachWriter::writeBeforeScan(Point min, Point max, int layer, int boundingBox, Axis axis, Angle angle)
{
    QString rv;
    return rv;
}

QString TormachWriter::writeBeforeRegion(RegionType type, int pathSize)
{
    QString rv;
    return rv;
}

QString TormachWriter::writeBeforePath(RegionType type)
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
        else
        {
        }
    }
    return rv;
}

QString TormachWriter::writeTravel(Point start_location, Point target_location, TravelLiftType lType,
                                QSharedPointer<SettingsBase> params)
{
    QString rv;

    Point new_start_location;
    RegionType rType = params->setting<RegionType>(Constants::SegmentSettings::kRegionType);

    m_tip_wipe = false;

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

        rv += m_G0 % writeCoordinates(lift_destination) % commentSpaceLine("TRAVEL LIFT Z");
        setFeedrate(m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kZSpeed));
    }

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
        rv += m_G0 % writeCoordinates(target_location) % commentSpaceLine("TRAVEL LOWER Z");
        setFeedrate(m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kZSpeed));
    }

    return rv;
}

QString TormachWriter::writeLine(const Point& start_point, const Point& target_point, const QSharedPointer<SettingsBase> params)
{
    Velocity speed = params->setting<Velocity>(Constants::SegmentSettings::kSpeed);
    int rpm = params->setting<int>(Constants::SegmentSettings::kExtruderSpeed);
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
            // Set Feedrate to 0 if turning extruder on so that an F parameter
            // is issued with the first G1 of the path
            setFeedrate(0);

        }
    }
    if ((path_modifiers == PathModifiers::kForwardTipWipe || path_modifiers == PathModifiers::kReverseTipWipe
            || path_modifiers == PathModifiers::kPerimeterTipWipe) && m_extruders_on[0] && !m_tip_wipe)
    {
        m_tip_wipe = true;
        rv += commentLine("UPDATE VOLTAGE FOR TIP WIPE");
    }

    rv += m_G1;
    //update feedrate and speed if needed
    if (getFeedrate() != speed || m_layer_start)
    {
        setFeedrate(speed);
        rv += m_f % QString::number(speed.to(m_meta.m_velocity_unit));
        m_layer_start = false;
    }

    if (rpm != m_current_rpm)
    {
        rv += m_s % QString::number(rpm);
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

QString TormachWriter::writeArc(const Point &start_point,
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
        rv += m_s % QString::number(rpm);
        m_current_rpm = rpm;
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

    // Add comment for gcode parser
    if (path_modifiers != PathModifiers::kNone)
        rv += commentSpaceLine(toString(region_type) % m_space % toString(path_modifiers));
    else
        rv += commentSpaceLine(toString(region_type));

    return rv;
}

QString TormachWriter::writeSpline(const Point &start_point,
                                const Point &a_control_point,
                                const Point &b_control_point,
                                const Point &end_point,
                                const QSharedPointer<SettingsBase> params)
{
    QString rv;
    return rv;
}

QString TormachWriter::writeScan(Point target_point, Velocity speed, bool on_off)
{
    QString rv;
    return rv;
}

QString TormachWriter::writeAfterPath(RegionType type)
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

QString TormachWriter::writeAfterRegion(RegionType type)
{
    QString rv;
    return rv;
}

QString TormachWriter::writeAfterScan(Distance beadWidth, Distance laserStep, Distance laserResolution)
{
    QString rv;
    return rv;
}

QString TormachWriter::writeAfterIsland()
{
    QString rv;
    return rv;
}

QString TormachWriter::writeAfterPart()
{
    QString rv;
    return rv;
}

QString TormachWriter::writeAfterLayer()
{
    QString rv;
    rv += m_sb->setting< QString >(Constants::PrinterSettings::GCode::kLayerCodeChange) % m_newline;
    return rv;
}

QString TormachWriter::writeShutdown()
{
    QString rv;
    rv += m_M5 % commentSpaceLine("TURN EXTRUDER OFF END OF PRINT") %
          writeTamperOff();

    rv += m_sb->setting< QString >(Constants::PrinterSettings::GCode::kEndCode) % m_newline %
          "M30" % commentSpaceLine("END OF G-CODE");
    return rv;
}

QString TormachWriter::writePurge(int RPM, int duration, int delay)
{
    QString rv;
    return rv;
}

QString TormachWriter::writeDwell(Time time)
{
    if (time > 0)
        return m_G4 % m_p % QString::number(time.to(m_meta.m_time_unit), 'f', 4) % commentSpaceLine("DWELL");
    else
        return {};
}

QString TormachWriter::writeTamperOn()
{
    QString rv;
    return rv;
}

QString TormachWriter::writeTamperOff()
{
    QString rv;
    return rv;
}

QString TormachWriter::writeExtruderOn(RegionType type, int rpm, int extruder_number)
{
    QString rv;
    m_extruders_on[extruder_number] = true;
    float output_rpm;

    rv += writeTamperOn();

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

QString TormachWriter::writeExtruderOff(int extruder_number)
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

QString TormachWriter::writeCoordinates(Point destination)
{
    QString rv;

    //always specify X and Y
    rv += m_x % QString::number(Distance(destination.x()).to(m_meta.m_distance_unit), 'f', 4) %
          m_y % QString::number(Distance(destination.y()).to(m_meta.m_distance_unit), 'f', 4);

    //write vertical coordinate along the correct axis (Z or W) according to printer settings
    //only output Z/W coordinate if there was a change in Z/W
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
