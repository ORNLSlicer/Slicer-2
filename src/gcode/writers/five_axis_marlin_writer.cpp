#include "gcode/writers/five_axis_marlin_writer.h"
#include <QStringBuilder>
#include "utilities/enums.h"
#include "utilities/mathutils.h"

namespace ORNL
{
FiveAxisMarlinWriter::FiveAxisMarlinWriter(GcodeMeta meta, const QSharedPointer<SettingsBase>& sb) : WriterBase(meta, sb) {

}

QString FiveAxisMarlinWriter::writeInitialSetup(Distance minimum_x, Distance minimum_y, Distance maximum_x, Distance maximum_y, int num_layers)
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
        rv += "M140 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kBed).to(degC))
              % commentSpaceLine("SET BED TEMPERATURE");
        if (m_sb->setting< int >(Constants::MaterialSettings::Temperatures::kBed) > 0)
        {
            rv += "M190 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kBed).to(degC))
                  % commentSpaceLine("SET BED TEMPERATURE AND WAIT");
        }

        if(m_sb->setting< bool >(Constants::MaterialSettings::Temperatures::kTwoZones))
        {
            rv += "M104 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder0Zone1).to(degC))
                  % " T0" % commentSpaceLine("SET EXTRUDER 0 ZONE 1 TEMPERATURE");
            rv += "M104 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder0Zone2).to(degC))
                  % " T1" % commentSpaceLine("SET EXTRUDER 0 ZONE 2 TEMPERATURE");
            rv += "M109 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder1Zone1).to(degC))
                  % " T0" % commentSpaceLine("SET EXTRUDER 1 ZONE 1 TEMPERATURE");
            rv += "M109 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder1Zone2).to(degC))
                  % " T1" % commentSpaceLine("SET EXTRUDER 1 ZONE 2 TEMPERATURE");
        }
        else if(m_sb->setting< bool >(Constants::MaterialSettings::Temperatures::kThreeZones))
        {
            rv += "M104 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder0Zone1).to(degC))
                  % " T0" % commentSpaceLine("SET EXTRUDER 0 ZONE 1 TEMPERATURE");
            rv += "M104 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder0Zone2).to(degC))
                  % " T1" % commentSpaceLine("SET EXTRUDER 0 ZONE 2 TEMPERATURE");
            rv += "M104 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder0Zone3).to(degC))
                  % " T2" % commentSpaceLine("SET EXTRUDER 0 ZONE 3 TEMPERATURE");
            rv += "M109 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder1Zone1).to(degC))
                  % " T0" % commentSpaceLine("SET EXTRUDER 1 ZONE 1 TEMPERATURE");
            rv += "M109 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder1Zone2).to(degC))
                  % " T1" % commentSpaceLine("SET EXTRUDER 1 ZONE 2 TEMPERATURE");
            rv += "M109 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder1Zone3).to(degC))
                  % " T2" % commentSpaceLine("SET EXTRUDER 1 ZONE 3 TEMPERATURE");
        }
        else if(m_sb->setting< bool >(Constants::MaterialSettings::Temperatures::kFourZones))
        {
            rv += "M104 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder0Zone1).to(degC))
                  % " T0" % commentSpaceLine("SET EXTRUDER 0 ZONE 1 TEMPERATURE");
            rv += "M104 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder0Zone2).to(degC))
                  % " T1" % commentSpaceLine("SET EXTRUDER 0 ZONE 2 TEMPERATURE");
            rv += "M104 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder0Zone3).to(degC))
                  % " T2" % commentSpaceLine("SET EXTRUDER 0 ZONE 3 TEMPERATURE");
            rv += "M104 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder0Zone4).to(degC))
                  % " T3" % commentSpaceLine("SET EXTRUDER 0 ZONE 4 TEMPERATURE");
            rv += "M109 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder1Zone1).to(degC))
                  % " T0" % commentSpaceLine("SET EXTRUDER 1 ZONE 1 TEMPERATURE");
            rv += "M109 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder1Zone2).to(degC))
                  % " T1" % commentSpaceLine("SET EXTRUDER 1 ZONE 2 TEMPERATURE");
            rv += "M109 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder1Zone3).to(degC))
                  % " T2" % commentSpaceLine("SET EXTRUDER 1 ZONE 3 TEMPERATURE");
            rv += "M109 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder1Zone4).to(degC))
                  % " T3" % commentSpaceLine("SET EXTRUDER 1 ZONE 4 TEMPERATURE");
        }
        else if(m_sb->setting< bool >(Constants::MaterialSettings::Temperatures::kFiveZones))
        {
            rv += "M104 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder0Zone1).to(degC))
                  % " T0" % commentSpaceLine("SET EXTRUDER 0 ZONE 1 TEMPERATURE");
            rv += "M104 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder0Zone2).to(degC))
                  % " T1" % commentSpaceLine("SET EXTRUDER 0 ZONE 2 TEMPERATURE");
            rv += "M104 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder0Zone3).to(degC))
                  % " T2" % commentSpaceLine("SET EXTRUDER 0 ZONE 3 TEMPERATURE");
            rv += "M104 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder0Zone4).to(degC))
                  % " T3" % commentSpaceLine("SET EXTRUDER 0 ZONE 4 TEMPERATURE");
            rv += "M104 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder0Zone5).to(degC))
                  % " T4" % commentSpaceLine("SET EXTRUDER 0 ZONE 5 TEMPERATURE");
            rv += "M109 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder1Zone1).to(degC))
                  % " T0" % commentSpaceLine("SET EXTRUDER 1 ZONE 1 TEMPERATURE");
            rv += "M109 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder1Zone2).to(degC))
                  % " T1" % commentSpaceLine("SET EXTRUDER 1 ZONE 2 TEMPERATURE");
            rv += "M109 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder1Zone3).to(degC))
                  % " T2" % commentSpaceLine("SET EXTRUDER 1 ZONE 3 TEMPERATURE");
            rv += "M109 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder1Zone4).to(degC))
                  % " T3" % commentSpaceLine("SET EXTRUDER 1 ZONE 4 TEMPERATURE");
            rv += "M109 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder1Zone5).to(degC))
                  % " T4" % commentSpaceLine("SET EXTRUDER 1 ZONE 5 TEMPERATURE");
        }
        else
        {
            rv += "M104 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder0).to(degC))
                  % " T0" % commentSpaceLine("SET EXTRUDER 0 TEMPERATURE");
            rv += "M109 S" % QString::number(m_sb->setting< Temperature >(Constants::MaterialSettings::Temperatures::kExtruder1).to(degC))
                  % " T0" % commentSpaceLine("SET EXTRUDER 1 TEMPERATURE");
        }


        rv += "G28" % commentSpaceLine("TRAVEL HOME ALL AXES");

        if (m_sb->setting< int >(Constants::MaterialSettings::Filament::kRelative) > 0)
        {
            rv += "M83" % commentSpaceLine("USE RELATIVE EXTRUSION DISTANCES");
        }
        else
        {
            rv += "M82" % commentSpaceLine("USE ABSOLUTE EXTRUSION DISTANCES");
        }

        // Cooling fan
        if(m_sb->setting< bool >(Constants::MaterialSettings::Cooling::kEnable))
            // Marlin index the fan from 0 to 255
            rv += "M106 S" % QString::number((m_sb->setting<int>(Constants::MaterialSettings::Cooling::kMaxSpeed) / 100) * 255) % commentSpaceLine("ENABLE FAN");
        else
            rv += "M107" % commentSpaceLine("DISABLE COOLING FAN");

        rv += m_newline;
    }

    if(!m_sb->setting< bool >(Constants::MaterialSettings::Filament::kDisableG92))
    {
        if(m_sb->setting< bool >(Constants::MaterialSettings::Filament::kFilamentBAxis))
        {
            rv += "G92 B0" % commentSpaceLine("RESET FILAMENT TO 0");
        }
        else
        {
            rv += "G92 E0" % commentSpaceLine("RESET FILAMENT TO 0");
        }
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

QString FiveAxisMarlinWriter::writeBeforeLayer(float new_min_z, QSharedPointer<SettingsBase> sb)
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

QString FiveAxisMarlinWriter::writeBeforePart(QVector3D normal)
{
    QString rv;
    return rv;
}

QString FiveAxisMarlinWriter::writeBeforeIsland()
{
    QString rv;
    return rv;
}

QString FiveAxisMarlinWriter::writeBeforeRegion(RegionType type, int pathSize)
{
    QString rv;
    return rv;
}

QString FiveAxisMarlinWriter::writeBeforePath(RegionType type)
{
    QString rv;
    if(!m_sb->setting<bool>(Constants::ProfileSettings::SpecialModes::kEnableSpiralize) || m_first_print)
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
    if (m_filament_location > 0 && !m_sb->setting< bool >(Constants::MaterialSettings::Filament::kDisableG92))
    {
        if(m_sb->setting< bool >(Constants::MaterialSettings::Filament::kFilamentBAxis))
        {
            rv += "G92 B0" % commentSpaceLine("RESET FILAMENT TO 0");
        }
        else
        {
            rv += "G92 E0" % commentSpaceLine("RESET FILAMENT TO 0");
        }
        m_filament_location = 0.0;
    }

    return rv;
}

QString FiveAxisMarlinWriter::writeTravel(Point start_location, Point target_location, TravelLiftType lType,
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
    if (travel_lift_required && (lType == TravelLiftType::kBoth || lType == TravelLiftType::kLiftUpOnly) && !m_first_travel)
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

QString FiveAxisMarlinWriter::writeLine(const Point& start_point, const Point& target_point, const QSharedPointer<SettingsBase> params)
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
            if(m_sb->setting< bool >(Constants::MaterialSettings::Filament::kFilamentBAxis))
            {
                rv += m_b % QString::number(Distance(segment_filament_length).to(m_meta.m_distance_unit), 'f', 4);
            }
            else
            {
                rv += m_e % QString::number(Distance(segment_filament_length).to(m_meta.m_distance_unit), 'f', 4);
            }
        }
        //if using absolute E, update total length and write value
        else
        {
            m_filament_location += segment_filament_length;
            if(m_sb->setting< bool >(Constants::MaterialSettings::Filament::kFilamentBAxis))
            {
                rv += m_b % QString::number(Distance(m_filament_location).to(m_meta.m_distance_unit), 'f', 4);
            }
            else
            {
                rv += m_e % QString::number(Distance(m_filament_location).to(m_meta.m_distance_unit), 'f', 4);
            }
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

QString FiveAxisMarlinWriter::writeArc(const Point &start_point,
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
            if(m_sb->setting< bool >(Constants::MaterialSettings::Filament::kFilamentBAxis))
            {
                rv += m_b % QString::number(segment_filament_length.to(m_meta.m_distance_unit));
            }
            else
            {
                rv += m_e % QString::number(segment_filament_length.to(m_meta.m_distance_unit));
            }
        }
        //if using absolute E, update total length and write value
        else
        {
            m_filament_location += segment_filament_length;
            if(m_sb->setting< bool >(Constants::MaterialSettings::Filament::kFilamentBAxis))
            {
                rv += m_b % QString::number(m_filament_location.to(m_meta.m_distance_unit));
            }
            else
            {
                rv += m_e % QString::number(m_filament_location.to(m_meta.m_distance_unit));
            }
        }

    }

    // Add comment for gcode parser
    if (path_modifiers != PathModifiers::kNone)
        rv += commentSpaceLine(toString(region_type) % m_space % toString(path_modifiers));
    else
        rv += commentSpaceLine(toString(region_type));

    return rv;
}

QString FiveAxisMarlinWriter::writeAfterPath(RegionType type)
{
    QString rv;
    if(!m_sb->setting<bool>(Constants::ProfileSettings::SpecialModes::kEnableSpiralize))
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

QString FiveAxisMarlinWriter::writeAfterRegion(RegionType type)
{
    QString rv;
    return rv;
}

QString FiveAxisMarlinWriter::writeAfterIsland()
{
    QString rv;
    return rv;
}

QString FiveAxisMarlinWriter::writeAfterPart()
{
    QString rv;
    return rv;
}

QString FiveAxisMarlinWriter::writeAfterLayer()
{
    QString rv;

    // 11/8/23 - Marlin purge based on force minimum layer time has been added to common parser
    /*if(m_sb->setting< bool >(Constants::MaterialSettings::Cooling::kForceMinLayerTime) &&
            ForceMinimumLayerTime::kUse_Purge_Dwells == m_sb->setting< ForceMinimumLayerTime >(Constants::MaterialSettings::Cooling::kForceMinLayerTimeMethod))
            rv = writePurge(0,0,0);*/

    rv += m_sb->setting< QString >(Constants::PrinterSettings::GCode::kLayerCodeChange) % m_newline;
    return rv;
}

QString FiveAxisMarlinWriter::writeShutdown()
{
    QString rv;
    rv += "G28" % commentSpaceLine("TRAVEL HOME ALL AXES");

    rv += m_sb->setting< QString >(Constants::PrinterSettings::GCode::kEndCode) % m_newline %
          "M104 S0" % commentSpaceLine("TURN EXTRUDER OFF") %
          "M140 S0" % commentSpaceLine("TURN BED OFF") %
          "M84" % commentSpaceLine("DISABLE MOTORS");
    return rv;
}

QString FiveAxisMarlinWriter::writePurge(int RPM, int duration, int delay)
{
    QString rv;

    QVector3D travel_lift = getTravelLift();
    auto liftZ = Distance(travel_lift.z()).to(m_meta.m_distance_unit) + m_current_z.to(m_meta.m_distance_unit);
    auto purgeZ = m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kPurgeZ).to(m_meta.m_distance_unit);

    rv += m_G1 % m_f % QString::number(m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kZSpeed).to(m_meta.m_velocity_unit))
          % m_z % QString::number(liftZ > purgeZ ? liftZ : purgeZ) % commentSpaceLine("TRAVEL LIFT Z");

    rv += m_G1 % m_x % QString::number(m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kPurgeX).to(m_meta.m_distance_unit))
          % m_y % QString::number(m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kPurgeY).to(m_meta.m_distance_unit))
          % commentSpaceLine("TRAVEL");

    if(liftZ > purgeZ)
        rv += m_G1 % m_z % QString::number(purgeZ)
              % commentSpaceLine("TRAVEL LOWER Z");

    auto length = m_sb->setting< Distance >(Constants::MaterialSettings::Purge::kPurgeLength).to(m_meta.m_distance_unit);
    if (m_sb->setting< bool >(Constants::MaterialSettings::Filament::kRelative))
        m_filament_location = length;
    else
        m_filament_location += length;

    if(m_sb->setting< bool >(Constants::MaterialSettings::Filament::kFilamentBAxis))
    {
        rv += m_G1 % m_f % QString::number(m_sb->setting< Velocity >(Constants::MaterialSettings::Purge::kPurgeFeedrate).to(m_meta.m_velocity_unit))
              % m_b % QString::number(Distance(m_filament_location).to(m_meta.m_distance_unit), 'f', 4)
              % commentSpaceLine("PURGE");
    }
    else
    {
        rv += m_G1 % m_f % QString::number(m_sb->setting< Velocity >(Constants::MaterialSettings::Purge::kPurgeFeedrate).to(m_meta.m_velocity_unit))
              % m_e % QString::number(Distance(m_filament_location).to(m_meta.m_distance_unit), 'f', 4)
              % commentSpaceLine("PURGE");
    }

    if(liftZ > purgeZ)
        rv += m_G1 % m_z % QString::number(liftZ)
              % commentSpaceLine("TRAVEL LIFT Z");

    return rv;
}

QString FiveAxisMarlinWriter::writeDwell(Time time)
{
    if (time > 0)
        return m_G4 % m_p % QString::number(time.to(m_meta.m_time_unit), 'f', 4) % commentSpaceLine("DWELL");
    else
        return {};
}

QString FiveAxisMarlinWriter::writeCoordinates(Point destination)
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

QString FiveAxisMarlinWriter::writeRetraction()
{
    QString rv;

    if(m_filament_location >= 0 && m_sb->setting< bool >(Constants::MaterialSettings::Retraction::kEnable))
    {
        if(m_filament_location != 0 && !m_sb->setting< bool >(Constants::MaterialSettings::Filament::kDisableG92))
        {
            if(m_sb->setting< bool >(Constants::MaterialSettings::Filament::kFilamentBAxis))
            {
                rv += "G92 B0" % commentSpaceLine("RESET FILAMENT TO 0");
            }
            else
            {
                rv += "G92 E0" % commentSpaceLine("RESET FILAMENT TO 0");
            }
            m_filament_location = 0.0;
        }

        if(m_sb->setting< bool >(Constants::MaterialSettings::Filament::kFilamentBAxis))
        {
            rv += m_G1 % m_f %
                  QString::number(m_sb->setting< Velocity >(Constants::MaterialSettings::Retraction::kSpeed).to(m_meta.m_velocity_unit)) %
                  m_b % QString::number(m_sb->setting< Distance >(Constants::MaterialSettings::Retraction::kLength).to(m_meta.m_distance_unit) * -1) %
                  commentSpaceLine("RETRACT FILAMENT");
        }
        else
        {
            rv += m_G1 % m_f %
                  QString::number(m_sb->setting< Velocity >(Constants::MaterialSettings::Retraction::kSpeed).to(m_meta.m_velocity_unit)) %
                  m_e % QString::number(m_sb->setting< Distance >(Constants::MaterialSettings::Retraction::kLength).to(m_meta.m_distance_unit) * -1) %
                  commentSpaceLine("RETRACT FILAMENT");
        }


        if(m_sb->setting< bool >(Constants::MaterialSettings::Filament::kRelative))
            m_filament_location = -1 * m_sb->setting< Distance >(Constants::MaterialSettings::Retraction::kLength);
        else
            m_filament_location -= m_sb->setting< Distance >(Constants::MaterialSettings::Retraction::kLength);
    }

    return rv;
}

QString FiveAxisMarlinWriter::writePrime()
{
    QString rv;

    if (m_sb->setting< bool >(Constants::MaterialSettings::Filament::kRelative))
    {
        m_filament_location = m_sb->setting< Distance >(Constants::MaterialSettings::Retraction::kLength) +
                              m_sb->setting< Distance >(Constants::MaterialSettings::Retraction::kPrimeAdditionalLength);
        if(m_sb->setting< bool >(Constants::MaterialSettings::Filament::kFilamentBAxis))
        {
            rv += m_G1 % m_f %
                  QString::number(m_sb->setting< Velocity >(Constants::MaterialSettings::Retraction::kPrimeSpeed).to(m_meta.m_velocity_unit)) %
                  m_b % QString::number(m_filament_location.to(m_meta.m_distance_unit)) %
                  commentSpaceLine("PRIME FILAMENT");
        }
        else
        {
            rv += m_G1 % m_f %
                  QString::number(m_sb->setting< Velocity >(Constants::MaterialSettings::Retraction::kPrimeSpeed).to(m_meta.m_velocity_unit)) %
                  m_e % QString::number(m_filament_location.to(m_meta.m_distance_unit)) %
                  commentSpaceLine("PRIME FILAMENT");
        }

    }
    else
    {
        m_filament_location += (m_sb->setting< Distance >(Constants::MaterialSettings::Retraction::kLength) +
                                m_sb->setting< Distance >(Constants::MaterialSettings::Retraction::kPrimeAdditionalLength));
        if(m_sb->setting< bool >(Constants::MaterialSettings::Filament::kFilamentBAxis))
        {
            rv += m_G1 % m_f %
                  QString::number(m_sb->setting< Velocity >(Constants::MaterialSettings::Retraction::kPrimeSpeed).to(m_meta.m_velocity_unit)) %
                  m_b % QString::number(m_filament_location.to(m_meta.m_distance_unit)) %
                  commentSpaceLine("PRIME FILAMENT");
        }
        else
        {
            rv += m_G1 % m_f %
                  QString::number(m_sb->setting< Velocity >(Constants::MaterialSettings::Retraction::kPrimeSpeed).to(m_meta.m_velocity_unit)) %
                  m_e % QString::number(m_filament_location.to(m_meta.m_distance_unit)) %
                  commentSpaceLine("PRIME FILAMENT");
        }

    }
    return rv;
}

QString FiveAxisMarlinWriter::setTools(QVector<int> extruders)
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
            rv += "T" % QString::number(i) % m_newline;
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
