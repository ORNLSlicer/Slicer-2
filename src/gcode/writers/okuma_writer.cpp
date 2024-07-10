#include <QStringBuilder>

#include "gcode/writers/okuma_writer.h"
#include "utilities/enums.h"
#include "utilities/mathutils.h"

namespace ORNL
{
    OkumaWriter::OkumaWriter(GcodeMeta meta, const QSharedPointer<SettingsBase>& sb) : WriterBase(meta, sb) {

    }

    QString OkumaWriter::writeInitialSetup(Distance minimum_x, Distance minimum_y, Distance maximum_x, Distance maximum_y, int num_layers)
    {
        m_current_z = m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kZOffset);
        m_current_w = m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kWMax);
        m_current_rpm = 0;
        m_extruders_on[0] = false;
        m_first_travel = true;
        m_first_print = true;
        m_layer_start = true;
        m_min_z = 0.0f;
        m_material_number = -1;
        QString rv;
        rv += "( --- header.txt --- )" % m_newline;
        if (m_sb->setting< int >(Constants::PrinterSettings::GCode::kEnableStartupCode))
        {
            rv += "( -------------------- laser_toolchange.txt --- )" % m_newline %
                    "T100 M6" % m_newline %
                    "( -------------------- )" % m_newline %
                    "( -------------------- MOVED FROM LASER TECH FILE --- )" % m_newline %
                    "(DISK_SPEED      XX  = DISK_SPEED)" % m_newline %
                    "(CARRIER_GAS     5   = CARRIER_GAS)" % m_newline %
                    "(SHIELDING	10	= hyperMILL_AM_SHIELDING_GAS)" % m_newline %
                    "M369    (MIST COLLECTOR ON)" % m_newline %
                    "M130                (OK TO RUN WITH NO SPINDLE RPM)" % m_newline %
                    "M1510               (LASER HEAD DOWN)" % m_newline %
                    "/G115 SPS=3.5  (SPOT SIZE SET TO VARIABLE)" % m_newline %
                    "/M1507             (FOCAL CONTROL ON. ALLOWS FOCAL POINT CHANGE)" % m_newline %
                    "/LHD=VC100         (ACTUAL SPOT SIZE CHANGE. VC100 IS SET BY G115 SPS=)" % m_newline %
                    "/LPWW=2000         (SET VARIABLE FOR LASER POWER, WATTS)" % m_newline %
                    "/M1500             (CHANGE FOCAL POINT)" % m_newline %
                    "/M1508             (FOCAL CONTROL OFF)" % m_newline %
                    "DW = 0.01 (Dwell time between border and layer)" % m_newline %
                    "/M1501             (LASER READY)" % m_newline %
                    "/LPW=0             (LASER POWER IS ZERO)" % m_newline %
                    "/M1541             (HOPPER-OFF: HOPPER_OFF)" % m_newline %
                    "/M1503     (LASER ON)" % m_newline %
                    "( -------------------- MOVED FROM LASER TECH FILE --- )" % m_newline %
                    "( --- files_x\\job_start.txt --- )" % m_newline %
                    "( OPERATION 2 )" % m_newline %
                    "( T100 Additive Manufacturing )" % m_newline %
                    "( --- )" % m_newline %
                    "M510 (CAS OFF)" % m_newline %
                    "G00 G17 G21 G40 G80 G90" % m_newline % m_newline %
                    "( --- sP.txt ---)" % m_newline %
                    "G0 Z=VPSLZ" % m_newline %
                    "G15 H1" % m_newline %
                    "M11" % m_newline %
                    "M27" % m_newline %
                    "( --- rotate_rotary_axes_additive.txt ---)" % m_newline %
                    "G611 HL=1" % m_newline %
                    "G90 G00 A0. C0.        (USER DEFINED)" % m_newline %
                    "X0. Y0.                (USER DEFINED)" % m_newline %
                    "C0" % m_newline %
                    "A0" % m_newline %
                    "G612" % m_newline %
                    "( --- )" % m_newline %
                    "()" % m_newline %
                    "G169 HL=1 X4.082 Y-52.5458 Z136. A0 C0 " % m_newline %
                    "( --- 5X_begin.txt --- )" % m_newline %
                    "G130 (TURN OFF SUPERNURBS)" % m_newline %
                    "G0 X4.082 Y-52.5458 Z136.0" % m_newline %
                    "G0 Z0.0503 A0 C0 " % m_newline;
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

        if(m_sb->setting< QString >(Constants::PrinterSettings::GCode::kStartCode) != "")
            rv += m_sb->setting< QString >(Constants::PrinterSettings::GCode::kStartCode);

        rv += m_newline;

        rv += commentLine("LAYER COUNT: " % QString::number(num_layers));

        return rv;
    }

    QString OkumaWriter::writeBeforeLayer(float new_min_z, QSharedPointer<SettingsBase> sb)
    {
        m_spiral_layer = sb->setting<bool>(Constants::ProfileSettings::SpecialModes::kEnableSpiralize);
        m_layer_start = true;
        QString rv;
        rv += "( -------------------- laser_on.txt --- )" % m_newline;
        return rv;
    }
    QString OkumaWriter::writeBeforePart(QVector3D normal)
    {
        QString rv;
        return rv;
    }

    QString OkumaWriter::writeBeforeIsland()
    {
        QString rv;
        return rv;
    }

    QString OkumaWriter::writeBeforeRegion(RegionType type, int pathSize)
    {
        QString rv;
        return rv;
    }

    QString OkumaWriter::writeBeforePath(RegionType type)
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

    QString OkumaWriter::writeTravel(Point start_location, Point target_location, TravelLiftType lType,
                                    QSharedPointer<SettingsBase> params)
    {
        QString rv;

        Point new_start_location;
        //check if this is the first travel
        if(m_first_travel)
        {
            m_inner_perimeter = false;
            m_outer_perimeter = false;
        }

        //if the last line segment was an inner perimeter on the bottom half of the hourglass, we need to rotate the A axis back to 0 by segmenting the travel
        if(m_inner_perimeter && start_location.y() >= -13096 && start_location.y() <= 13140 && start_location.x() >= -12677 && start_location.x() <= 12665 && m_current_z < 17530)
        {
            //adjust the target z to account for the z_offset
            target_location.z(target_location.z() + m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kZOffset)());
            //create a new variable to define the angle of rotation NOTE: This will need to be a setting
            float angle = 45;
            //create a new variable for the maximum allowable tilt per line segment. NOTE: This will need to be a setting
            float max_tilt = 0.5;
            //find the number of line segments required to get to the desired angle
            int num_segments = angle / max_tilt + 1;
            //set a variable to the change in angle for each segment
            float delta_angle = angle / num_segments;
            //find the change in x and y for each segment
            float delta_x = (target_location.x() - start_location.x()) / num_segments;
            float delta_y = (target_location.y() - start_location.y()) / num_segments;

            Point segment_destination;
            //Now we need to break our travel into segments and write each segment to the file, this time we decrease A from 45 to 0
            for(int i = 1; i < num_segments; i++)
            {
                //create a new point that is the destination of the segment
                segment_destination = Point(start_location.x() + delta_x * i, start_location.y() + delta_y * i, target_location.z());
                //write the segment to the file
                rv += m_G0 %
                      m_x % QString::number(Distance(segment_destination.x()).to(m_meta.m_distance_unit), 'f', 4) %
                      m_y % QString::number(Distance(segment_destination.y()).to(m_meta.m_distance_unit), 'f', 4) %
                      m_z % QString::number(Distance(segment_destination.z()).to(m_meta.m_distance_unit), 'f', 4) %
                      " A" % QString::number(angle - delta_angle * i) % commentSpaceLine("TRAVEL");
            }
            rv += m_G0 %
                  m_x % QString::number(Distance(target_location.x()).to(m_meta.m_distance_unit), 'f', 4) %
                  m_y % QString::number(Distance(target_location.y()).to(m_meta.m_distance_unit), 'f', 4) %
                  m_z % QString::number(Distance(target_location.z()).to(m_meta.m_distance_unit), 'f', 4) %
                  " A0" % commentSpaceLine("TRAVEL");
        }
        //if the last line segment was an outer perimeter on the top half of the hourglass, we need to rotate the A axis back to 0 by segmenting the travel
        else if(m_current_z >= 17530 && m_outer_perimeter)
        {
            //adjust the target z to account for the z_offset
            target_location.z(target_location.z() + m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kZOffset)());
            //create a new variable to define the angle of rotation NOTE: This will need to be a setting
            float angle = 45;
            //create a new variable for the maximum allowable tilt per line segment. NOTE: This will need to be a setting
            float max_tilt = 0.5;
            //find the number of line segments required to get to the desired angle
            int num_segments = angle / max_tilt + 1;
            //set a variable to the change in angle for each segment
            float delta_angle = angle / num_segments;
            //find the change in x and y for each segment
            float delta_x = (target_location.x() - start_location.x()) / num_segments;
            float delta_y = (target_location.y() - start_location.y()) / num_segments;

            Point segment_destination;
            //Now we need to break our travel into segments and write each segment to the file, this time we decrease A from 45 to 0
            for(int i = 1; i < num_segments; i++)
            {
                //create a new point that is the destination of the segment
                segment_destination = Point(start_location.x() + delta_x * i, start_location.y() + delta_y * i, target_location.z());
                //write the segment to the file
                rv += m_G0 %
                      m_x % QString::number(Distance(segment_destination.x()).to(m_meta.m_distance_unit), 'f', 4) %
                      m_y % QString::number(Distance(segment_destination.y()).to(m_meta.m_distance_unit), 'f', 4) %
                      m_z % QString::number(Distance(segment_destination.z()).to(m_meta.m_distance_unit), 'f', 4) %
                      " A" % QString::number(angle - delta_angle * i) % commentSpaceLine("TRAVEL");
            }
            rv += m_G0 %
                  m_x % QString::number(Distance(target_location.x()).to(m_meta.m_distance_unit), 'f', 4) %
                  m_y % QString::number(Distance(target_location.y()).to(m_meta.m_distance_unit), 'f', 4) %
                  m_z % QString::number(Distance(target_location.z()).to(m_meta.m_distance_unit), 'f', 4) %
                  " A0" % commentSpaceLine("TRAVEL");
        }
        /*else if(m_outer_perimeter && start_location.z() < 17530 && start_location.z()>760)
        {
            //write nothing. We need to tilt.
        }
        else if(m_layer_start && start_location.z() >= 17530)
        {
            //write nothing. We need to tilt.
        }*/
        else
        {
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

            //write the travel lower (undo the lift)
            if (travel_lift_required && (lType == TravelLiftType::kBoth || lType == TravelLiftType::kLiftLowerOnly))
            {
                rv += m_G0 % writeCoordinates(target_location) % commentSpaceLine("TRAVEL LOWER Z");
                setFeedrate(m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kZSpeed));
            }
        }
        if (m_first_travel) //if this is the first travel
            m_first_travel = false; //update for next one

        //save m_last line as the starting point and the ending point of the travel
        m_last_line = writeCoordinates(new_start_location) + " " + writeCoordinates(target_location);
        //set the was travel variable to true
        was_travel = true;
        m_inner_perimeter = false;
        m_outer_perimeter = false;
        return rv;
    }

    QString OkumaWriter::writeLine(const Point& start_point, const Point& target_point, const QSharedPointer<SettingsBase> params)
    {
        Velocity speed = params->setting<Velocity>(Constants::SegmentSettings::kSpeed);
        int rpm = params->setting<int>(Constants::SegmentSettings::kExtruderSpeed);
        RegionType region_type = params->setting<RegionType>(Constants::SegmentSettings::kRegionType);
        PathModifiers path_modifiers = params->setting<PathModifiers>(Constants::SegmentSettings::kPathModifiers);
        float output_rpm = rpm * m_sb->setting< float >(Constants::PrinterSettings::MachineSpeed::kGearRatio);
        Distance z_offset = m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kZOffset);
        Point segment_destination;
        QString rv;

        if (region_type == RegionType::kPerimeter)
        {
            Distance current_z = start_point.z() + z_offset;
            //check if the travel is moving to the inner perimeter
            if(start_point.y() >= -13096 && start_point.y() <= 13140 && start_point.x() >= -12677 && start_point.x() <= 12665 && current_z < 17530)
            {
                if(was_travel)
                {
                    rv += writeExtruderOff();
                    //G0 set z to be 6 times m_last_z
                    //rv += "G0 Z" % QString::number(target_point.z()*6,'f',4) % commentSpaceLine("TRAVEL");
                    //grab the values from yhe previous saved line in the file. It should be the last travel line
                    QString last_line = m_last_line;
                    //parse the line to get the start and end points
                    QStringList last_line_parts = last_line.split(" ");

                    for(int j = 1; j < last_line_parts.size(); j++)
                    {
                        last_line_parts[j].remove(0, 1);
                    }
                    Point last_start_point;
                    Point last_end_point;
                        if(last_line_parts.size() == 8)
                    {
                        last_start_point = Point(last_line_parts[1].toDouble(), last_line_parts[2].toDouble(), last_line_parts[7].toDouble());
                        last_end_point = Point(last_line_parts[5].toDouble(), last_line_parts[6].toDouble(), last_line_parts[7].toDouble());
                    }
                    else
                    {
                        last_start_point = Point(last_line_parts[1].toDouble(), last_line_parts[2].toDouble(), current_z/1000.0);
                        last_end_point = Point(last_line_parts[4].toDouble(), last_line_parts[5].toDouble(), current_z/1000.0);
                    }
                    //create a new variable to define the angle of rotation NOTE: This will need to be a setting
                    float angle = 45;
                    //find the angle between the target point and the point aligned with x0 on the Okuma.
                    //Add some conditional statements because I am lazy at the moment and on a time crunch RETURN TO THIS
                    //let's handle the case where angle A is positive
                    if (angle > 0 && current_z < 17530)
                    {
                        //if the target point is in the first or second quadrants, we need to move counter-clockwise/clockwise respectively to get to x0.
                        if (start_point.y() >= 0)
                        {
                            //find the angle between the target point and the point aligned with x0 on the Okuma.
                            c_angle = -1.0 * atan(start_point.x()/start_point.y())*57.2958;
                        }
                        //if the target point is in the third or fourth quadrants, we need to move clockwise/counter-clockwise respectively to get to x0.
                        else
                        {
                            //find the angle between the target point and the point aligned with x0 on the Okuma.
                            c_angle = 180 - atan(start_point.x()/start_point.y())*57.2958;
                        }
                    }
                    else if (angle > 0 && current_z >=17530)
                    {
                        //if the target point is in the first or second quadrants, we need to move counter-clockwise/clockwise respectively to get to x0.
                        if (start_point.y() >= 0)
                        {
                            //find the angle between the target point and the point aligned with x0 on the Okuma.
                            c_angle = 180 - atan(start_point.x()/start_point.y())*57.2958;
                        }
                        //if the target point is in the third or fourth quadrants, we need to move clockwise/counter-clockwise respectively to get to x0.
                        else
                        {
                            //find the angle between the target point and the point aligned with x0 on the Okuma.
                            c_angle = -1.0 * atan(start_point.x()/start_point.y())*57.2958;
                        }
                    }
                    else if (angle < 0)
                    {
                        //COME BACK this doesn't matter for right now.
                        //if the target point is in the first or second quadrants, we need to move clockwise/counter-clockwise respectively to get to x0.
                        if (start_point.y() < 0)
                        {
                            //find the angle between the target point and the point aligned with x0 on the Okuma.
                            c_angle = -1.0 * atan(start_point.x()/start_point.y())*57.2958;
                        }
                        //if the target point is in the third or fourth quadrants, we need to move clockwise/counter-clockwise respectively to get to x0.
                        else
                        {
                            //find the angle between the target point and the point aligned with x0 on the Okuma.
                            c_angle = 180 - atan(start_point.x()/start_point.y())*57.2958;
                        }
                    }
                    else
                    {
                        c_angle = 0;
                    }
                    //format c_angle to have 5 decimal points
                    std::ostringstream out;
                    out << std::fixed << std::setprecision(5) << c_angle;
                    // Convert the string stream to double
                    c_angle = std::stod(out.str());
                    //create a new variable for the maximum allowable tilt per line segment. NOTE: This will need to be a setting
                    float max_tilt = 0.5;
                    //find the number of line segments required to get to the desired angle
                    int num_segments = angle / max_tilt + 1;
                    //set a variable to the change in angle for each segment
                    float delta_angle = angle / num_segments;
                    //find the change in x and y for each segment
                    float delta_x = (last_end_point.x() - last_start_point.x()) / num_segments;
                    float delta_y = (Distance(start_point.y()).to(m_meta.m_distance_unit) - last_start_point.y()) / num_segments;

                    //Now we need to break our travel into segments and write each segment to the file
                    for(int i = 1; i < num_segments; i++)
                    {
                        //create a new point that is the destination of the segment
                        segment_destination = Point(last_start_point.x() + delta_x * i, last_start_point.y() + delta_y * i, last_end_point.z());
                        //write the segment to the file
                        rv += m_G0 % writeBrokenTravel(segment_destination)% " A" % QString::number(delta_angle * i) % commentSpaceLine("TRAVEL");
                    }
                    segment_destination = Point(Distance(start_point.x()).to(m_meta.m_distance_unit), Distance(start_point.y()).to(m_meta.m_distance_unit), segment_destination.z());
                    rv += m_G0 % writeBrokenTravel(segment_destination)% " A" % QString::number(angle) % " C" % QString::number(c_angle) % commentSpaceLine("TRAVEL");
                }
                //turn on the extruder if it isn't already on
                if (m_extruders_on[0] == false && rpm > 0)
                {
                    rv += writeExtruderOn(region_type, rpm);
                }
                //start the perimeter. Break it up into sections so that the angle of rotation for each segment is increased by less than 2 degrees
               //find the difference between the start point and the target point
                double xdiff = (target_point.x() - start_point.x());
                double ydiff = (target_point.y() - start_point.y());
                double zdiff = (target_point.z() - start_point.z());
                //change in c
                double cdiff = 0;
                //if x does not change, increment y so that for each segment C gains less than 2 degrees

                //set the origin of the build
                Point origin = Point(0,0,50.2);
                //to find the angle that we need to rotate about c, we use some basic trig
                //the printed perimeter is the same as the arclength of the c-axis. If we draw a line from the origin,through the start point, to an imaginary arced surface
                //(circle surrounding the plane), and a line through the origin and the target point, we can find the angle between the two lines. This arclength angle is
                //the total change in C required for the segment. Because our part here is symmetric, we have an iscoseles triangle. Finding (and doubling) the angle of
                //the iscoseles triangle gives us the angle of rotation required for the segment.
                //TOA: Our angle is equal to the arctan(opposite/adjacent) or the arctan(start-point/mid-point)
                Point mid_point = Point((start_point.x()+target_point.x())/2,(start_point.y()+target_point.y())/2,start_point.z());
                //find the distance between the origin and the start point
                double a = std::sqrt(std::pow(start_point.x() - mid_point.x(),2)+std::pow(start_point.y() - mid_point.y(),2));
                //find the distance between the origin and the mid point
                double b = std::sqrt(std::pow(mid_point.x() - origin.x(),2)+std::pow(mid_point.y() - origin.y(),2));
                //compute the central angle of the rotation arc
                cdiff = 2*atan(a/b)*57.2958;
                //now that we have the change in degrees, we need to break the perimeter into segments so that the change in C is less than 2 degrees
                int num_segments = cdiff / 2 + 1; //2 needs to be a user setting
                //set the change in c for each segment
                double cdelta = cdiff / num_segments;
                // Determine the direction of the rotation
                double crossProduct = start_point.x() * target_point.y() - start_point.y() * target_point.x();
                if (crossProduct >= 0) {
                    cdiff = -cdiff;
                }
                //rv += "Change in C: " % QString::number(cdiff) % " Number of segments: " % QString::number(num_segments) % " Change in C per segment: " % QString::number(cdelta) % "\n";
                //set the change in a and y for each segment
                double xdelta = xdiff / num_segments;
                double ydelta = ydiff / num_segments;
                //increment over the segments and write the coordinates and the C value
                for(int i = 1; i < num_segments; i++)
                {
                    //increment c
                    c_angle -= cdelta;
                    if (c_angle > 360)
                    {
                        c_angle -= 360;
                    }
                    if (c_angle < -360)
                    {
                        c_angle += 360;
                    }
                    segment_destination = Point(start_point.x() + xdelta * i, start_point.y() + ydelta * i, start_point.z());
                    rv += m_G1 % writeCoordinates(segment_destination) % " C" % QString::number(c_angle) % m_f % QString::number(speed.to(m_meta.m_velocity_unit)) % commentSpaceLine("INNER PERIMETER");
                }
                //we want c to rotate in the opposite direction of the print, so we subtract the change in c
                c_angle -= cdelta;
                if (c_angle > 360)
                {
                    c_angle -= 360;
                }
                if (c_angle < -360)
                {
                    c_angle += 360;
                }
                //format c_angle to have 5 decimal points
                std::ostringstream out;
                out << std::fixed << std::setprecision(5) << c_angle;
                // Convert the string stream to double
                c_angle = std::stod(out.str());
                segment_destination = Point(target_point.x(), target_point.y(), target_point.z());
                rv += m_G1 % writeCoordinates(segment_destination) % " C" % QString::number(c_angle) % m_f % QString::number(speed.to(m_meta.m_velocity_unit)) % commentSpaceLine("INNER PERIMETER");
                m_inner_perimeter = true;
                m_outer_perimeter = false;
            }
            else if (start_point.y() >= -13096 && start_point.y() <= 13140 && start_point.x() >= -12677 && start_point.x() <= 12665 && current_z >= 17530)
            {
                //turn on the extruder if it isn't already on
                if (m_extruders_on[0] == false && rpm > 0)
                {
                    rv += writeExtruderOn(region_type, rpm);
                }
                //writes WXYZ to destination
                rv += m_G1 % writeCoordinates(target_point) % m_f % QString::number(speed.to(m_meta.m_velocity_unit)) % commentSpaceLine("INNER PERIMETER");
                m_inner_perimeter = true;
                m_outer_perimeter = false;
            }
            //check if the travel is moving to the outer perimeter on the second half of the hourglass
            else if (current_z >= 17530)
            {
                Distance z_offset = m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kZOffset);
                if(was_travel)
                {
                    rv += writeExtruderOff();
                    //G0 set z to be 6 times m_last_z
                    //rv += "G0 Z" % QString::number(target_point.z()*6,'f',4) % commentSpaceLine("TRAVEL");
                    //grab the values from yhe previous saved line in the file. It should be the last travel line
                    QString last_line = m_last_line;
                    //parse the line to get the start and end points
                    QStringList last_line_parts = last_line.split(" ");

                    for(int j = 1; j < last_line_parts.size(); j++)
                    {
                        last_line_parts[j].remove(0, 1);
                    }
                    Point last_start_point;
                    Point last_end_point;
                    if(last_line_parts.size() == 8)
                    {
                        last_start_point = Point(last_line_parts[1].toDouble(), last_line_parts[2].toDouble(), last_line_parts[7].toDouble());
                        last_end_point = Point(last_line_parts[5].toDouble(), last_line_parts[6].toDouble(), last_line_parts[7].toDouble());
                    }
                    else
                    {
                        last_start_point = Point(last_line_parts[1].toDouble(), last_line_parts[2].toDouble(), current_z/1000.0);
                        last_end_point = Point(last_line_parts[4].toDouble(), last_line_parts[5].toDouble(), current_z/1000.0);
                    }
                    //create a new variable to define the angle of rotation NOTE: This will need to be a setting
                    float angle = 45;
                    if (angle > 0 && current_z < 17530)
                    {
                        //if the target point is in the first or second quadrants, we need to move counter-clockwise/clockwise respectively to get to x0.
                        if (start_point.y() >= 0)
                        {
                            //find the angle between the target point and the point aligned with x0 on the Okuma.
                            c_angle = -1.0 * atan(start_point.x()/start_point.y())*57.2958;
                        }
                        //if the target point is in the third or fourth quadrants, we need to move clockwise/counter-clockwise respectively to get to x0.
                        else
                        {
                            //find the angle between the target point and the point aligned with x0 on the Okuma.
                            c_angle = 180 - atan(start_point.x()/start_point.y())*57.2958;
                        }
                    }
                    else if (angle > 0 && current_z >=17530)
                    {
                        //if the target point is in the first or second quadrants, we need to move counter-clockwise/clockwise respectively to get to x0.
                        if (start_point.y() >= 0)
                        {
                            //find the angle between the target point and the point aligned with x0 on the Okuma.
                            c_angle = 180 - atan(start_point.x()/start_point.y())*57.2958;
                        }
                        //if the target point is in the third or fourth quadrants, we need to move clockwise/counter-clockwise respectively to get to x0.
                        else
                        {
                            //find the angle between the target point and the point aligned with x0 on the Okuma.
                            c_angle = -1.0 * atan(start_point.x()/start_point.y())*57.2958;
                        }
                    }
                    else if (angle < 0)
                    {
                        //COME BACK this doesn't matter for right now.
                        //if the target point is in the first or second quadrants, we need to move clockwise/counter-clockwise respectively to get to x0.
                        if (start_point.y() < 0)
                        {
                            //find the angle between the target point and the point aligned with x0 on the Okuma.
                            c_angle = -1.0 * atan(start_point.x()/start_point.y())*57.2958;
                        }
                        //if the target point is in the third or fourth quadrants, we need to move clockwise/counter-clockwise respectively to get to x0.
                        else
                        {
                            //find the angle between the target point and the point aligned with x0 on the Okuma.
                            c_angle = 180 - atan(start_point.x()/start_point.y())*57.2958;
                        }
                    }
                    else
                    {
                        c_angle = 0;
                    }
                    //format c_angle to have 5 decimal points
                    std::ostringstream out;
                    out << std::fixed << std::setprecision(5) << c_angle;
                    // Convert the string stream to double
                    c_angle = std::stod(out.str());
                    //create a new variable for the maximum allowable tilt per line segment. NOTE: This will need to be a setting
                    float max_tilt = 0.5;
                    //find the number of line segments required to get to the desired angle
                    int num_segments = angle / max_tilt + 1;
                    //set a variable to the change in angle for each segment
                    float delta_angle = angle / num_segments;
                    //find the change in x and y for each segment
                    float delta_x = (last_end_point.x() - last_start_point.x()) / num_segments;
                    float delta_y = (Distance(start_point.y()).to(m_meta.m_distance_unit) - last_start_point.y()) / num_segments;

                    segment_destination;
                    //Now we need to break our travel into segments and write each segment to the file
                    for(int i = 1; i < num_segments; i++)
                    {
                        //create a new point that is the destination of the segment
                        segment_destination = Point(last_start_point.x() + delta_x * i, last_start_point.y() + delta_y * i, last_end_point.z());
                        //write the segment to the file
                        rv += m_G0 % writeBrokenTravel(segment_destination)% " A" % QString::number(delta_angle * i) % commentSpaceLine("TRAVEL");
                    }
                    segment_destination = Point(Distance(start_point.x()).to(m_meta.m_distance_unit), Distance(start_point.y()).to(m_meta.m_distance_unit), segment_destination.z());
                    //write last line of travel
                    rv += m_G0 % writeBrokenTravel(segment_destination)% " A" % QString::number(angle) % " C" % QString::number(c_angle) % m_f % QString::number(speed.to(m_meta.m_velocity_unit)) % commentSpaceLine("TRAVEL");
                    m_inner_perimeter = false;
                    m_outer_perimeter = true;
                }
                //turn on the extruder if it isn't already on
                if (m_extruders_on[0] == false && rpm > 0)
                {
                    rv += writeExtruderOn(region_type, rpm);
                }
                //start the perimeter. Break it up into sections so that the angle of rotation for each segment is increased by less than 2 degrees
                //find the difference between the start point and the target point
                double xdiff = (target_point.x() - start_point.x());
                double ydiff = (target_point.y() - start_point.y());
                double zdiff = (target_point.z() - start_point.z());
                //define the diameter of the table
                //double radius =  27.59*1000; //25.78986*1000; //mm
                //change in c
                double cdiff = 0;
                //set the origin of the build
                Point origin = Point(0,0,50.2);
                //to find the angle that we need to rotate about c, we use some basic trig
                //the printed perimeter is the same as the arclength of the c-axis. If we draw a line from the origin,through the start point, to an imaginary arced surface
                //(circle surrounding the plane), and a line through the origin and the target point, we can find the angle between the two lines using the law of cosines.
                //find the distance between the start point and the end point
                double a = std::sqrt(std::pow(start_point.x() - target_point.x(),2)+std::pow(start_point.y() - target_point.y(),2));
                //find the distance between the origin and the mid point
                double b = std::sqrt(std::pow(target_point.x() - origin.x(),2)+std::pow(target_point.y() - origin.y(),2));
                //find the distance between the start point and the origin
                double c = std::sqrt(std::pow(start_point.x() - origin.x(),2)+std::pow(start_point.y() - origin.y(),2));

                //compute the central angle of the rotation arc
                cdiff = std::acos((std::pow(b,2) + std::pow(c,2) - std::pow(a,2))/(2.0*b*c))*57.2958;

                // Determine the direction of the rotation
                double crossProduct = start_point.x() * target_point.y() - start_point.y() * target_point.x();
                if (crossProduct >= 0) {
                   cdiff = -cdiff;
                }

                //now that we have the change in degrees, we need to break the perimeter into segments so that the change in C is less than 2 degrees
                double deg_allow_c = 2.0; //2 needs to be a user setting
                int num_segments = 1;
                if (cdiff >= deg_allow_c)
                {
                    num_segments = cdiff / deg_allow_c + 1;
                }

                //set the change in c for each segment
                double cdelta = cdiff / num_segments;
                //rv += "Change in C: " % QString::number(cdiff) % " Number of segments: " % QString::number(num_segments) % " Change in C per segment: " % QString::number(cdelta) % "\n";
                //set the change in a and y for each segment
                double xdelta = xdiff / num_segments;
                double ydelta = ydiff / num_segments;
                //increment over the segments and write the coordinates and the C vale
                for(int i = 1; i < num_segments; i++)
                {
                    //we want c to rotate in the opposite direction of the print, so we subtract the change in c
                    c_angle -= cdelta;
                    if (c_angle > 360)
                    {
                        c_angle -= 360;
                    }
                    if (c_angle < -360)
                    {
                        c_angle += 360;
                    }
                    //format c_angle to have 5 decimal points
                    std::ostringstream out;
                    out << std::fixed << std::setprecision(5) << c_angle;
                    // Convert the string stream to double
                    c_angle = std::stod(out.str());
                    segment_destination = Point(start_point.x() + xdelta * i, start_point.y() + ydelta * i, start_point.z());
                    rv += m_G1 % writeCoordinates(segment_destination) % " C" % QString::number(c_angle) % m_f % QString::number(speed.to(m_meta.m_velocity_unit)) % commentSpaceLine("OUTER PERIMETER");
                }
                //we want c to rotate in the opposite direction of the print, so we subtract the change in c
                c_angle -= cdelta;
                if (c_angle > 360)
                {
                    c_angle -= 360;
                }
                if (c_angle < -360)
                {
                    c_angle += 360;
                }
                //format c_angle to have 5 decimal points
                std::ostringstream out;
                out << std::fixed << std::setprecision(5) << c_angle;
                // Convert the string stream to double
                c_angle = std::stod(out.str());
                segment_destination = Point(target_point.x(), target_point.y(), target_point.z());
                rv += m_G1 % writeCoordinates(segment_destination) % " C" % QString::number(c_angle) % m_f % QString::number(speed.to(m_meta.m_velocity_unit)) % commentSpaceLine("OUTER PERIMETER");

                m_outer_perimeter = true;
                m_inner_perimeter = false;
            }
            else
            {
                //turn on the extruder if it isn't already on
                if (m_extruders_on[0] == false && rpm > 0)
                {
                    rv += writeExtruderOn(region_type, rpm);
                }
                //writes WXYZ to destination
                rv += m_G1 % writeCoordinates(target_point) % m_f % QString::number(speed.to(m_meta.m_velocity_unit)) % commentSpaceLine("OUTER PERIMETER");
                m_outer_perimeter = true;
                m_inner_perimeter = false;
            }
        }
        else
        {
            //turn on the extruder if it isn't already on
            if (m_extruders_on[0] == false && rpm > 0)
            {
                rv += writeExtruderOn(region_type, rpm);
            }

            rv += m_G1 % m_f % QString::number(speed.to(m_meta.m_velocity_unit));
            // Forces first motion of layer to issue speed (needed for spiralize mode so that feedrate is scaled properly)
            /*if (m_layer_start)
            {
                setFeedrate(speed);
                rv += m_f % QString::number(speed.to(m_meta.m_velocity_unit));

                m_layer_start = false;
            }*/

            // Update feedrate and extruder speed if needed
            if (getFeedrate() != speed)
            {
                setFeedrate(speed);
                rv += m_f % QString::number(speed.to(m_meta.m_velocity_unit));
            }

            //writes WXYZ to destination
            rv += writeCoordinates(target_point);

            //add comment for gcode parser
            if (path_modifiers != PathModifiers::kNone)
                rv += commentSpaceLine(toString(region_type) % m_space % toString(path_modifiers));
            else
                rv += commentSpaceLine(toString(region_type));
            m_outer_perimeter = false;
            m_inner_perimeter = false;
        }
        m_first_print = false;
        //set was_travel to false since the last segment is now a perimeter.
        was_travel = false;
        return rv;
    }

    QString OkumaWriter::writeArc(const Point &start_point,
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

        // Turn on the extruder if it isn't already on
        if (!m_extruders_on[0] && rpm > 0)
        {
            rv += writeExtruderOn(region_type, rpm);
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

    QString OkumaWriter::writeAfterPath(RegionType type)
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

    QString OkumaWriter::writeAfterRegion(RegionType type)
    {
        QString rv;
        return rv;
    }

    QString OkumaWriter::writeAfterIsland()
    {
        QString rv;
        return rv;
    }

    QString OkumaWriter::writeAfterPart()
    {
        QString rv;
        return rv;
    }

    QString OkumaWriter::writeAfterLayer()
    {
        QString rv;
        rv += m_sb->setting< QString >(Constants::PrinterSettings::GCode::kLayerCodeChange) % m_newline;
        return rv;
    }

    QString OkumaWriter::writeShutdown()
    {
        QString rv;

        rv += "( --- 5X_end.txt --- )" % m_newline;
        rv += "G170 " % m_newline;
        rv += "G0 Z=VPSLZ" % m_newline;
        rv += "()" % m_newline;
        rv += "( --- files_x\\job_end.txt --- )" % m_newline;
        rv += "M511 (CAS ON)" % m_newline;
        rv += "( --- laser_end.txt --- )" % m_newline;
        rv += "G90 G0 A0. C0." % m_newline;
        rv += "/M1504" % m_newline;
        rv += "/M1542" % m_newline;
        rv += "/M1532" % m_newline;
        rv += "/M1522" % m_newline;
        rv += "/M1512" % m_newline;
        rv += "/M1502               (LASER READY OFF)" % m_newline;
        rv += "/M1509               (LASER HEAD UP)" % m_newline;
        rv += "M2" % m_newline;
        rv += "( --- )" % m_newline;
        return rv;
    }

    QString OkumaWriter::writePurge(int RPM, int duration, int delay)
    {
        return {};
    }

    QString OkumaWriter::writeDwell(Time time)
    {
        if (time > 0)
            return m_G4 % m_p % QString::number(time.to(m_meta.m_time_unit), 'f', 4) % commentSpaceLine("DWELL");
        else
            return {};
    }

    QString OkumaWriter::writeExtruderOn(RegionType type, int rpm)
    {
        QString rv;
        m_extruders_on[0] = true;
        rv += "( -------------------- laser_on.txt --- )" % m_newline;
        rv += "/LPW=LPWW        (LASER POWER)" % m_newline;
        rv += "( -------------------- )" % m_newline;
        return rv;
    }

    QString OkumaWriter::writeExtruderOff()
    {
        QString rv;
        m_extruders_on[0] = false;
        rv += "( -------------------- laser_off.txt --- )" % m_newline;
        rv += "/LPW=0               (LASER POWER)" % m_newline;
        rv += "(/)" % m_newline;
        rv += "( -------------------- )" % m_newline;
        rv += "(Laser Tech)" % m_newline;
        return rv;
    }

    QString OkumaWriter::writeCoordinates(Point destination)
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

    QString OkumaWriter::writeBrokenTravel(Point destination)
    {
        QString rv;

        //always specify X and Y
        rv += m_x % QString::number(destination.x(), 'f', 4) %
              m_y % QString::number(destination.y(), 'f', 4) %
              m_z % QString::number(destination.z(), 'f', 4);
        return rv;
    }

}  // namespace ORNL
