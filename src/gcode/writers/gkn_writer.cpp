#include <QStringBuilder>

#include "gcode/writers/gkn_writer.h"
#include "utilities/enums.h"
#include "utilities/mathutils.h"

namespace ORNL
{
    GKNWriter::GKNWriter(GcodeMeta meta, const QSharedPointer<SettingsBase>& sb) : WriterBase(meta, sb) {
        m_laser_prefix = "";
        m_variable_delim = ", ";
        m_zero = "0.000";
        m_v = " V";
        m_A = m_sb->setting<Angle>(Constants::PrinterSettings::MachineSetup::kAxisA);
        m_B = m_sb->setting<Angle>(Constants::PrinterSettings::MachineSetup::kAxisB);
        m_C = m_sb->setting<Angle>(Constants::PrinterSettings::MachineSetup::kAxisC);
        m_E1 = 0.0;
        m_E2 = m_actual_angle = M_PI;
        m_z_axis = QVector3D(0, 0, -1);
        m_ccw = true;
        m_suppress_next_travel_lift = false;
        m_override_travel_lift = false;

        m_can_rotate = sb->setting<bool>(Constants::PrinterSettings::MachineSetup::kSupportsE2);
        m_can_tilt = sb->setting<bool>(Constants::PrinterSettings::MachineSetup::kSupportsE1);

        //For some crazy reason, when using E1, the entire reference frame changes and B must be positive
        //to offset a negative tilt
        if(m_can_tilt)
            m_arm_orientation = MathUtils::CreateQuaternion(m_C, -m_B, m_A, QuaternionOrder::kZYX);
        else
            m_arm_orientation = MathUtils::CreateQuaternion(m_C, m_B, m_A, QuaternionOrder::kZYX);

        if(m_can_rotate || m_can_tilt)
             m_table_orientation = MathUtils::CreateQuaternion(0.0, m_E1, m_E2, QuaternionOrder::kZYX);
    }

    QString GKNWriter::writeSlicerHeader(const QString &syntax)
    {
        return "&ACCESS RVP\n&REL 5\nDEF GKN_gcode()\n" % WriterBase::writeSlicerHeader(syntax);
    }

    QString GKNWriter::writeSettingsHeader(GcodeSyntax syntax)
    {
        QString text = "";
        text += commentLine("Slicing Parameters");

        text += commentLine(
            QString("Wire Diameter: %0mm")
                .arg(m_sb->setting< Distance >(Constants::MaterialSettings::Filament::kDiameter).to(mm)));
        text += commentLine(
            QString("Printer Base Offset: %0mm")
                .arg(m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kZOffset).to(mm)));
        text += commentLine(
            QString("Layer Height: %0mm")
                .arg(m_sb->setting< Distance >(Constants::ProfileSettings::Layer::kLayerHeight).to(mm)));
        text += commentLine(
            QString("Default Extrusion Width: %0mm")
                .arg(m_sb->setting< Distance >(Constants::ProfileSettings::Layer::kBeadWidth).to(mm)));

        if(m_sb->setting< int >(Constants::ProfileSettings::SpecialModes::kEnableSpiralize))
        {
            text += commentLine(QString("Spiralize is turned ON"));
            if (m_sb->setting< int >(Constants::ProfileSettings::SpecialModes::kSmoothing))
                text += commentLine("Smoothing is turned ON");
            if (m_sb->setting< int >(Constants::ProfileSettings::SpecialModes::kEnableOversize))
            {
                text += commentLine(
                    QString("Oversize part by: %0mm")
                        .arg(m_sb->setting< Distance >(Constants::ProfileSettings::SpecialModes::kOversizeDistance).to(mm)));
            }
            text += m_newline;
            return text;
        }
        else {
            text += commentLine(QString("Perimeter Count: %0")
                            .arg(m_sb->setting< int >(Constants::ProfileSettings::Perimeter::kCount)));
            text += commentLine(QString("Inset Count: %0")
                            .arg(m_sb->setting< int >(Constants::ProfileSettings::Inset::kCount)));
            // TODO: This doesn't output the layer time, need to change the setting <int> to Time and convert the setting value to a time... somehow
            if (m_sb->setting< int >(Constants::MaterialSettings::Cooling::kForceMinLayerTime))
                text += commentLine(
                    QString("Forced Minimum / Maximum Layer Time: %0 %1 seconds")
                        .arg(m_sb->setting<Time>(Constants::MaterialSettings::Cooling::kMinLayerTime)())
                        .arg(m_sb->setting<Time>(Constants::MaterialSettings::Cooling::kMaxLayerTime)()));
            if (m_sb->setting< bool >(Constants::ProfileSettings::SpecialModes::kSmoothing))
                text += commentLine("Smoothing is turned ON");
            if (m_sb->setting< int >(Constants::ProfileSettings::SpecialModes::kEnableOversize))
            {
                text += commentLine(
                    QString("Oversize part by: %0mm")
                        .arg(m_sb->setting< Distance >(Constants::ProfileSettings::SpecialModes::kOversizeDistance).to(mm)));
            }
        }

        text += m_newline;
        return text;
    }

    QString GKNWriter::writeInitialSetup(Distance minimum_x, Distance minimum_y, Distance maximum_x, Distance maximum_y, int num_layers)
    {
        m_current_z = m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kZOffset);
        m_current_rpm = 0;
        m_extruders_on[0] = false;
        m_first_travel = true;
        m_first_layer_travel = true;
        m_wire_cut = false;
        m_wipe_tip = true;
        m_min_z = 0.0f;
        QString rv;
        if (m_sb->setting< int >(Constants::PrinterSettings::GCode::kEnableStartupCode))
        {
            rv = QString("&ACCESS RVP\n"
                  "&REL 5\n"
                  "DEF GKN_gcode()\n"
                  "decl real lsr_pwr\n"
                  "decl real wr_speed\n"
                  "decl real print_speed\n"
                  "decl real scan_speed\n"
                  "decl real rapid_speed\n") %
                  commentLine("FOLD INI;%{PE}") %
                  m_newline %
                  commentLine("FOLD BASISTECH INI") %
                  "GLOBAL INTERRUPT DECL 3 WHEN $STOPMESS==TRUE DO IR_STOPM ( )\n" %
                  "INTERRUPT ON 3 \n" %
                  "BAS (#INITMOV,0 )\n" %
                  commentLine("ENDFOLD (BASISTECH INI)") %
                  commentLine("FOLD USER INI") %
                  commentLine("Make your modifications here") %
                  m_newline %
                  commentLine("ENDFOLD (USER INI)") %
                  commentLine("ENDFOLD (INI)") %
                  m_newline %
                  commentLine("Default Values") %
                  QString("lsr_pwr = %0").arg(m_sb->setting< Voltage >(Constants::PrinterSettings::Auxiliary::kGKNLaserPower).to(V)) % m_newline %
                  QString("wire_speed = %0").arg(m_sb->setting< Voltage >(Constants::PrinterSettings::Auxiliary::kGKNWireSpeed).to(V)) % m_newline %
                  QString("print_speed = %0")
                    .arg(m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kGKNPrintSpeed).to(m_meta.m_velocity_unit)) % m_newline %
                  QString("rapid_speed = %0")
                    .arg(m_sb->setting< Velocity >(Constants::ProfileSettings::Travel::kSpeed).to(m_meta.m_velocity_unit)) % m_newline %
                  QString("scan_speed = %0")
                    .arg(m_sb->setting< Velocity >(Constants::ProfileSettings::LaserScanner::kSpeed).to(m_meta.m_velocity_unit)) % m_newline %
                  "$APO.CVEL=100\n" %
                  "$ACC.CP=10\n" %
                  m_newline %
                  commentLine("FOLD PTP P1 Vel=10 % PDAT1 Tool[2]:Side Feed Base[0];%{PE}%R ") %
                  "8.3.40,%MKUKATPBASIS,%CMOVE,%VPTP,%P 1:PTP, 2:P1, 3:, 5:10, 7:PDAT1\n" %
                  "$BWDSTART=FALSE\n" %
                  "PDAT_ACT=PPDAT1\n" %
                  "FDAT_ACT=FP1\n" %
                  "BAS(#PTP_PARAMS,10)\n" %
                  "PTP XP1 \n" %
                  commentLine("ENDFOLD") %
                  m_newline %
                  commentLine("start preparation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx") %
                  "M1() ;initialize lasers\n" %
                  "M2() ;laser program select\n" %
                  "M3() ;argon gas on\n" %
                  "M8(FALSE) ;turn keyence laser scanner shutter off\n" %
                  "M0(TRUE) ; wire cutoff\n" %
                  m_newline %
                  commentLine("Start Build");
        }

        // Borish
        // Laser Scanner
        // IR camera
        /* if (use scale)
         * M233 (RAISE WEIGH STATION)
         * writeDwell(3)
         * G104 P1000 (SCALE CONNECT)
         * G104 P5000 (SCALE TARE)
         */

        if(m_sb->setting<bool>(Constants::ProfileSettings::LaserScanner::kLaserScanner))
        {
            Distance scannerZOffset = m_sb->setting<Distance>(Constants::ProfileSettings::LaserScanner::kLaserScannerHeight)
                    - m_sb->setting<Distance>(Constants::ProfileSettings::LaserScanner::kLaserScannerHeightOffset);

            rv += commentLine("BEGINNING LAYER: 0");
            rv += "M14(" % QString::number(m_sb->setting<Distance>(Constants::ProfileSettings::Layer::kBeadWidth).to(mm), 'f', 4) % m_variable_delim %
                     QString::number(m_sb->setting<Distance>(Constants::ProfileSettings::Layer::kLayerHeight).to(mm), 'f', 4) % m_variable_delim %
                     QString::number(m_sb->setting<Distance>(Constants::ProfileSettings::LaserScanner::kLaserScannerWidth).to(mm), 'f', 4) % m_variable_delim %
                     QString::number(m_sb->setting<Distance>(Constants::ProfileSettings::LaserScanner::kLaserScanLineResolution).to(mm), 'f', 4) % m_variable_delim %
                     QString::number((int)m_sb->setting<bool>(Constants::ProfileSettings::LaserScanner::kInvertLaserScannerHead)) % m_variable_delim %
                     QString::number((int)m_sb->setting<bool>(Constants::ProfileSettings::LaserScanner::kEnableBedScan)) % m_variable_delim %
                     QString::number(m_sb->setting<int>(Constants::ProfileSettings::LaserScanner::kLaserScannerAxis)) % m_variable_delim %
                     QString::number(scannerZOffset.to(mm), 'f', 4) % m_variable_delim %
                     QString::number((int)m_sb->setting<bool>(Constants::ProfileSettings::LaserScanner::kTransmitHeightMap)) % ") "
                     % commentLine(Constants::RegionTypeStrings::kLaserScan % " - TRANSMIT NECESSARY BUILD PARAMETERS");

            rv += "M13(#TEMPFILENAME#)" % commentLine(Constants::RegionTypeStrings::kLaserScan % " - TRANSMIT FILENAME");
        }

        if(!m_sb->setting< QString >(Constants::PrinterSettings::GCode::kStartCode).isEmpty())
            rv += m_sb->setting< QString >(Constants::PrinterSettings::GCode::kStartCode);

        rv += m_newline;

        rv += commentLine("LAYER COUNT: " % QString::number(num_layers));

        return rv;
    }

    QString GKNWriter::writeBeforeLayer(float new_min_z, QSharedPointer<SettingsBase> sb)
    {
        if (m_first_travel)
        {
            m_current_z += sb->setting< Distance >(Constants::ProfileSettings::Layer::kLayerHeight);
        }
        m_first_layer_travel = true;
        return {};
    }

    QString GKNWriter::writeBeforePart(QVector3D normal)
    {      
//        if(m_can_tilt)
//        {
//            QVector3D productVec = normal * m_z_axis;
//            float val = qAcos((productVec.x() + productVec.y() + productVec.z()) / (normal.length() * m_z_axis.length()));

//            if(normal.length() == 0)
//                m_E1 = 0.0;
//            else
//                m_E1 = M_PI_2 - qAcos((productVec.x() + productVec.y() + productVec.z()) / (normal.length() * m_z_axis.length()));
//        }
        return QString();
    }

    QString GKNWriter::writeBeforeIsland()
    {
        QString rv;
        return rv;
    }

    QString GKNWriter::writeBeforeScan(Point min, Point max, int layer, int boundingBox, Axis axis, Angle angle)
    {
        QString rv;

        rv += "M15(" %
            QString::number(Distance(min.x()).to(mm), 'f', 4) % m_variable_delim %
            QString::number(Distance(max.x()).to(mm), 'f', 4) % m_variable_delim %
            QString::number(Distance(min.y()).to(mm), 'f', 4) % m_variable_delim %
            QString::number(Distance(max.y()).to(mm), 'f', 4) % m_variable_delim %
            QString::number(layer) % m_variable_delim % QString::number(boundingBox) % m_variable_delim %
            QString::number((int)axis) % m_variable_delim % QString::number(round(angle.to(deg))) % ") "
                % commentLine(Constants::RegionTypeStrings::kLaserScan % " - TRANSMIT BOUND BOX AND LAYER");

        return rv;
    }

    QString GKNWriter::writeBeforeRegion(RegionType type, int pathSize)
    {
        return {};
    }

    QString GKNWriter::writeBeforePath(RegionType type)
    {
        QString rv;
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
        return rv;
    }

    QString GKNWriter::writeTravel(Point start_location, Point target_location, TravelLiftType lType,
                                   QSharedPointer<SettingsBase> params)
    {
            QString rv;

            if(m_can_tilt)
            {
                QVector<QVector3D> normals = params->setting<QVector<QVector3D>>(Constants::SegmentSettings::kTilt);
                if(normals.size() > 0)
                {
                    bool isCCW = params->setting<bool>(Constants::SegmentSettings::kCCW);

                    //m_z_axis magnitude is always 1, so we can drop it
                    if(isCCW == m_ccw)
                        m_E1 = -(M_PI_2 - qAcos(QVector3D::dotProduct(normals[0], m_z_axis) / normals[0].length()));
                    else
                        m_E1 = -(M_PI_2 - qAcos(QVector3D::dotProduct(normals[1], m_z_axis) / normals[1].length()));
                }
                else
                {
                    m_E1 = 0.0;
                }
            }
            RegionType rType = params->setting<RegionType>(Constants::SegmentSettings::kRegionType);
            Velocity speed = params->setting<Velocity>(Constants::SegmentSettings::kSpeed);

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

            // Travel_lift vector in direction normal to the layer
            // With length = lift height as defined in settings
            QVector3D travel_lift = getTravelLift();

            if(!m_suppress_next_travel_lift || rType == RegionType::kLaserScan)
            {
                // Write the lift
                if(travel_lift_required && !m_first_travel && (lType == TravelLiftType::kBoth || lType == TravelLiftType::kLiftUpOnly))
                {
                    if(m_first_layer_travel)
                    {
                        if(m_ccw)
                        {
                            m_E2 += M_PI_2;
                            m_actual_angle -= M_PI_2;
                        }
                        else
                        {
                            m_E2 -= M_PI_2;
                            m_actual_angle += M_PI_2;
                        }
                        m_table_orientation = MathUtils::CreateQuaternion(Angle(0.0), Angle(m_E1), Angle(m_E2), QuaternionOrder::kZYX);
                    }
                    Point lift_destination = start_location + travel_lift; //lift destination is above start location

                    if(m_override_travel_lift)
                    {
                        lift_destination = target_location + travel_lift;
                        m_override_travel_lift = false;
                    }
                    rv += m_G0 % writeCoordinates(lift_destination, speed, rType, true) % commentSpaceLine("TRAVEL LIFT Z"); //output WXYZ
                    setFeedrate(m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kZSpeed));

                    if(m_first_layer_travel)
                    {
                        if(m_ccw)
                        {
                            m_E2 -= M_PI_2;
                            m_actual_angle += M_PI_2;
                        }
                        else
                        {
                            m_E2 += M_PI_2;
                            m_actual_angle -= M_PI_2;
                        }
                        m_table_orientation = MathUtils::CreateQuaternion(Angle(0.0), Angle(m_E1), Angle(m_E2), QuaternionOrder::kZYX);
                        if(rType != RegionType::kLaserScan)
                            m_first_layer_travel = false;
                    }
                }
            }
            if(m_suppress_next_travel_lift)
                m_suppress_next_travel_lift = false;

            // Wire cutoff
            if(!m_wire_cut && rType != RegionType::kLaserScan)
            {
                rv += "M0(TRUE)" % commentSpaceLine("WIRE CUTOFF");
                m_wire_cut = true;
            }

            // Open Shutter
            if (rType == RegionType::kLaserScan)
            {
                rv += "M8(TRUE)" % commentSpaceLine("OPEN SHUTTER");
            }

            // Pyrometer
            if(m_sb->setting< int >(Constants::ProfileSettings::ThermalScanner::kPyrometerMove) && rType != RegionType::kLaserScan)
            {
                rv += "M22(" % QString::number(Distance(target_location.z()).to(m_meta.m_distance_unit), 'f', 4)
                        % ")" % commentSpaceLine("MOVE TO PYROMETER POSITION");
            }


            // Write the travel
            Point travel_destination = target_location;
            if(m_first_travel)
                travel_destination.z(qAbs(m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kZOffset)() +
                                          m_sb->setting< Distance >(Constants::ProfileSettings::Travel::kLiftHeight)));
            else if(travel_lift_required)
                travel_destination = travel_destination + travel_lift; //travel destination is above the target point

            rv += m_G0 % writeCoordinates(travel_destination, speed, rType) % commentSpaceLine("TRAVEL");
            setFeedrate(m_sb->setting< Velocity >(Constants::ProfileSettings::Travel::kSpeed));

            //write the travel lower (undo the lift)
            if(travel_lift_required && (lType == TravelLiftType::kBoth || lType == TravelLiftType::kLiftLowerOnly))
            {
                rv += m_G0 % writeCoordinates(target_location, speed, rType) % commentSpaceLine("TRAVEL LOWER Z");
                setFeedrate(m_sb->setting< Velocity >(Constants::PrinterSettings::MachineSpeed::kZSpeed));
            }

            if(m_first_travel) //if this is the first travel
                m_first_travel = false; //update for next one

            return rv;
        }

    QString GKNWriter::writeLine(const Point& start_point, const Point& target_point, const QSharedPointer<SettingsBase> params)
    {
        Velocity speed = params->setting<Velocity>(Constants::SegmentSettings::kSpeed);
        RegionType region_type = params->setting<RegionType>(Constants::SegmentSettings::kRegionType);
        PathModifiers path_modifiers = params->setting<PathModifiers>(Constants::SegmentSettings::kPathModifiers);

        if(m_can_rotate)
        {
            Angle segmentAngle = params->setting<Angle>(Constants::SegmentSettings::kRotation);
            segmentAngle = abs(segmentAngle);

            if(m_ccw)
            {
                m_E2 += segmentAngle;
                m_actual_angle -= segmentAngle;
            }
            else
            {
                m_E2 -= segmentAngle;
                m_actual_angle += segmentAngle;
            }
        }

        if(m_can_tilt)
        {
            QVector<QVector3D> normals = params->setting<QVector<QVector3D>>(Constants::SegmentSettings::kTilt);
            bool isCCW = params->setting<bool>(Constants::SegmentSettings::kCCW);

            //m_z_axis magnitude is always 1, so we can drop it
            if(isCCW == m_ccw)
                m_E1 = -(M_PI_2 - qAcos(QVector3D::dotProduct(normals[0], m_z_axis) / normals[0].length()));
            else
                m_E1 = -(M_PI_2 - qAcos(QVector3D::dotProduct(normals[1], m_z_axis) / normals[1].length()));

        }

        m_table_orientation = MathUtils::CreateQuaternion(Angle(0.0), Angle(m_E1), Angle(m_E2), QuaternionOrder::kZYX);
        QString rv;

        //turn on the extruder if it isn't already on
        if (m_extruders_on[0] == false)
            rv += writeExtruderOn(region_type);

        if ((path_modifiers == PathModifiers::kForwardTipWipe || path_modifiers == PathModifiers::kReverseTipWipe
                || path_modifiers == PathModifiers::kPerimeterTipWipe) && m_extruders_on[0] && !m_wipe_tip)
            rv += writeUpdateLaserAndWire();

        rv += m_G1 % writeCoordinates(target_point, speed, region_type);


        //printf("---\n%s\n", params->json().dump(4).c_str());

        if (params->contains(Constants::SegmentSettings::kESP)) {
            rv += " ESP" + QString::number(params->setting<float>(Constants::SegmentSettings::kESP), 'f', 2);
        }

        //add comment for gcode parser
        if (path_modifiers != PathModifiers::kNone)
            rv += commentSpaceLine(toString(region_type) % " " % toString(path_modifiers));
        else
            rv += commentSpaceLine(toString(region_type));

        return rv;
    }

    QString GKNWriter::writeScan(Point target_point, Velocity speed, bool on_off)
    {
        QString rv;
        if(on_off)
        {
            rv += m_G1 % writeCoordinates(target_point, speed, RegionType::kLaserScan) %
                    m_s % QString::number(m_sb->setting<Distance>(Constants::ProfileSettings::LaserScanner::kBufferDistance).to(m_meta.m_distance_unit), 'f', 4) %
                    m_v % QString::number(m_sb->setting<Distance>(Constants::ProfileSettings::LaserScanner::kLaserScannerStepDistance).to(m_meta.m_distance_unit), 'f', 4) %
                    commentSpaceLine(Constants::RegionTypeStrings::kLaserScan % " - START");
        }
        else
        {
            rv += m_G0 % writeCoordinates(target_point, speed, RegionType::kLaserScan) % commentLine(Constants::RegionTypeStrings::kLaserScan % " - MOVE");
        }
        return rv;
    }

    QString GKNWriter::writeAfterPath(RegionType type)
    {
        QString rv;
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

        m_ccw = !m_ccw;

        return rv;
    }

    QString GKNWriter::writeAfterRegion(RegionType type)
    {
        QString rv;
        return rv;
    }

    QString GKNWriter::writeAfterScan(Distance beadWidth, Distance laserStep, Distance laserResolution)
    {
        QString rv;

        int xDistance = round((beadWidth.to(mm) / laserStep.to(mm)));
        int yDistance = round((beadWidth.to(mm) / laserResolution.to(mm)));
        rv += "M17(" % QString::number(xDistance) % m_variable_delim % QString::number(yDistance) % ") " %
                commentLine(Constants::RegionTypeStrings::kLaserScan % " - PROCESS PATCH");
        rv += "M8(FALSE)" % commentSpaceLine("CLOSE SHUTTER");

        m_override_travel_lift = true;

        return rv;
    }

    QString GKNWriter::writeAfterAllScans()
    {
        QString rv;
        if(m_sb->setting<bool>(Constants::ProfileSettings::LaserScanner::kLaserScanner))
            rv += "M16 " % commentLine(Constants::RegionTypeStrings::kLaserScan % " - PROCESS LAYER");
        return rv;
    }

    QString GKNWriter::writeAfterIsland()
    {
        QString rv;
        return rv;
    }

    QString GKNWriter::writeAfterPart()
    {
        QString rv;
        return rv;
    }

    QString GKNWriter::writeAfterLayer()
    {
        QString rv;
        rv += m_sb->setting< QString >(Constants::PrinterSettings::GCode::kLayerCodeChange) % m_newline;
        return rv;
    }

    QString GKNWriter::writeShutdown()
    {
        QString rv;
        rv += "M84";
        return rv;
    }

    QString GKNWriter::writeDwell(Time time)
    {
        if (time > 0)
            return "G4(" % QString::number(time.to(m_meta.m_time_unit), 'f', 4) % ")" % commentSpaceLine("DWELL");
        else
            return {};
    }

    QString GKNWriter::writeExtruderOn(RegionType type)
    {
        QString rv;
        m_extruders_on[0] = true;
        m_wire_cut = false;
        m_wipe_tip = false;
        rv += "M21(TRUE)" % commentSpaceLine("GIVE CONTROL OF LASER TO EMAQS");
        rv += "M5(" % QString::number(m_sb->setting< Voltage >(Constants::PrinterSettings::Auxiliary::kGKNLaserPower).to(V)) % ","
                % QString::number(m_sb->setting< Voltage >(Constants::PrinterSettings::Auxiliary::kGKNLaserPower).to(V)) % ")" % commentSpaceLine("FIRE LASERS");
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
        rv += "M12(TRUE)" % commentSpaceLine("TURN HOTWIRE ON");
        rv += "M6(" % QString::number(m_sb->setting< Voltage >(Constants::PrinterSettings::Auxiliary::kGKNWireSpeed).to(V))
                % ")" % commentSpaceLine("TURN WIRE FEEDER ON");
        return rv;
    }

    QString GKNWriter::writeExtruderOff()
    {
        QString rv;
        m_extruders_on[0] = false;
        if(m_sb->setting< Time >(Constants::MaterialSettings::Extruder::kOffDelay) > 0)
        {
            rv += writeDwell(m_sb->setting< Time >(Constants::MaterialSettings::Extruder::kOffDelay));
        }
        rv += "M7" % commentSpaceLine("TURN OFF LASER AND WIRE FEED");
        m_suppress_next_travel_lift = true;
        return rv;
    }

    QString GKNWriter::writeUpdateLaserAndWire()
    {
        m_wipe_tip = true;
        QString rv;
        rv += "M19(" % QString::number(m_sb->setting< Voltage >(Constants::PrinterSettings::Auxiliary::kGKNLaserPower).to(V)) %
                " * " % QString::number(m_sb->setting< float >(Constants::MaterialSettings::TipWipe::kLaserPowerMultiplier), 'f', 4) %
                "," % QString::number(m_sb->setting< Voltage >(Constants::PrinterSettings::Auxiliary::kGKNLaserPower).to(V)) %
                " * " % QString::number(m_sb->setting< float >(Constants::MaterialSettings::TipWipe::kLaserPowerMultiplier), 'f', 4) %
                ")" % commentSpaceLine("UPDATE LASER FOR TIP WIPE");
        rv += "M20(" % QString::number(m_sb->setting< Voltage >(Constants::PrinterSettings::Auxiliary::kGKNWireSpeed).to(V))
                % " * " % QString::number(m_sb->setting< float >(Constants::MaterialSettings::TipWipe::kWireFeedMultiplier), 'f', 4) %
                ")" % commentSpaceLine("UPDATE WIRE FEED FOR TIP WIPE");
        return rv;
    }

    QString GKNWriter::writeCoordinates(Point destination, Velocity speed, RegionType type, bool isTravel)
    {
        QString rv;

        // X and Y
        rv += "(" % QString::number(Distance(destination.x()).to(m_meta.m_distance_unit), 'f', 4) %
              m_variable_delim % QString::number(Distance(destination.y()).to(m_meta.m_distance_unit), 'f', 4);

        // Z
        Distance z_offset = m_sb->setting< Distance >(Constants::PrinterSettings::Dimensions::kZOffset);
        QString target_z = QString::number(Distance(destination.z() + z_offset).to(m_meta.m_distance_unit), 'f', 4);
        rv += m_variable_delim % target_z;
        m_current_z = destination.z() + z_offset;

        // A, B, C
        if(type == RegionType::kLaserScan && m_sb->setting<int>(Constants::ProfileSettings::LaserScanner::kEnableOrientationDefinition) == 1 && !isTravel)
        {
            rv += m_variable_delim % QString::number(m_sb->setting<Angle>(Constants::ProfileSettings::LaserScanner::kOrientationA).to(m_meta.m_angle_unit), 'f', 4);
            rv += m_variable_delim % QString::number(m_sb->setting<Angle>(Constants::ProfileSettings::LaserScanner::kOrientationB).to(m_meta.m_angle_unit), 'f', 4);
            rv += m_variable_delim % QString::number(m_sb->setting<Angle>(Constants::ProfileSettings::LaserScanner::kOrientationC).to(m_meta.m_angle_unit), 'f', 4);
        }
        else
        {
            QQuaternion q = MathUtils::QuatMult(m_table_orientation, m_arm_orientation);
            double pitch, roll, yaw;
            MathUtils::EulerAngles(q, &pitch, &roll, &yaw);
//            if(m_can_rotate)
//            {
////                if(qFuzzyCompare(qAbs(yaw), M_PI))
////                {
////                    bool c = std::signbit(yaw) == std::signbit(m_E2());
////                    if(!c)
////                        yaw *= -1;
////                }
//                rv += m_variable_delim % QString::number(Angle(-yaw).to(m_meta.m_angle_unit), 'f', 4);
//            }
//            else
                rv += m_variable_delim % QString::number(Angle(yaw).to(m_meta.m_angle_unit), 'f', 4);

            rv += m_variable_delim % QString::number(Angle(roll).to(m_meta.m_angle_unit), 'f', 4);
            rv += m_variable_delim % QString::number(Angle(pitch).to(m_meta.m_angle_unit), 'f', 4);
        }

        // Tool Coordinate
        rv += m_variable_delim % QString::number(m_sb->setting< int >(Constants::PrinterSettings::MachineSetup::kToolCoordinate));

        // Base Coordinate
        rv += m_variable_delim % QString::number(m_sb->setting< int >(Constants::PrinterSettings::MachineSetup::kBaseCoordinate));

        // Velocity
        rv += m_variable_delim % QString::number(speed.to(m_meta.m_velocity_unit), 'f', 4);

        if(m_can_tilt || m_can_rotate)
        {
            if(type == RegionType::kLaserScan && !isTravel)
                rv += m_variable_delim % m_zero % m_variable_delim % m_zero;
            else
                rv += m_variable_delim % QString::number(m_E1.to(m_meta.m_angle_unit), 'f', 4) % m_variable_delim % QString::number(m_actual_angle.to(m_meta.m_angle_unit), 'f', 4);
        }

        rv += ")";

        return rv;
    }

}  // namespace ORNL
