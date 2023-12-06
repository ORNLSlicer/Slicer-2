// Local
#include "step/layer/regions/perimeter.h"
#include "geometry/segments/line.h"
#include "managers/preferences_manager.h"
#include "optimizers/path_order_optimizer.h"
#include "geometry/path_modifier.h"
#include "utilities/mathutils.h"
#include "algorithms/knn.h"
#include "geometry/curve_fitting.h"

#ifdef HAVE_SINGLE_PATH
#include "single_path/single_path.h"
Q_DECLARE_METATYPE(QList<SinglePath::Bridge>);
#endif

#ifdef HAVE_WIRE_FEED
#include "wire_feed/wire_feed.h"
#endif

namespace ORNL {
    Perimeter::Perimeter(const QSharedPointer<SettingsBase>& sb, const int index, const QVector<SettingsPolygon>& settings_polygons, const SingleExternalGridInfo& gridInfo,
                         PolygonList uncut_geometry)
        : RegionBase(sb, index, settings_polygons, gridInfo, uncut_geometry) {
    }

    QString Perimeter::writeGCode(QSharedPointer<WriterBase> writer) {
        QString gcode;
        gcode += writer->writeBeforeRegion(RegionType::kPerimeter);
        for (Path path : m_paths) {
            gcode += writer->writeBeforePath(RegionType::kPerimeter);
            for (QSharedPointer<SegmentBase> segment : path.getSegments()) {
                gcode += segment->writeGCode(writer);
            }
            gcode += writer->writeAfterPath(RegionType::kPerimeter);
        }
        gcode += writer->writeAfterRegion(RegionType::kPerimeter);
        return gcode;
    }

    void Perimeter::compute(uint layer_num, QSharedPointer<SyncManager>& sync)
    {
        m_paths.clear();
        m_outer_most_path_set.clear();
        m_inner_most_path_set.clear();

        setMaterialNumber(m_sb->setting<int>(Constants::MaterialSettings::MultiMaterial::kPerimterNum));
        Distance beadWidth = m_sb->setting<Distance>(Constants::ProfileSettings::Perimeter::kBeadWidth);
        PolygonList lastPerimeter;

        int rings = m_sb->setting<int>(Constants::ProfileSettings::Perimeter::kCount);
        if (m_sb->setting<bool>(Constants::ExperimentalSettings::DirectedPerimeter::kEnableDirectedPerimeter))
        {
            computeDirected(beadWidth, rings);
        }
        else
        {
            if(m_sb->setting<bool>(Constants::ExperimentalSettings::WireFeed::kWireFeedEnable) && m_uncut_geometry != PolygonList() && rings == 3)
            {
                    #ifdef HAVE_WIRE_FEED
                        QVector<QVector<QPair<double, double>>> result = WireFeed::WireFeed::computePerimetersForBase(m_geometry.getRawPoints(),
                                                                                                                      m_uncut_geometry.getRawPoints(),
                                                                                                                      beadWidth(), rings);
                        m_computed_geometry.append(PolygonList(result));
                    #endif
            }
            else
            {
                PolygonList path_line = m_geometry.offset(-beadWidth / 2);

                QVector<Point> previousPoints;
                QVector<Point> currentPoints;
                for(Polygon& poly : m_geometry)
                {
                    for(Point& p : poly)
                    {
                        previousPoints.push_back(p);
                    }
                }

                int ring_nr = 0;
                int path_line_num = 0;
                while (!path_line.isEmpty() && ring_nr < rings)
                {
                    for(Polygon& poly : path_line)
                    {
                        for(Point& p : poly)
                        {
                            kNN neighbor(previousPoints, QVector<Point> { p }, 1);
                            neighbor.execute();

                            int closest = neighbor.getNearestIndices().first();
                            p.setNormals(previousPoints[closest].getNormals());
                            currentPoints.push_back(p);
                        }
                    }
                    previousPoints = currentPoints;
                    currentPoints.clear();

                    if((path_line_num + 1) == rings)
                        lastPerimeter = path_line;

                    m_computed_geometry.append(path_line);

                    path_line = path_line.offset(-beadWidth, -beadWidth / 2);
                    ring_nr++;

                    path_line_num++;
                }
            }
        }

        #ifdef HAVE_SINGLE_PATH
            if(!m_sb->setting<bool>(Constants::ExperimentalSettings::SinglePath::kEnableSinglePath))
                this->createPaths();
        #else
            m_layer_num = layer_num;

            this->createPaths();

            // Append the innermost perimeter to allow for correct computation in the infill region when shifted beads is enabled
            if (!m_computed_geometry.isEmpty()){
                m_computed_geometry.append(lastPerimeter);
                m_geometry = m_computed_geometry.last().offset(-beadWidth / 2, -beadWidth / 2);
            }

        #endif

        if(m_sb->setting<bool>(Constants::ExperimentalSettings::CurveFitting::kEnableArcFitting) ||
           m_sb->setting<bool>(Constants::ExperimentalSettings::CurveFitting::kEnableSplineFitting))
            for(auto& path : this->m_paths)
                CurveFitting::Fit(path, m_sb);

        if(static_cast<PrintDirection>(m_sb->setting<int>(Constants::ProfileSettings::Ordering::kPerimeterReverseDirection)) != PrintDirection::kReverse_off)
            for(Path& path : m_paths)
                path.reverseSegments();
    }

    void Perimeter::computeDirected(Distance bead_width, int rings)
    {
        //! Collect interior/exterior geometries
        //! \note Identification relies on poly orientation
        QVector<PolygonList> border_geometry(2);
        enum geometry_locality : uint8_t {interior = 0, exterior = 1};
        for (Polygon &poly : m_geometry)
        {
            if (!poly.orientation())
                border_geometry[interior] += poly;
            else
                border_geometry[exterior] += poly;
        }


        //! Compute directed perimeters
        Direction gen_dir = static_cast<Direction>(m_sb->setting<int>(Constants::ExperimentalSettings::DirectedPerimeter::kGenerationDirection));
        if (gen_dir == Direction::kOutward && !border_geometry[interior].isEmpty()) //! Compute Outward
        {
            //! Offset interior geometry outward
            PolygonList path_line = border_geometry[interior].offset(bead_width / 2);
            path_line.reverseNormalDirections();

            //! Continue offset
            int ring_nr = 0;
            while ((path_line - border_geometry[exterior]).isEmpty() && ring_nr < rings)
            {
                m_computed_geometry.append(path_line);
                path_line = path_line.offset(bead_width, bead_width / 2);
                ring_nr++;
            }
        }
        else //! Compute Inward
        {
            //! Offset exterior geometry inward
            PolygonList path_line = border_geometry[exterior].offset(-bead_width / 2);

            //! Continue offset
            int ring_nr = 0;
            while ((border_geometry[interior] - path_line).isEmpty() && ring_nr < rings)
            {
                m_computed_geometry.append(path_line);
                path_line = path_line.offset(-bead_width, -bead_width / 2);
                ring_nr++;
            }
        }


        //! Discard perimeters which bulge outside border geometry
        if (m_sb->setting<bool>(Constants::ExperimentalSettings::DirectedPerimeter::kEnableDiscardBulgingPerimeter))
        {
            if (!m_computed_geometry.isEmpty())
            {
                if (gen_dir == Direction::kOutward)
                {
                    if (!(m_computed_geometry.last().offset(bead_width / 2) - border_geometry[exterior]).isEmpty())
                        m_computed_geometry.removeLast();

                    if (!m_computed_geometry.isEmpty())
                        m_geometry = border_geometry[exterior] - m_computed_geometry.last().offset(bead_width / 2);
                }
                else //! gen_dir == Inward
                {
                    if (!(border_geometry[interior] - m_computed_geometry.last().offset(-bead_width / 2)).isEmpty())
                        m_computed_geometry.removeLast();

                    if (!m_computed_geometry.isEmpty())
                        m_geometry = m_computed_geometry.last().offset(-bead_width / 2) - border_geometry[interior];
                }
            }
        }
    }

    void Perimeter::optimize(QSharedPointer<PathOrderOptimizer> poo, Point& current_location, QVector<Path>& innerMostClosedContour, QVector<Path>& outerMostClosedContour, bool& shouldNextPathBeCCW)
    {
        if(m_sb->setting<bool>(Constants::ProfileSettings::SpecialModes::kEnableSpiralize))
        {
            if(m_paths.size() > 0)
            {
                Path path = m_paths.first();
                m_paths.clear();
                poo->setPathsToEvaluate(QVector<Path> { path });
                m_paths.append(poo->linkSpiralPath2D(m_was_last_region_spiral));
            }
        }
        else
        {
            poo->setPathsToEvaluate(m_paths);
            poo->setParameters(shouldNextPathBeCCW);
            m_paths.clear();
            while(poo->getCurrentPathCount() > 0)
            {
                Path nextPath = poo->linkNextPath();

                QVector<Path> tmp_path;
                calculateModifiers(nextPath, m_sb->setting<bool>(Constants::PrinterSettings::MachineSetup::kSupportG3), tmp_path, poo->getCurrentLocation());
                m_paths.push_back(nextPath);
            }
            shouldNextPathBeCCW = poo->getCurrentCCW();
        }
    }

    void Perimeter::calculateModifiers(Path& path, bool supportsG3, QVector<Path>& innerMostClosedContour, Point& current_location)
    {
        if(m_sb->setting<bool>(Constants::ExperimentalSettings::Ramping::kTrajectoryAngleEnabled))
        {
            PathModifierGenerator::GenerateTrajectorySlowdown(path, m_sb);
        }

        if(m_sb->setting<bool>(Constants::MaterialSettings::Slowdown::kPerimeterEnable))
        {
            PathModifierGenerator::GenerateSlowdown(path, m_sb->setting<Distance>(Constants::MaterialSettings::Slowdown::kPerimeterDistance),
                                                    m_sb->setting<Distance>(Constants::MaterialSettings::Slowdown::kPerimeterLiftDistance),
                                                    m_sb->setting<Distance>(Constants::MaterialSettings::Slowdown::kPerimeterCutoffDistance),
                                                    m_sb->setting<Velocity>(Constants::MaterialSettings::Slowdown::kPerimeterSpeed),
                                                    m_sb->setting<AngularVelocity>(Constants::MaterialSettings::Slowdown::kPerimeterExtruderSpeed),
                                                    m_sb->setting<bool>(Constants::ProfileSettings::SpecialModes::kEnableWidthHeight),
                                                    m_sb->setting<double>(Constants::MaterialSettings::Slowdown::kSlowDownAreaModifier));
        }
        if(m_sb->setting<bool>(Constants::MaterialSettings::TipWipe::kPerimeterEnable))
        {
            if(static_cast<TipWipeDirection>(m_sb->setting<int>(Constants::MaterialSettings::TipWipe::kPerimeterDirection)) == TipWipeDirection::kForward ||
                    static_cast<TipWipeDirection>(m_sb->setting<int>(Constants::MaterialSettings::TipWipe::kPerimeterDirection)) == TipWipeDirection::kOptimal)
                    PathModifierGenerator::GenerateTipWipe(path, PathModifiers::kForwardTipWipe, m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kPerimeterDistance),
                                                       m_sb->setting<Velocity>(Constants::MaterialSettings::TipWipe::kPerimeterSpeed),
                                                       m_sb->setting<Angle>(Constants::MaterialSettings::TipWipe::kPerimeterAngle),
                                                       m_sb->setting<AngularVelocity>(Constants::MaterialSettings::TipWipe::kPerimeterExtruderSpeed),
                                                       m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kPerimeterLiftHeight),
                                                       m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kPerimeterCutoffDistance));
            else if(static_cast<TipWipeDirection>(m_sb->setting<int>(Constants::MaterialSettings::TipWipe::kPerimeterDirection)) == TipWipeDirection::kAngled)
            {
                PathModifierGenerator::GenerateTipWipe(path, PathModifiers::kAngledTipWipe, m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kPerimeterDistance),
                                                       m_sb->setting<Velocity>(Constants::MaterialSettings::TipWipe::kPerimeterSpeed),
                                                       m_sb->setting<Angle>(Constants::MaterialSettings::TipWipe::kPerimeterAngle),
                                                       m_sb->setting<AngularVelocity>(Constants::MaterialSettings::TipWipe::kPerimeterExtruderSpeed),
                                                       m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kPerimeterLiftHeight),
                                                       m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kPerimeterCutoffDistance));
            }
            else
                PathModifierGenerator::GenerateTipWipe(path, PathModifiers::kReverseTipWipe, m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kPerimeterDistance),
                                                       m_sb->setting<Velocity>(Constants::MaterialSettings::TipWipe::kPerimeterSpeed),
                                                       m_sb->setting<Angle>(Constants::MaterialSettings::TipWipe::kPerimeterAngle),
                                                       m_sb->setting<AngularVelocity>(Constants::MaterialSettings::TipWipe::kPerimeterExtruderSpeed),
                                                       m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kPerimeterLiftHeight),
                                                       m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kPerimeterCutoffDistance));

            current_location = path.back()->end();
        }
        if(m_sb->setting<bool>(Constants::MaterialSettings::SpiralLift::kPerimeterEnable))
        {
            PathModifierGenerator::GenerateSpiralLift(path, m_sb->setting<Distance>(Constants::MaterialSettings::SpiralLift::kLiftRadius),
                                                      m_sb->setting<Distance>(Constants::MaterialSettings::SpiralLift::kLiftHeight),
                                                      m_sb->setting<int>(Constants::MaterialSettings::SpiralLift::kLiftPoints),
                                                      m_sb->setting<Velocity>(Constants::MaterialSettings::SpiralLift::kLiftSpeed), supportsG3);

            current_location = path.back()->end();
        }
        if(m_sb->setting<bool>(Constants::MaterialSettings::Startup::kPerimeterEnable))
        {
            if(m_sb->setting<bool>(Constants::MaterialSettings::Startup::kPerimeterRampUpEnable))
            {
                PathModifierGenerator::GenerateInitialStartupWithRampUp(path, m_sb->setting<Distance>(Constants::MaterialSettings::Startup::kPerimeterDistance),
                                                              m_sb->setting<Velocity>(Constants::MaterialSettings::Startup::kPerimeterSpeed),
                                                              m_sb->setting<Velocity>(Constants::ProfileSettings::Perimeter::kSpeed),
                                                              m_sb->setting<AngularVelocity>(Constants::MaterialSettings::Startup::kPerimeterExtruderSpeed),
                                                              m_sb->setting<AngularVelocity>(Constants::ProfileSettings::Perimeter::kExtruderSpeed),
                                                              m_sb->setting<int>(Constants::MaterialSettings::Startup::kPerimeterSteps),
                                                              m_sb->setting<bool>(Constants::ProfileSettings::SpecialModes::kEnableWidthHeight),
                                                              m_sb->setting<double>(Constants::MaterialSettings::Startup::kStartUpAreaModifier));
            }
            else
            {
                PathModifierGenerator::GenerateInitialStartup(path, m_sb->setting<Distance>(Constants::MaterialSettings::Startup::kPerimeterDistance),
                                                              m_sb->setting<Velocity>(Constants::MaterialSettings::Startup::kPerimeterSpeed),
                                                              m_sb->setting<AngularVelocity>(Constants::MaterialSettings::Startup::kPerimeterExtruderSpeed),
                                                              m_sb->setting<bool>(Constants::ProfileSettings::SpecialModes::kEnableWidthHeight),
                                                              m_sb->setting<double>(Constants::MaterialSettings::Startup::kStartUpAreaModifier));
            }
        }
    }

    void Perimeter::createPaths()
    {
        const Point origin(m_sb->setting<double>(Constants::PrinterSettings::Dimensions::kXOffset), m_sb->setting<double>(Constants::PrinterSettings::Dimensions::kYOffset));
        for(int i = 0, end = m_computed_geometry.size(); i < end; ++i)
        {
            PolygonList& polygon_list = m_computed_geometry[i];
            for (Polygon polygon: polygon_list)
            {
                int rep = 0;

                Path new_path;

                new_path.setCCW(polygon.orientation());
                new_path.setContainsOrigin(polygon.inside(origin));

                Distance default_width                  = m_sb->setting< Distance >(Constants::ProfileSettings::Perimeter::kBeadWidth);
                Distance default_height                 = m_sb->setting< Distance >(Constants::ProfileSettings::Layer::kLayerHeight);
                Velocity default_speed                  = m_sb->setting< Velocity >(Constants::ProfileSettings::Perimeter::kSpeed);
                Acceleration default_acceleration       = m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kPerimeter);
                AngularVelocity default_extruder_speed  = m_sb->setting< AngularVelocity >(Constants::ProfileSettings::Perimeter::kExtruderSpeed);
                float default_esp_value                 = m_sb->setting< float >(Constants::PrinterSettings::Embossing::kESPNominalValue);
                int material_number                     = m_sb->setting< int >(Constants::MaterialSettings::MultiMaterial::kPerimterNum);

                bool embossing_enable = m_sb->setting<bool>(Constants::PrinterSettings::Embossing::kEnableEmbossing);

                for (int j = 0, end_cond = polygon.size(); j <  end_cond; ++j)
                {
                    Point start = polygon[j];
                    Point end = polygon[(j + 1) % end_cond];

                    bool is_settings_region = false;

                    QVector<Point> intersections;
                    for(auto settings_poly : m_settings_polygons)
                    {
                        QSharedPointer<SettingsBase> updatedBase = QSharedPointer<SettingsBase>::create(*m_sb);
                        updatedBase->populate(settings_poly.getSettings());
                        bool contains_start = settings_poly.inside(start);
                        bool contains_end   = settings_poly.inside(end);

                        if (contains_start) start.setSettings(updatedBase);
                        else start.setSettings(m_sb);

                        if (contains_end) end.setSettings(updatedBase);
                        else end.setSettings(m_sb);

                        if (contains_start) is_settings_region = true;
                        else if (contains_end) is_settings_region = false;

                        // Find if/ where this line intersects with a settings polygon
                        QVector<Point> poly_intersect = settings_poly.clipLine(start, end);

                        for (Point& point : poly_intersect) {
                            point.setSettings(updatedBase);
                        }

                        //qDebug() << poly_intersect;
                        intersections.append(poly_intersect);
                    }

                    // Divide lines into subsections
                    if(intersections.size() > 0)
                    {
                        // Sort points in order to start
                        std::sort(intersections.begin(), intersections.end(), [start](auto lhs, auto rhs){
                            return start.distance(lhs) < start.distance(rhs);
                        });

                        for(Point& point : intersections)
                        {
                            // If no settings change, skip this point
                            if(point.getSettings()->json() == m_sb->json())
                                continue;

                            //qDebug() << "\t\tCreating segment from" << start.toQVector3D() << "to" << point.toQVector3D() << "Distance:" << start.distance(point)();

                            QSharedPointer<LineSegment> segment = QSharedPointer<LineSegment>::create(start, point);

                            segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            is_settings_region ? start.getSettings()->setting< Distance >(Constants::ProfileSettings::Perimeter::kBeadWidth) : default_width);
                            segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,       is_settings_region ? start.getSettings()->setting< Distance >(Constants::ProfileSettings::Layer::kLayerHeight) : default_height);
                            segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,        is_settings_region ? start.getSettings()->setting< Velocity >(Constants::ProfileSettings::Perimeter::kSpeed) : default_speed);
                            segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            is_settings_region ? start.getSettings()->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kPerimeter) : default_acceleration);
                            segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    is_settings_region ? start.getSettings()->setting< AngularVelocity >(Constants::ProfileSettings::Perimeter::kExtruderSpeed) : default_extruder_speed);
                            segment->getSb()->setSetting(Constants::SegmentSettings::kMaterialNumber,   material_number);
                            segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       RegionType::kPerimeter);

                            if (embossing_enable) {
                                segment->getSb()->setSetting(Constants::SegmentSettings::kESP, is_settings_region ? start.getSettings()->setting< float >(Constants::PrinterSettings::Embossing::kESPEmbossingValue) : default_esp_value);
                                if (is_settings_region) segment->getSb()->setSetting(Constants::SegmentSettings::kPathModifiers, PathModifiers::kEmbossing);
                            }

                            new_path.append(segment);
                            is_settings_region = !is_settings_region;
                            start = point;
                        }
                    }

                    // Add final segment
                    QSharedPointer<LineSegment> segment = QSharedPointer<LineSegment>::create(start, end);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            is_settings_region ? start.getSettings()->setting< Distance >(Constants::ProfileSettings::Perimeter::kBeadWidth) : default_width);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,       is_settings_region ? start.getSettings()->setting< Distance >(Constants::ProfileSettings::Layer::kLayerHeight) : default_height);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,        is_settings_region ? start.getSettings()->setting< Velocity >(Constants::ProfileSettings::Perimeter::kSpeed) : default_speed);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            is_settings_region ? start.getSettings()->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kPerimeter) : default_acceleration);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    is_settings_region ? start.getSettings()->setting< AngularVelocity >(Constants::ProfileSettings::Perimeter::kExtruderSpeed) : default_extruder_speed);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kMaterialNumber,   material_number);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       RegionType::kPerimeter);

                    if (embossing_enable) {
                        segment->getSb()->setSetting(Constants::SegmentSettings::kESP, is_settings_region ? start.getSettings()->setting< float >(Constants::PrinterSettings::Embossing::kESPEmbossingValue) : default_esp_value);
                        if (is_settings_region) segment->getSb()->setSetting(Constants::SegmentSettings::kPathModifiers, PathModifiers::kEmbossing);
                    }

                    new_path.append(segment);
                }

                if (new_path.calculateLength() > m_sb->setting< Distance >(Constants::ProfileSettings::Perimeter::kMinPathLength))
                {
                    m_paths.append(new_path);
                    if (i == 0)
                        m_outer_most_path_set.append(new_path);
                    if(i == end - 1)
                        m_inner_most_path_set.append(new_path);
                }
            }
        }
    }

    #ifdef HAVE_SINGLE_PATH
    void Perimeter::setSinglePathGeometry(QVector<SinglePath::PolygonList> sp_geometry)
    {
        m_single_path_geometry = sp_geometry;
    }
    #endif

    #ifdef HAVE_SINGLE_PATH
    void Perimeter::createSinglePaths()
    {
        Distance perim_width                  = m_sb->setting< Distance >(Constants::ProfileSettings::Perimeter::kBeadWidth);
        Distance perim_height                 = m_sb->setting< Distance >(Constants::ProfileSettings::Layer::kLayerHeight);
        Velocity perim_speed                  = m_sb->setting< Velocity >(Constants::ProfileSettings::Perimeter::kSpeed);
        Acceleration perim_acceleration       = m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kPerimeter);
        AngularVelocity perim_extruder_speed  = m_sb->setting< AngularVelocity >(Constants::ProfileSettings::Perimeter::kExtruderSpeed);
        //int material_number                   = m_sb->setting< int >(Constants::MaterialSettings::MultiMaterial::kPerimterNum);

        Distance inset_width                  = m_sb->setting< Distance >(Constants::ProfileSettings::Inset::kBeadWidth);
        Distance inset_height                 = m_sb->setting< Distance >(Constants::ProfileSettings::Layer::kLayerHeight);
        Velocity inset_speed                  = m_sb->setting< Velocity >(Constants::ProfileSettings::Inset::kSpeed);
        Acceleration inset_acceleration       = m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kInset);
        AngularVelocity inset_extruder_speed  = m_sb->setting< AngularVelocity >(Constants::ProfileSettings::Inset::kExtruderSpeed);

        for (SinglePath::PolygonList polygonList : m_single_path_geometry)
        {
            for (SinglePath::Polygon polygon: polygonList)
            {
                Path new_path;
                for (int i = 0; i < polygon.size() - 1; i++) {
                    SinglePath::Point start = polygon[i];
                    SinglePath::Point end = polygon[i + 1];

                    QSharedPointer<LineSegment> segment = QSharedPointer<LineSegment>::create(start, end);

                    if(start.getRegionType() != end.getRegionType()) // This bridge jumps regions
                    {
                        segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            perim_width);
                        segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           perim_height);
                        segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            perim_speed);
                        segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            perim_acceleration);
                        segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    perim_extruder_speed);
                        //segment->setMaterialNumber(material_number);
                        segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType, RegionType::kPerimeter);
                    }else
                    {
                        segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            (start.getRegionType() == SinglePath::RegionType::kPerimeter) ? perim_width : inset_width);
                        segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           (start.getRegionType() == SinglePath::RegionType::kPerimeter) ? perim_height : inset_height);
                        segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            (start.getRegionType() == SinglePath::RegionType::kPerimeter) ? perim_speed : inset_speed);
                        segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            (start.getRegionType() == SinglePath::RegionType::kPerimeter) ? perim_acceleration : inset_acceleration);
                        segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    (start.getRegionType() == SinglePath::RegionType::kPerimeter) ? perim_extruder_speed : inset_extruder_speed);
                        //segment->setMaterialNumber(material_number);
                        segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       (start.getRegionType() == SinglePath::RegionType::kPerimeter) ? RegionType::kPerimeter : RegionType::kInset);
                    }
                    new_path.append(segment);
                }

                //! \note Close Polygon
                QSharedPointer<LineSegment> segment = QSharedPointer<LineSegment>::create(polygon.last(), polygon.first());
                segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            (polygon.last().getRegionType() == SinglePath::RegionType::kPerimeter) ? perim_width : inset_width);
                segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           (polygon.last().getRegionType() == SinglePath::RegionType::kPerimeter) ? perim_height : inset_height);
                segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            (polygon.last().getRegionType() == SinglePath::RegionType::kPerimeter) ? perim_speed : inset_speed);
                segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            (polygon.last().getRegionType() == SinglePath::RegionType::kPerimeter) ? perim_acceleration : inset_acceleration);
                segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    (polygon.last().getRegionType() == SinglePath::RegionType::kPerimeter) ? perim_extruder_speed : inset_extruder_speed);
                //segment->setMaterialNumber(material_number);
                segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       (polygon.last().getRegionType() == SinglePath::RegionType::kPerimeter) ? RegionType::kPerimeter : RegionType::kInset);

                new_path.append(segment);
                if (new_path.calculateLength() > m_sb->setting< Distance >(Constants::ProfileSettings::Perimeter::kMinPathLength))
                    m_paths.append(new_path);

            }
        }
    }
    #endif

    void Perimeter::setLayerCount(uint layer_count)
    {
        m_layer_count = layer_count;
        m_layer_count--;
    }

    QVector<Path>& Perimeter::getOuterMostPathSet()
    {
        return m_outer_most_path_set;
    }

    QVector<Path>& Perimeter::getInnerMostPathSet()
    {
        return m_inner_most_path_set;
    }

    QVector<PolygonList> Perimeter::getComputedGeometry()
    {
        return m_computed_geometry;
    }
}
