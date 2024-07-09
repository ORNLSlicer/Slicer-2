// Main Module
#include "step/layer/regions/inset.h"

#include <geometry/segments/line.h>
#include <optimizers/polyline_order_optimizer.h>
#include "geometry/path_modifier.h"
#include "utilities/mathutils.h"
#include "geometry/curve_fitting.h"

#ifdef HAVE_SINGLE_PATH
#include "single_path/single_path.h"
Q_DECLARE_METATYPE(QList<SinglePath::Bridge>);
#endif

namespace ORNL {
    Inset::Inset(const QSharedPointer<SettingsBase>& sb, const int index, const QVector<SettingsPolygon>& settings_polygons, const SingleExternalGridInfo& gridInfo)
        : RegionBase(sb, index, settings_polygons, gridInfo)
    {
        // NOP
    }

    QString Inset::writeGCode(QSharedPointer<WriterBase> writer) {
        QString gcode;
        gcode += writer->writeBeforeRegion(RegionType::kInset);
        for (Path path : m_paths) {
            gcode += writer->writeBeforePath(RegionType::kInset);
            for (QSharedPointer<SegmentBase> segment : path.getSegments()) {
                gcode += segment->writeGCode(writer);
            }
            gcode += writer->writeAfterPath(RegionType::kInset);
        }
        gcode += writer->writeAfterRegion(RegionType::kInset);
        return gcode;
    }

    void Inset::compute(uint layer_num, QSharedPointer<SyncManager>& sync) {
        m_paths.clear();
        m_outer_most_path_set.clear();
        m_inner_most_path_set.clear();

        setMaterialNumber(m_sb->setting<int>(Constants::MaterialSettings::MultiMaterial::kInsetNum));

        Distance beadWidth = m_sb->setting<Distance>(Constants::ProfileSettings::Inset::kBeadWidth);
        int rings = m_sb->setting<int>(Constants::ProfileSettings::Inset::kCount);

        PolygonList path_line = m_geometry.offset(-beadWidth / 2);

        Distance overlap = m_sb->setting<Distance>(Constants::ProfileSettings::Inset::kOverlap);
        if(overlap > 0)
            path_line = path_line.offset(overlap);

        int ring_nr = 0;
        while (!path_line.isEmpty() && ring_nr < rings)
        {
            for(Polygon poly : path_line)
            {
                Polyline line = poly.toPolyline();
                line.pop_back();
                m_computed_geometry.push_back(line);
            }

            ring_nr++;

            m_geometry = path_line.offset(-beadWidth / 2, -beadWidth / 2);
            path_line = path_line.offset(-beadWidth, -beadWidth / 2);
        }

        //Have to test with refactor
//        #ifdef HAVE_SINGLE_PATH
//            if(!m_sb->setting<bool>(Constants::ExperimentalSettings::SinglePath::kEnableSinglePath))
//                this->createPaths();
//        #else
//            this->createPaths();
//        #endif
    }

    void Inset::optimize(int layerNumber, Point& current_location, QVector<Path>& innerMostClosedContour, QVector<Path>& outerMostClosedContour, bool& shouldNextPathBeCCW)
    {
        PolylineOrderOptimizer poo(current_location, layerNumber);

        PathOrderOptimization pathOrderOptimization = static_cast<PathOrderOptimization>(
                    this->getSb()->setting<int>(Constants::ProfileSettings::Optimizations::kPathOrder));
        if(pathOrderOptimization == PathOrderOptimization::kCustomPoint)
        {
            Point startOverride(getSb()->setting<double>(Constants::ProfileSettings::Optimizations::kCustomPathXLocation),
                                getSb()->setting<double>(Constants::ProfileSettings::Optimizations::kCustomPathYLocation));

            poo.setStartOverride(startOverride);
        }

        PointOrderOptimization pointOrderOptimization = static_cast<PointOrderOptimization>(
                    this->getSb()->setting<int>(Constants::ProfileSettings::Optimizations::kPointOrder));

        if(pointOrderOptimization == PointOrderOptimization::kCustomPoint)
        {
            Point startOverride(getSb()->setting<double>(Constants::ProfileSettings::Optimizations::kCustomPointXLocation),
                                getSb()->setting<double>(Constants::ProfileSettings::Optimizations::kCustomPointYLocation));

            poo.setStartPointOverride(startOverride);
        }

        poo.setPointParameters(pointOrderOptimization, getSb()->setting<bool>(Constants::ProfileSettings::Optimizations::kMinDistanceEnabled),
                               getSb()->setting<Distance>(Constants::ProfileSettings::Optimizations::kMinDistanceThreshold),
                               getSb()->setting<Distance>(Constants::ProfileSettings::Optimizations::kConsecutiveDistanceThreshold),
                               getSb()->setting<bool>(Constants::ProfileSettings::Optimizations::kLocalRandomnessEnable),
                               getSb()->setting<Distance>(Constants::ProfileSettings::Optimizations::kLocalRandomnessRadius));

        m_paths.clear();

        if(static_cast<PrintDirection>(m_sb->setting<int>(Constants::ProfileSettings::Ordering::kInsetReverseDirection)) != PrintDirection::kReverse_off)
            for(Polyline& line : m_computed_geometry)
                line = line.reverse();

        poo.setGeometryToEvaluate(m_computed_geometry, RegionType::kInset, static_cast<PathOrderOptimization>(m_sb->setting<int>(Constants::ProfileSettings::Optimizations::kPathOrder)));

        while(poo.getCurrentPolylineCount() > 0)
        {
            Polyline result = poo.linkNextPolyline();
            Path newPath = createPath(result);

            if(newPath.size() > 0)
            {
                if(m_sb->setting<bool>(Constants::ExperimentalSettings::CurveFitting::kEnableArcFitting) ||
                        m_sb->setting<bool>(Constants::ExperimentalSettings::CurveFitting::kEnableSplineFitting))
                    CurveFitting::Fit(newPath, m_sb);

                QVector<Path> temp_path;
                calculateModifiers(newPath, m_sb->setting<bool>(Constants::PrinterSettings::MachineSetup::kSupportG3), temp_path);
                PathModifierGenerator::GenerateTravel(newPath, current_location, m_sb->setting<Velocity>(Constants::ProfileSettings::Travel::kSpeed));
                current_location = newPath.back()->end();
                m_paths.push_back(newPath);
            }
        }
    }

    Path Inset::createPath(Polyline line)
    {
        Path new_path;

        Distance default_width                  = m_sb->setting< Distance >(Constants::ProfileSettings::Inset::kBeadWidth);
        Distance default_height                 = m_sb->setting< Distance >(Constants::ProfileSettings::Layer::kLayerHeight);
        Velocity default_speed                  = m_sb->setting< Velocity >(Constants::ProfileSettings::Inset::kSpeed);
        Acceleration default_acceleration       = m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kInset);
        AngularVelocity default_extruder_speed  = m_sb->setting< AngularVelocity >(Constants::ProfileSettings::Inset::kExtruderSpeed);
        float default_esp_value                 = m_sb->setting< float >(Constants::PrinterSettings::Embossing::kESPNominalValue);
        int material_number                     = m_sb->setting< int >(Constants::MaterialSettings::MultiMaterial::kPerimterNum);

        bool embossing_enable = m_sb->setting<bool>(Constants::PrinterSettings::Embossing::kEnableEmbossing);

        for (int j = 0, end_cond = line.size(); j < end_cond; ++j)
        {
            Point start = line[j];
            Point end = line[(j + 1) % end_cond];

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

                for (Point& point : poly_intersect)
                    point.setSettings(updatedBase);

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


                    QSharedPointer<LineSegment> segment = QSharedPointer<LineSegment>::create(start, point);

                    segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            is_settings_region ? start.getSettings()->setting< Distance >(Constants::ProfileSettings::Inset::kBeadWidth) : default_width);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,       is_settings_region ? start.getSettings()->setting< Distance >(Constants::ProfileSettings::Layer::kLayerHeight) : default_height);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,        is_settings_region ? start.getSettings()->setting< Velocity >(Constants::ProfileSettings::Inset::kSpeed) : default_speed);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            is_settings_region ? start.getSettings()->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kInset) : default_acceleration);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    is_settings_region ? start.getSettings()->setting< AngularVelocity >(Constants::ProfileSettings::Inset::kExtruderSpeed) : default_extruder_speed);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kMaterialNumber,   material_number);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       RegionType::kInset);

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
            segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            is_settings_region ? start.getSettings()->setting< Distance >(Constants::ProfileSettings::Inset::kBeadWidth) : default_width);
            segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,       is_settings_region ? start.getSettings()->setting< Distance >(Constants::ProfileSettings::Layer::kLayerHeight) : default_height);
            segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,        is_settings_region ? start.getSettings()->setting< Velocity >(Constants::ProfileSettings::Inset::kSpeed) : default_speed);
            segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            is_settings_region ? start.getSettings()->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kInset) : default_acceleration);
            segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    is_settings_region ? start.getSettings()->setting< AngularVelocity >(Constants::ProfileSettings::Inset::kExtruderSpeed) : default_extruder_speed);
            segment->getSb()->setSetting(Constants::SegmentSettings::kMaterialNumber,   material_number);
            segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       RegionType::kInset);

            if (embossing_enable) {
                segment->getSb()->setSetting(Constants::SegmentSettings::kESP, is_settings_region ? start.getSettings()->setting< float >(Constants::PrinterSettings::Embossing::kESPEmbossingValue) : default_esp_value);
                if (is_settings_region) segment->getSb()->setSetting(Constants::SegmentSettings::kPathModifiers, PathModifiers::kEmbossing);
            }

            new_path.append(segment);
        }

        if (new_path.calculateLength() > m_sb->setting< Distance >(Constants::ProfileSettings::Inset::kMinPathLength))
            return new_path;
        else
            return Path();
    }

    void Inset::calculateModifiers(Path& path, bool supportsG3, QVector<Path>& innerMostClosedContour)
    {
        if(m_sb->setting<bool>(Constants::ExperimentalSettings::Ramping::kTrajectoryAngleEnabled))
        {
            PathModifierGenerator::GenerateTrajectorySlowdown(path, m_sb);
        }

        //add the modifiers
        if(m_sb->setting<bool>(Constants::MaterialSettings::Slowdown::kInsetEnable))
        {
            PathModifierGenerator::GenerateSlowdown(path, m_sb->setting<Distance>(Constants::MaterialSettings::Slowdown::kInsetDistance),
                                                    m_sb->setting<Distance>(Constants::MaterialSettings::Slowdown::kInsetLiftDistance),
                                                    m_sb->setting<Distance>(Constants::MaterialSettings::Slowdown::kInsetCutoffDistance),
                                                    m_sb->setting<Velocity>(Constants::MaterialSettings::Slowdown::kInsetSpeed),
                                                    m_sb->setting<AngularVelocity>(Constants::MaterialSettings::Slowdown::kInsetExtruderSpeed),
                                                    m_sb->setting<bool>(Constants::ProfileSettings::SpecialModes::kEnableWidthHeight),
                                                    m_sb->setting<double>(Constants::MaterialSettings::Slowdown::kSlowDownAreaModifier));
        }
        if(m_sb->setting<bool>(Constants::MaterialSettings::TipWipe::kInsetEnable))
        {
            if(static_cast<TipWipeDirection>(m_sb->setting<int>(Constants::MaterialSettings::TipWipe::kInsetDirection)) == TipWipeDirection::kForward ||
                    static_cast<TipWipeDirection>(m_sb->setting<int>(Constants::MaterialSettings::TipWipe::kInsetDirection)) == TipWipeDirection::kOptimal)
                PathModifierGenerator::GenerateTipWipe(path, PathModifiers::kForwardTipWipe, m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kInsetDistance),
                                                       m_sb->setting<Velocity>(Constants::MaterialSettings::TipWipe::kInsetSpeed),
                                                       m_sb->setting<Angle>(Constants::MaterialSettings::TipWipe::kInsetAngle),
                                                       m_sb->setting<AngularVelocity>(Constants::MaterialSettings::TipWipe::kInsetExtruderSpeed),
                                                       m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kInsetLiftHeight),
                                                       m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kInsetCutoffDistance));
            else if(static_cast<TipWipeDirection>(m_sb->setting<int>(Constants::MaterialSettings::TipWipe::kInsetDirection)) == TipWipeDirection::kAngled)
            {
                PathModifierGenerator::GenerateTipWipe(path, PathModifiers::kAngledTipWipe, m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kInsetDistance),
                                                       m_sb->setting<Velocity>(Constants::MaterialSettings::TipWipe::kInsetSpeed),
                                                       m_sb->setting<Angle>(Constants::MaterialSettings::TipWipe::kInsetAngle),
                                                       m_sb->setting<AngularVelocity>(Constants::MaterialSettings::TipWipe::kInsetExtruderSpeed),
                                                       m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kInsetLiftHeight),
                                                       m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kInsetCutoffDistance));
            }
            else
                PathModifierGenerator::GenerateTipWipe(path, PathModifiers::kReverseTipWipe, m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kInsetDistance),
                                                       m_sb->setting<Velocity>(Constants::MaterialSettings::TipWipe::kInsetSpeed),
                                                       m_sb->setting<Angle>(Constants::MaterialSettings::TipWipe::kInsetAngle),
                                                       m_sb->setting<AngularVelocity>(Constants::MaterialSettings::TipWipe::kInsetExtruderSpeed),
                                                       m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kInsetLiftHeight),
                                                       m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kInsetCutoffDistance));
        }
        if(m_sb->setting<bool>(Constants::MaterialSettings::SpiralLift::kInsetEnable))
        {
            PathModifierGenerator::GenerateSpiralLift(path, m_sb->setting<Distance>(Constants::MaterialSettings::SpiralLift::kLiftRadius),
                                                      m_sb->setting<Distance>(Constants::MaterialSettings::SpiralLift::kLiftHeight),
                                                      m_sb->setting<int>(Constants::MaterialSettings::SpiralLift::kLiftPoints),
                                                      m_sb->setting<Velocity>(Constants::MaterialSettings::SpiralLift::kLiftSpeed), supportsG3);
        }
        if(m_sb->setting<bool>(Constants::MaterialSettings::Startup::kInsetEnable))
        {
            if(m_sb->setting<bool>(Constants::MaterialSettings::Startup::kInsetRampUpEnable))
            {
                PathModifierGenerator::GenerateInitialStartupWithRampUp(path, m_sb->setting<Distance>(Constants::MaterialSettings::Startup::kInsetDistance),
                                                              m_sb->setting<Velocity>(Constants::MaterialSettings::Startup::kInsetSpeed),
                                                              m_sb->setting<Velocity>(Constants::ProfileSettings::Inset::kSpeed),
                                                              m_sb->setting<AngularVelocity>(Constants::MaterialSettings::Startup::kInsetExtruderSpeed),
                                                              m_sb->setting<AngularVelocity>(Constants::ProfileSettings::Inset::kExtruderSpeed),
                                                              m_sb->setting<int>(Constants::MaterialSettings::Startup::kInsetSteps),
                                                              m_sb->setting<bool>(Constants::ProfileSettings::SpecialModes::kEnableWidthHeight),
                                                              m_sb->setting<double>(Constants::MaterialSettings::Startup::kStartUpAreaModifier));
            }
            else
            {
                PathModifierGenerator::GenerateInitialStartup(path, m_sb->setting<Distance>(Constants::MaterialSettings::Startup::kInsetDistance),
                                                              m_sb->setting<Velocity>(Constants::MaterialSettings::Startup::kInsetSpeed),
                                                              m_sb->setting<AngularVelocity>(Constants::MaterialSettings::Startup::kInsetExtruderSpeed),
                                                              m_sb->setting<bool>(Constants::ProfileSettings::SpecialModes::kEnableWidthHeight),
                                                              m_sb->setting<double>(Constants::MaterialSettings::Startup::kStartUpAreaModifier));
            }
        }
    }

    #ifdef HAVE_SINGLE_PATH
    void Inset::setSinglePathGeometry(QVector<SinglePath::PolygonList> sp_geometry)
    {
        m_single_path_geometry = sp_geometry;
    }
    #endif

    #ifdef HAVE_SINGLE_PATH
    void Inset::createSinglePaths()
    {
        Distance perim_width                  = m_sb->setting< Distance >(Constants::ProfileSettings::Perimeter::kBeadWidth);
        Distance perim_height                 = m_sb->setting< Distance >(Constants::ProfileSettings::Layer::kLayerHeight);
        Velocity perim_speed                  = m_sb->setting< Velocity >(Constants::ProfileSettings::Perimeter::kSpeed);
        Acceleration perim_acceleration       = m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kPerimeter);
        AngularVelocity perim_extruder_speed  = m_sb->setting< AngularVelocity >(Constants::ProfileSettings::Perimeter::kExtruderSpeed);

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
                        segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       RegionType::kPerimeter);
                    }else
                    {
                        segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            (start.getRegionType() == SinglePath::RegionType::kPerimeter) ? perim_width : inset_width);
                        segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           (start.getRegionType() == SinglePath::RegionType::kPerimeter) ? perim_height : inset_height);
                        segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            (start.getRegionType() == SinglePath::RegionType::kPerimeter) ? perim_speed : inset_speed);
                        segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            (start.getRegionType() == SinglePath::RegionType::kPerimeter) ? perim_acceleration : inset_acceleration);
                        segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    (start.getRegionType() == SinglePath::RegionType::kPerimeter) ? perim_extruder_speed : inset_extruder_speed);
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
                segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       (polygon.last().getRegionType() == SinglePath::RegionType::kPerimeter) ? RegionType::kPerimeter : RegionType::kInset);

                new_path.append(segment);
                if (new_path.calculateLength() > m_sb->setting< Distance >(Constants::ProfileSettings::Perimeter::kMinPathLength))
                    m_paths.append(new_path);

            }
        }
    }
    #endif

    QVector<Path>& Inset::getOuterMostPathSet()
    {
        return m_outer_most_path_set;
    }

    QVector<Path>& Inset::getInnerMostPathSet()
    {
        return m_inner_most_path_set;
    }

    QVector<Polyline> Inset::getComputedGeometry()
    {
        return m_computed_geometry;
    }
}
