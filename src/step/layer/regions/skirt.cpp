#include "step/layer/regions/skirt.h"
#include "geometry/segments/line.h"
#include "optimizers/path_order_optimizer.h"

namespace ORNL {
    Skirt::Skirt(const QSharedPointer<SettingsBase>& sb, const QVector<SettingsPolygon>& settings_polygons) : RegionBase(sb, settings_polygons) {
        // NOP
    }

    QString Skirt::writeGCode(QSharedPointer<WriterBase> writer) {
        QString gcode;
        gcode += writer->writeBeforeRegion(RegionType::kSkirt);
        for (Path path : m_paths) {
            gcode += writer->writeBeforePath(RegionType::kSkirt);
            for (QSharedPointer<SegmentBase> segment : path.getSegments()) {
                gcode += segment->writeGCode(writer);
            }
            gcode += writer->writeAfterPath(RegionType::kSkirt);
        }
        gcode += writer->writeAfterRegion(RegionType::kSkirt);
        return gcode;
    }

    void Skirt::compute(uint layer_num, QSharedPointer<SyncManager>& sync){
        m_paths.clear();

        setMaterialNumber(m_sb->setting<int>(Constants::MaterialSettings::MultiMaterial::kPerimterNum));

        Distance beadWidth = m_sb->setting<Distance>(Constants::MaterialSettings::PlatformAdhesion::kSkirtBeadWidth);
        int rings = m_sb->setting<int>(Constants::MaterialSettings::PlatformAdhesion::kSkirtLoops);
        Distance skirt_offset = m_sb->setting<Distance>(Constants::MaterialSettings::PlatformAdhesion::kSkirtDistanceFromObject);
        Distance min_length = m_sb->setting<Distance>(Constants::MaterialSettings::PlatformAdhesion::kSkirtMinLength);
        Distance totalDist = 0;

        //construct skirp loops from the inner most loop, going outwards
        QVector<PolygonList> skirtLoops;
        for (int ring_nr = 0; ring_nr < rings; ring_nr++)
        {
            //the offset is determined by assuring that the inner most loop keeps "skirt_distance_from_object" from the part
            PolygonList path_line = m_geometry.offset(beadWidth * ring_nr + beadWidth / 2 + skirt_offset);
            skirtLoops += path_line;

            totalDist += path_line.totalLength();
            //on the outer most loop, check if the total printed distance is long enough, if not, add loops
            if(ring_nr == rings - 1)
            {
                if(min_length > totalDist)
                    rings++;
            }
        }

        //add the skirp loops to m_computed_geometry, from the outer most loop first, going inwards
        QVectorIterator<PolygonList> i(skirtLoops);
        i.toBack();
        while(i.hasPrevious())
        {
            m_computed_geometry += i.previous();
        }

        this->createPaths();
    }

    void Skirt::optimize(QSharedPointer<PathOrderOptimizer> poo, Point& current_location, QVector<Path>& innerMostClosedContour, QVector<Path>& outerMostClosedContour, bool& shouldNextPathBeCCW)
    {
        poo->setPathsToEvaluate(m_paths);
        m_paths.clear();
        while(poo->getCurrentPathCount() > 0)
            m_paths.push_back(poo->linkNextPath());
    }

    void Skirt::calculateModifiers(Path& path, bool supportsG3, QVector<Path>& innerMostClosedContour, Point& current_location)
    {
        //NOP
    }

    void Skirt::createPaths()
    {
        for (const PolygonList& polygon_list : m_computed_geometry)
        {
            for (Polygon polygon: polygon_list)
            {
                Path new_path;

                Distance default_width                  = m_sb->setting< Distance >(Constants::MaterialSettings::PlatformAdhesion::kRaftBeadWidth);
                Distance default_height                 = m_sb->setting< Distance >(Constants::ProfileSettings::Layer::kLayerHeight);
                Velocity default_speed                  = m_sb->setting< Velocity >(Constants::ProfileSettings::Layer::kSpeed);
                Acceleration default_acceleration       = m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kDefault);
                AngularVelocity default_extruder_speed  = m_sb->setting< AngularVelocity >(Constants::ProfileSettings::Layer::kExtruderSpeed);
                int material_number             = m_sb->setting< int >(Constants::MaterialSettings::MultiMaterial::kPerimterNum);

                for (int i = 0, end_cond = polygon.size(); i <  end_cond; ++i)
                {
                    Point start = polygon[i];
                    Point end = polygon[(i + 1) % end_cond];

                    bool is_settings_region = false;

                    QVector<Point> intersections;
                    for(auto settings_poly : m_settings_polygons)
                    {
                        QSharedPointer<SettingsBase> updatedBase = QSharedPointer<SettingsBase>::create(*m_sb);
                        updatedBase->populate(settings_poly.getSettings());
                        // Determine if the start and end points are in a settings region
                        if(settings_poly.inside(start))
                        {
                            start.setSettings(updatedBase);
                            is_settings_region = true;
                        }else
                        {
                            start.setSettings(m_sb);
                        }

                        if(settings_poly.inside(end))
                            end.setSettings(updatedBase);

                        // Find if/ where this line intersects with a settings polygon
                        QVector<Point> poly_intersect = settings_poly.clipLine(start, end);
                        intersections.append(poly_intersect);
                    }

                    // Divide lines into subsections
                    if(intersections.size() > 1)
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
                    
                            segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            is_settings_region ? start.getSettings()->setting< Distance >(Constants::MaterialSettings::PlatformAdhesion::kRaftBeadWidth) : default_width);
                            segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           is_settings_region ? start.getSettings()->setting< Distance >(Constants::ProfileSettings::Layer::kLayerHeight) : default_height);
                            segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            is_settings_region ? start.getSettings()->setting< Velocity >(Constants::ProfileSettings::Layer::kSpeed) : default_speed);
                            segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            is_settings_region ? start.getSettings()->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kDefault) : default_acceleration);
                            segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    is_settings_region ? start.getSettings()->setting< AngularVelocity >(Constants::ProfileSettings::Layer::kExtruderSpeed) : default_extruder_speed);
                            segment->getSb()->setSetting(Constants::SegmentSettings::kMaterialNumber,   material_number);
                            segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       RegionType::kSkirt);
							
                            new_path.append(segment);
                            is_settings_region = !is_settings_region;
                            start = point;
                        }
                    }

                    // Add final segment
                    QSharedPointer<LineSegment> segment = QSharedPointer<LineSegment>::create(start, end);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            is_settings_region ? start.getSettings()->setting< Distance >(Constants::MaterialSettings::PlatformAdhesion::kRaftBeadWidth) : default_width);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           is_settings_region ? start.getSettings()->setting< Distance >(Constants::ProfileSettings::Layer::kLayerHeight) : default_height);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            is_settings_region ? start.getSettings()->setting< Velocity >(Constants::ProfileSettings::Layer::kSpeed) : default_speed);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            is_settings_region ? start.getSettings()->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kDefault) : default_acceleration);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    is_settings_region ? start.getSettings()->setting< AngularVelocity >(Constants::ProfileSettings::Layer::kExtruderSpeed) : default_extruder_speed);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kMaterialNumber,   material_number);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       RegionType::kSkirt);

                    new_path.append(segment);
                }

                if (new_path.calculateLength() > m_sb->setting< Distance >(Constants::ProfileSettings::Layer::kMinExtrudeLength))
                    m_paths.append(new_path);
            }
        }

    }
}
