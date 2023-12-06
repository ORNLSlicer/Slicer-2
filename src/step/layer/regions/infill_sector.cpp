#include "step/layer/regions/infill_sector.h"
#include "geometry/segments/line.h"
#include "optimizers/path_order_optimizer.h"
#include "geometry/path_modifier.h"
#include "geometry/pattern_generator.h"
#include "geometry/point.h"
#include "utilities/mathutils.h"

namespace ORNL {
    InfillSector::InfillSector(const QSharedPointer<SettingsBase>& sb, const int index, const QVector<SettingsPolygon>& settings_polygons) : RegionBase(sb, index, settings_polygons) {
        // NOP
    }

    QString InfillSector::writeGCode(QSharedPointer<WriterBase> writer) {
        QString gcode;

        if(m_paths.size() > 0)
        {
            gcode += writer->writeBeforeRegion(RegionType::kInfill, m_paths.size());
            for (Path path : m_paths) {
                for (QSharedPointer<SegmentBase> segment : path.getSegments()) {
                    gcode += segment->writeGCode(writer);
                }
            }
            gcode += writer->writeAfterPath(RegionType::kInfill);
        }
        else
            gcode += writer->writeEmptyStep();

        return gcode;
    }

    void InfillSector::compute(uint layer_num, QSharedPointer<SyncManager>& sync){
        m_paths.clear();

        Point center(m_sb->setting<double>(Constants::PrinterSettings::Dimensions::kXOffset), m_sb->setting<double>(Constants::PrinterSettings::Dimensions::kYOffset));
        InfillPatterns infillPattern = static_cast<InfillPatterns>(m_sb->setting<int>(Constants::ProfileSettings::Infill::kPattern));
        Distance lineSpacing = m_sb->setting<Distance>(Constants::ProfileSettings::Infill::kLineSpacing);
        Distance hatchBeadWidth = m_sb->setting<Distance>(Constants::ProfileSettings::Infill::kBeadWidth);

        //kAngle in the setting has already been updated for each layer
        Angle infillAngle = m_sb->setting<Angle>(Constants::ProfileSettings::Infill::kAngle);
        Point min, max;
        bool globalPrinterArea = m_sb->setting<bool>(Constants::ProfileSettings::Infill::kBasedOnPrinter);
        if(globalPrinterArea)
        {
            //! Get the bounding box for the printer
            min = Point(m_sb->setting<Distance>(Constants::PrinterSettings::Dimensions::kXMin), m_sb->setting<Distance>(Constants::PrinterSettings::Dimensions::kYMin));
            max = Point(m_sb->setting<Distance>(Constants::PrinterSettings::Dimensions::kXMax), m_sb->setting<Distance>(Constants::PrinterSettings::Dimensions::kYMax));
        }

        PolygonList geometry_copy = m_geometry;
        //Adjust for overlap
        Distance default_overlap = m_sb->setting<Distance>(Constants::ProfileSettings::Infill::kOverlap);
        geometry_copy = m_geometry.offset(default_overlap);

        switch(infillPattern)
        {
            case InfillPatterns::kLines:
                 m_computed_geometry = PatternGenerator::GenerateLines(geometry_copy, lineSpacing, infillAngle, globalPrinterArea, min, max);
            break;
            case InfillPatterns::kGrid:
                 m_computed_geometry = PatternGenerator::GenerateGrid(geometry_copy, lineSpacing, infillAngle, globalPrinterArea, min, max);
            break;
            case InfillPatterns::kConcentric:
                 m_computed_geometry = PatternGenerator::GenerateConcentric(geometry_copy, hatchBeadWidth, lineSpacing);
            break;
            case InfillPatterns::kTriangles:
                 m_computed_geometry = PatternGenerator::GenerateTriangles(geometry_copy, lineSpacing, infillAngle, globalPrinterArea, min, max);
            break;
            case InfillPatterns::kHexagonsAndTriangles:
                 m_computed_geometry = PatternGenerator::GenerateHexagonsAndTriangles(geometry_copy, lineSpacing, infillAngle, globalPrinterArea, min, max);
            break;
            case InfillPatterns::kHoneycomb:
                 m_computed_geometry = PatternGenerator::GenerateHoneyComb(geometry_copy, hatchBeadWidth, lineSpacing, infillAngle, globalPrinterArea, min, max);
            break;
            case InfillPatterns::kRadialHatch:
                m_computed_geometry = PatternGenerator::GenerateRadialHatch(geometry_copy, center, lineSpacing, m_sector_angle, infillAngle);
           break;
        }

        QVector<QPair<Distance, Polyline>> circularOrder;
        for(Polyline& line : m_computed_geometry)
        {
            Angle ang1 = MathUtils::internalAngle(line.first(), center, m_start_vec);
            Angle ang2 = MathUtils::internalAngle(line.last(), center, m_start_vec);

            if(ang2 < ang1)
                line = line.reverse();
        }

        for(Polyline line : m_computed_geometry)
        {
            QPair<Distance, Polyline> pair;
            pair.first = center.distance(line.first());
            pair.second = line;
            circularOrder.push_back(pair);
        }

//        bool shouldReverse = static_cast<PrintDirection>(m_sb->setting<int>(Constants::ProfileSettings::Ordering::kPerimeterReverseOrder))
//                == PrintDirection::kReverse_All_Layers;

//        std::sort(std::begin(circularOrder), std::end(circularOrder),
//                  [&](const auto& a, const auto& b)
//        {
//            if(shouldReverse)
//                return a.first < b.first;
//            else
//                return a.first > b.first;
//        });

        std::reverse(m_computed_geometry.begin(), m_computed_geometry.end());

        uniform(m_computed_geometry);

        this->createPaths();
    }

    void InfillSector::uniform(QVector<Polyline>& sector)
    {
        for(int i = 1, end = sector.size(); i < end; i += 2)
            sector[i] = sector[i].reverse();
    }

    void InfillSector::skipRaster(QVector<Polyline>& sector)
    {
//        QVector<Polyline> firstHalf, secondHalf;

//        bool toggle = false;
//        std::partition_copy(sector.begin(),
//                            sector.end(),
//                            std::back_inserter(firstHalf),
//                            std::back_inserter(secondHalf),
//                            [&toggle]() { return toggle = !toggle; });

//        raster(firstHalf);
//        raster(secondHalf);

//        toggle = false;
//        sector.clear();
//        sector.reserve(firstHalf.size() + secondHalf.size());
//        while(firstHalf.size() > 0 || secondHalf.size() > 0)
//        {
//            if(toggle)
//            {
//                if(firstHalf.size() > 0)
//                {
//                    sector.push_back(firstHalf.front());
//                    firstHalf.pop_front();
//                }
//            }
//            else
//            {
//                if(secondHalf.size() > 0)
//                {
//                    sector.push_back(secondHalf.front());
//                    secondHalf.pop_front();
//                }
//            }
//            toggle = !toggle;
//        }
    }

//    void Sector::stripedRaster(Point center, QVector<Polyline>& sector)
//    {
//        //! Create a non-path border for the infill pattern
//        //! Typical use for RPBF, 0 = y axis so must start at 90 degrees
//        double start = M_PI / 2;
//        double step = 2 * M_PI / sectors;
//        Point vertAxis = m_center + Point(0, 1);
//        double angleAdjust = 0;

//        Point min, max;
//        for(Polyline line : sector)
//        {
//            if(line.first().x() < min)
//            {

//            }
//            if(line.first().y() < min)
//        }
//        QVector<Polyline> sector_copy = sector;
//        for(int i = 0; i < 3; ++i)
//        {
//            QVector<Polyline> allLines;
//            for(Polyline line : sector)
//            {
//                double spiralAngle = 0;
//                line = line.rotateAround(center, spiralAngle);

//                double angle = (start + step * i) - angleAdjust;
//                Polyline rotatedLine = line.rotateAround(center, angle);

//                PolygonList rotatedGeo = m_geometry.rotateAround(center, angle);
//                Point m_min = rotatedGeo.min();
//                Point m_max = rotatedGeo.max();

//                Polygon polyClip;
//                polyClip.append(m_center);

//                //radius needs to extend slightly past bounding box of object
//                Distance radius = (m_max.distance(m_min) + 10) / 2.0;
//                polyClip.append(Point(center.x() + radius, center.y() - radius));

//                QVector<Polyline> newLine;
//                //if few sectors, clipping polygon cannot be a triangle, additional points required
//                polyClip.append(Point(center.x() + radius * qCos(step + (angleAdjust * 2)), center.y() +  radius * qSin(step + (angleAdjust * 2))));

//                newLine = polyClip & rotatedLine;
//            }
//        }
//    }

    void InfillSector::optimize(QSharedPointer<PathOrderOptimizer> poo, Point& current_location, QVector<Path>& innerMostClosedContour, QVector<Path>& outerMostClosedContour, bool& shouldNextPathBeCCW)
    {
//        Point center;
//        int i = 0;
//        for(QVector<Polyline>& sector : m_computed_infill)
//        {
//            if(i == 1)
//                raster(sector);
//            else if(i == 2)
//                skipRaster(sector);
////            else if(i == 3)
////                stripedRaster(center, sector);
//        }
    }

    void InfillSector::calculateModifiers(Path& path, bool supportsG3, QVector<Path>& innerMostClosedContour, Point& current_location)
    {
       //NOP
    }

    void InfillSector::createPaths() {
        Distance width                  = m_sb->setting< Distance >(Constants::ProfileSettings::Infill::kBeadWidth);
        Distance height                 = m_sb->setting< Distance >(Constants::ProfileSettings::Layer::kLayerHeight);
        Velocity speed                  = m_sb->setting< Velocity >(Constants::ProfileSettings::Infill::kSpeed);
        Acceleration acceleration       = m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kInfill);
        AngularVelocity extruder_speed  = m_sb->setting< AngularVelocity >(Constants::ProfileSettings::Infill::kExtruderSpeed);

        for(Polyline line : m_computed_geometry)
        {
            Path newPath;
            for (int i = 0, end = line.size() - 1; i < end; ++i) {

                QSharedPointer<LineSegment> segment = QSharedPointer<LineSegment>::create(line[i], line[i + 1]);

                segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            width);
                segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           height);
                segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            speed);
                segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            acceleration);
                segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    extruder_speed);
                segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       RegionType::kInfill);

                newPath.append(segment);
            }
            m_paths.push_back(newPath);
        }
    }

    void InfillSector::setSectorAngle(Angle angle)
    {
        m_sector_angle = angle;
    }

    void InfillSector::setStartVector(Point p)
    {
        m_start_vec = p;
    }
}
