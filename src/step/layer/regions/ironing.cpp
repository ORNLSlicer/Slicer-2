// Main Module
#include "step/layer/regions/ironing.h"
#include "utilities/enums.h"
#include "geometry/segments/line.h"
#include "optimizers/polyline_order_optimizer.h"
#include "geometry/path_modifier.h"
#include "geometry/pattern_generator.h"
#include "utilities/mathutils.h"

namespace ORNL {
    Ironing::Ironing(const QSharedPointer<SettingsBase>& sb, const int index, const QVector<SettingsPolygon>& settings_polygons, const SingleExternalGridInfo& gridInfo)
        : RegionBase(sb, index, settings_polygons, gridInfo) {
    }

    QString Ironing::writeGCode(QSharedPointer<WriterBase> writer) {
        QString gcode;
        if(m_paths.count() > 0){
            gcode += writer->writeBeforeRegion(RegionType::kIroning);
            for (Path path : m_paths) {
                gcode += writer->writeBeforePath(RegionType::kIroning);
                for (QSharedPointer<SegmentBase> segment : path.getSegments()) {
                    gcode += segment->writeGCode(writer);
                }
                gcode += writer->writeAfterPath(RegionType::kIroning);
            }
            gcode += writer->writeAfterRegion(RegionType::kIroning);
        }
        return gcode;
    }

    void Ironing::compute(uint layer_num, QSharedPointer<SyncManager>& sync) {
        m_paths.clear();

        bool useLayerGeometry = true;
        if(m_sb->setting<bool>(Constants::ExperimentalSettings::Ironing::kTop)){
            for (PolygonList poly : m_upper_geometry){
                m_geometry_copy += m_geometry - poly;
                useLayerGeometry = false;
            }
        }

        if(useLayerGeometry)
            m_geometry_copy = m_geometry;

        setMaterialNumber(m_sb->setting<int>(Constants::MaterialSettings::MultiMaterial::kInfillNum));

        QVector<SettingsPolygon> settings_holes_to_fill;

        // Fill with base geometry and default settings
        fillGeometry(m_geometry_copy, m_sb);

        // Fill any regions with different settings
        for(auto settings_polygon : settings_holes_to_fill)
        {
            if(settings_polygon.getSettings()->setting<bool>(Constants::ExperimentalSettings::Ironing::kEnable))
            {
                PolygonList geometry;
                geometry += (m_geometry & settings_polygon);
                QSharedPointer<SettingsBase> region_settings = QSharedPointer<SettingsBase>::create(*m_sb);
                region_settings->setSetting(Constants::ExperimentalSettings::Ironing::kLineSpacing,
                                            settings_polygon.getSettings()->setting<Distance>(Constants::ExperimentalSettings::Ironing::kLineSpacing));

                fillGeometry(geometry, region_settings);
            }
        }
    }

    void Ironing::fillGeometry(PolygonList geometry, const QSharedPointer<SettingsBase>& sb)
    {
        bool default_global_printer_area = sb->setting<bool>(Constants::ProfileSettings::Infill::kBasedOnPrinter);
        InfillPatterns infill_pattern = static_cast<InfillPatterns>(sb->setting<int>(Constants::ProfileSettings::Infill::kPattern));
        Distance inset_offset = -(
                sb->setting<Distance>(Constants::ProfileSettings::Infill::kBeadWidth) *
                sb->setting<double>  (Constants::ExperimentalSettings::Ironing::kInsetWidth) / 100.0);

        Distance default_line_spacing = sb->setting<Distance>(Constants::ExperimentalSettings::Ironing::kLineSpacing);
        Angle default_angle = sb->setting<Angle>(Constants::ExperimentalSettings::Ironing::kAngle);

        PolygonList adjustedGeometry = geometry.offset(inset_offset);

        Point min, max;
        if(default_global_printer_area)
        {
            //! Get the bounding box for the printer
            min = Point(sb->setting<Distance>(Constants::PrinterSettings::Dimensions::kXMin), sb->setting<Distance>(Constants::PrinterSettings::Dimensions::kYMin));
            max = Point(sb->setting<Distance>(Constants::PrinterSettings::Dimensions::kXMax), sb->setting<Distance>(Constants::PrinterSettings::Dimensions::kYMax));
        }

        switch(infill_pattern)
        {
            case InfillPatterns::kGrid:
                m_computed_geometry.append(PatternGenerator::GenerateGrid(adjustedGeometry, default_line_spacing, default_angle, default_global_printer_area, min, max));
                break;
            case InfillPatterns::kLines:
            default:
                m_computed_geometry.append(PatternGenerator::GenerateLines(adjustedGeometry, default_line_spacing, default_angle, default_global_printer_area, min, max));
                break;
        }
    }

    void Ironing::optimize(int layerNumber, Point& current_location, QVector<Path>& innerMostClosedContour, QVector<Path>& outerMostClosedContour, bool& shouldNextPathBeCCW)
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
        poo.setInfillParameters(static_cast<InfillPatterns>(m_sb->setting<int>(Constants::ProfileSettings::Infill::kPattern)),
                                m_geometry_copy, getSb()->setting<Distance>(Constants::ProfileSettings::Infill::kMinPathLength),
                                getSb()->setting<Distance>(Constants::ProfileSettings::Travel::kMinLength));

        poo.setPointParameters(pointOrderOptimization, getSb()->setting<bool>(Constants::ProfileSettings::Optimizations::kMinDistanceEnabled),
                               getSb()->setting<Distance>(Constants::ProfileSettings::Optimizations::kMinDistanceThreshold),
                               getSb()->setting<Distance>(Constants::ProfileSettings::Optimizations::kConsecutiveDistanceThreshold),
                               getSb()->setting<bool>(Constants::ProfileSettings::Optimizations::kLocalRandomnessEnable),
                               getSb()->setting<Distance>(Constants::ProfileSettings::Optimizations::kLocalRandomnessRadius));

        for(QVector<Polyline> lines : m_computed_geometry)
        {
            poo.setGeometryToEvaluate(lines, RegionType::kInfill, static_cast<PathOrderOptimization>(m_sb->setting<int>(Constants::ProfileSettings::Optimizations::kPathOrder)));

            QVector<Polyline> previouslyLinkedLines;
            while(poo.getCurrentPolylineCount() > 0)
            {
                Polyline result = poo.linkNextPolyline(previouslyLinkedLines);
                if(result.size() > 0)
                {
                    Path newPath = createPath(result);
                    if(newPath.size() > 0)
                    {
                        PathModifierGenerator::GenerateTravel(newPath, current_location, m_sb->setting<Velocity>(Constants::ProfileSettings::Travel::kSpeed));
                        current_location = newPath.back()->end();
                        previouslyLinkedLines.push_back(result);
                        m_paths.push_back(newPath);
                    }
                }
            }
        }
    }

    Path Ironing::createPath(Polyline line) {
        Distance width                  = m_sb->setting< Distance >(Constants::ProfileSettings::Infill::kBeadWidth);
        Distance height                 = m_sb->setting< Distance >(Constants::ProfileSettings::Layer::kLayerHeight);
        Velocity speed                  = m_sb->setting< Velocity >(Constants::ExperimentalSettings::Ironing::kSpeed);
        Acceleration acceleration       = m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kInfill);
        AngularVelocity extruder_speed  = m_sb->setting< AngularVelocity >(Constants::ExperimentalSettings::Ironing::kExtruderSpeed);
        int material_number             = m_sb->setting< int >(Constants::MaterialSettings::MultiMaterial::kInfillNum);

        Path newPath;
        for(int i = 0, end = line.size() - 1; i < end; ++i)
        {
            QSharedPointer<LineSegment> segment = QSharedPointer<LineSegment>::create(line[i], line[i + 1]);

            segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            width);
            segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           height);
            segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            speed);
            segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            acceleration);
            segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    extruder_speed);
            segment->getSb()->setSetting(Constants::SegmentSettings::kMaterialNumber,   material_number);
            segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       RegionType::kIroning);

            newPath.append(segment);
        }

        if(newPath.calculateLength() > m_sb->setting<Distance>(Constants::ProfileSettings::Layer::kMinExtrudeLength))
            return newPath;
        else
            return Path();
    }

    void Ironing::calculateModifiers(Path& path, bool supportsG3, QVector<Path>& innerMostClosedContour) {
    }

    void Ironing::addUpperGeometry(const PolygonList& poly_list) {
        m_upper_geometry.push_back(poly_list);
    }
}
