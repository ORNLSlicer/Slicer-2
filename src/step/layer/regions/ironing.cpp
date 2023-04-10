// Main Module
#include "step/layer/regions/ironing.h"
#include "utilities/enums.h"
#include "geometry/segments/line.h"
#include "optimizers/path_order_optimizer.h"
#include "geometry/path_modifier.h"
#include "geometry/pattern_generator.h"
#include "utilities/mathutils.h"
#include "geometry/curve_fitting.h"

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
        m_region_paths.clear();

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

        this->createPaths();
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

    void Ironing::optimize(QSharedPointer<PathOrderOptimizer> poo, Point& current_location, QVector<Path>& innerMostClosedContour, QVector<Path>& outerMostClosedContour, bool& shouldNextPathBeCCW)
    {
        if(m_paths.count() < 1) return;

        bool supportsG3 = m_sb->setting<bool>(Constants::PrinterSettings::MachineSetup::kSupportG3);
        InfillPatterns IroningPattern = static_cast<InfillPatterns>(m_sb->setting<int>(Constants::ProfileSettings::Infill::kPattern));
        poo->setPathsToEvaluate(m_paths);
        poo->setParameters(IroningPattern, m_geometry_copy);

        QVector<Path> new_paths;
        while(poo->getCurrentPathCount() > 0)
        {
            Path new_path = poo->linkNextPath(new_paths);
            if(new_path.size() > 0)
            {
                calculateModifiers(new_path, supportsG3, innerMostClosedContour, poo->getCurrentLocation());
                new_paths.append(new_path);
            }
        }

        for(QVector<Path> paths : m_region_paths)
        {
            poo->setPathsToEvaluate(paths);
            poo->setParameters(IroningPattern, m_geometry);

            while(poo->getCurrentPathCount() > 0)
            {
                Path new_path = poo->linkNextPath(new_paths);
                if(new_path.size() > 0)
                {
                    calculateModifiers(new_path, supportsG3, innerMostClosedContour, poo->getCurrentLocation());
                    new_paths.append(new_path);
                }
            }
        }

        m_paths = new_paths;
    }

    void Ironing::createPaths() {
        Distance width                  = m_sb->setting< Distance >(Constants::ProfileSettings::Infill::kBeadWidth);
        Distance height                 = m_sb->setting< Distance >(Constants::ProfileSettings::Layer::kLayerHeight);
        Velocity speed                  = m_sb->setting< Velocity >(Constants::ExperimentalSettings::Ironing::kSpeed);
        Acceleration acceleration       = m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kInfill);
        AngularVelocity extruder_speed  = m_sb->setting< AngularVelocity >(Constants::ExperimentalSettings::Ironing::kExtruderSpeed);
        int material_number             = m_sb->setting< int >(Constants::MaterialSettings::MultiMaterial::kInfillNum);

        m_region_paths.resize(m_computed_geometry.size() - 1);
        for(int i = 0, end = m_computed_geometry.size(); i < end; ++i)
        {
            QVector<Polyline> polylineList = m_computed_geometry[i];

            for(Polyline polyline : polylineList)
            {
                Path newPath;
                for (int j = 0, polyEnd = polyline.size() - 1; j < polyEnd; ++j)
                {
                    QSharedPointer<LineSegment> segment = QSharedPointer<LineSegment>::create(polyline[j], polyline[j + 1]);

                    segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            width);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           height);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            speed);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            acceleration);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    extruder_speed);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kMaterialNumber,   material_number);
                    segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       RegionType::kIroning);

                    newPath.append(segment);
                }

                if(i == 0)
                    m_paths.append(newPath);
                else
                    m_region_paths[i - 1].append(newPath);
            }
        }
    }

    void Ironing::calculateModifiers(Path& path, bool supportsG3, QVector<Path>& innerMostClosedContour, Point& current_location) {
    }

    void Ironing::addUpperGeometry(const PolygonList& poly_list) {
        m_upper_geometry.push_back(poly_list);
    }
}
