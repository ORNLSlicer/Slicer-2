#include "threading/slicers/skeleton_slicer.h"

#include "managers/settings/settings_manager.h"
#include "slicing/preprocessor.h"
#include "geometry/curve_fitting.h"

namespace ORNL
{
    SkeletonSlicer::SkeletonSlicer(QString gcodeLocation) : TraditionalAST(gcodeLocation)
    {

    }

    void SkeletonSlicer::preProcess(nlohmann::json opt_data)
    {
        m_skeleton_layers.clear();

        Preprocessor pp(true); // This pre-processor will use cgal cross-sectioning

        pp.addStepBuilder([this](QSharedPointer<BufferedSlicer::SliceMeta> next_layer_meta, Preprocessor::ActivePartMeta& meta)
        {
            auto polylines = next_layer_meta->opt_polylines;

            for(auto& polygon : next_layer_meta->geometry)
                polylines.append(polygon.toPolyline());

            Point last_pos(0,0,0);

            SkeletonLayer layer;
            for(auto& polyline : polylines)
            {
                Path new_path;

                bool first = true;

                for(auto point : polyline)
                {
                    QSharedPointer<SegmentBase> segment;

                    if(first)
                        segment = QSharedPointer<TravelSegment>::create(m_last_pos, point);
                    else
                        segment = QSharedPointer<LineSegment>::create(m_last_pos, point);

                    first = false;

                    segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            GSM->getGlobal()->setting< Distance >(Constants::ProfileSettings::Perimeter::kBeadWidth));
                    segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           GSM->getGlobal()->setting< Distance >(Constants::ProfileSettings::Layer::kLayerHeight));
                    segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            GSM->getGlobal()->setting< Velocity >(Constants::ProfileSettings::Perimeter::kSpeed));
                    segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            GSM->getGlobal()->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kPerimeter));
                    segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    GSM->getGlobal()->setting< AngularVelocity >(Constants::ProfileSettings::Perimeter::kExtruderSpeed));
                    segment->getSb()->setSetting(Constants::SegmentSettings::kMaterialNumber,   GSM->getGlobal()->setting< int >(Constants::MaterialSettings::MultiMaterial::kPerimterNum));
                    segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       RegionType::kPerimeter);

                    new_path.append(segment);
                    m_last_pos = point;
                }

                layer.push_back(new_path);
            }
            m_skeleton_layers.push_back(layer);

            return false; // No error, so continune slicing
        });

        pp.addStatusUpdate([this](double percentage)
        {
            emit statusUpdate(StatusUpdateStepType::kPreProcess, percentage);
        });

        pp.processAll();
    }

    void SkeletonSlicer::postProcess(nlohmann::json opt_data)
    {
        emit statusUpdate(StatusUpdateStepType::kPostProcess, 100);
    }

    void SkeletonSlicer::writeGCode()
    {
        QTextStream stream(&m_temp_gcode_output_file);
        QSharedPointer<WriterBase> base = m_base;

        int layer_count = m_skeleton_layers.size();

        int layer_index = 1;
        for(auto& layer : m_skeleton_layers)
        {
            stream << base->writeLayerChange(layer_index);
            stream << base->writeBeforeLayer(0, GSM->getGlobal());

            for(auto& path : layer)
            {
                for(auto& segment : path)
                {
                    stream << segment->writeGCode(base);
                }
            }

            stream << base->writeAfterLayer();
            emit statusUpdate(StatusUpdateStepType::kGcodeGeneraton, (layer_index) / layer_count * 100);
            ++layer_index;
        }

    }
}
