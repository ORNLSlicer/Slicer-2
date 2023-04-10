#include "threading/slicers/conformal_slicer.h"

#include "managers/session_manager.h"
#include "managers/settings/settings_manager.h"
#include "part/part.h"
#include "step/layer/layer.h"
#include "step/layer/island/polymer_island.h"
#include "utilities/mathutils.h"
#include "geometry/mesh/closed_mesh.h"
#include "geometry/mesh/open_mesh.h"

namespace ORNL{

    ConformalSlicer::ConformalSlicer(QString gcodeLocation) : TraditionalAST(gcodeLocation){}

    void ConformalSlicer::preProcess(nlohmann::json opt_data)
    {
        // Fetch settings
        QSharedPointer<SettingsBase> global_sb = GSM->getGlobal();
        int layer_count =      global_sb->setting<int>(Constants::ExperimentalSettings::ConformalSlicing::kConformalLayers);
        Angle stacking_pitch = global_sb->setting<Angle>(Constants::ExperimentalSettings::SlicingAngle::kStackingDirectionPitch);
        Angle stacking_yaw   = global_sb->setting<Angle>(Constants::ExperimentalSettings::SlicingAngle::kStackingDirectionYaw);
        Angle stacking_roll  = global_sb->setting<Angle>(Constants::ExperimentalSettings::SlicingAngle::kStackingDirectionRoll);

        // Build a normal vector using the quaternion
        QVector3D normal_vector(0, 0, 1);
        QQuaternion quaternion = MathUtils::CreateQuaternion(stacking_pitch, stacking_yaw, stacking_roll);
        normal_vector = quaternion.rotatedVector(normal_vector).normalized();

        int total_layers = (layer_count * CSM->parts().size());

        int layer_index = 1;
        for(auto part : CSM->parts())
        {   
            // Get settings
            auto part_ranges = part->ranges();

            auto mesh = part->rootMesh();
            auto surface = mesh->extractUpwardFaces();

            // Compute parameterization
            QSharedPointer<Parameterization> parameterization = QSharedPointer<Parameterization>::create(surface, mesh->dimensions());

            if(part->countStepPairs() > layer_count)
            {
                // Need to remove layers since there are too many now
                for(int i = 0, end = part->countStepPairs() - layer_count; i < end; ++i)
                {
                    part->removeStepAtIndex(part->countStepPairs() - 1);
                }
            }

            // Builds layers according the settings
            for(int i =  1, end = layer_count; i <= end; ++i, ++layer_index)
            {
                QSharedPointer<SettingsBase> layerSpecificSettings = QSharedPointer<SettingsBase>::create(*global_sb);
                for(auto range : part_ranges)
                {

                    if(range->includesIndex(layer_count) && !range->getSb()->json().is_null())
                    {
                        QSharedPointer<SettingsBase> settings = range->getSb();
                        layerSpecificSettings->populate(settings);
                    }
                }

                Distance layer_height = layerSpecificSettings->setting<Distance>(Constants::ProfileSettings::Layer::kLayerHeight);

                if(i <= part->countStepPairs()) // The layer already exists
                {
                    Layer& existing_layer = dynamic_cast<Layer&>(*part->step(i - 1, StepType::kLayer));
                    existing_layer.flagIfDirtySettings(layerSpecificSettings);

                    // If they layer is dirty then we need to build a new one
                    if(existing_layer.isDirty())
                    {
                        // Extract geometry from parameterization
                        PolygonList polygons;
                        polygons += parameterization->getBoarderPolygon();

                        QSharedPointer<PolymerIsland> island = QSharedPointer<PolymerIsland>::create(PolymerIsland(polygons, layerSpecificSettings, QVector<SettingsPolygon>()));

                        // Add to island to layer
                        QSharedPointer<Layer> layer = QSharedPointer<Layer>::create(Layer(layer_index, layerSpecificSettings));
                        layer->addIsland(IslandType::kPolymer, island);
                        layer->setParameterization(parameterization);
                        layer->setNormalOffset(normal_vector * layer_height() * layer_index);

                        part->replaceStep(i - 1, layer);
                    }

                }else
                {
                    // Extract geometry from parameterization
                    PolygonList polygons;
                    polygons += parameterization->getBoarderPolygon();

                    QSharedPointer<PolymerIsland> island = QSharedPointer<PolymerIsland>::create(PolymerIsland(polygons, layerSpecificSettings, QVector<SettingsPolygon>()));

                    // Add to island to layer
                    QSharedPointer<Layer> layer = QSharedPointer<Layer>::create(Layer(layer_index, layerSpecificSettings));
                    layer->addIsland(IslandType::kPolymer, island);
                    layer->setParameterization(parameterization);
                    layer->setNormalOffset(normal_vector * layer_height()  * layer_index);

                    part->appendStep(layer);
                }

                emit statusUpdate(StatusUpdateStepType::kPreProcess, (double)(layer_index) / (double)total_layers * 100);
            }

            // Ensures the progress bar is marked as done
            emit statusUpdate(StatusUpdateStepType::kPreProcess, 100);
        }

    }

    void ConformalSlicer::postProcess(nlohmann::json opt_data)
    {
        // Count number of steps for status bar
        int total_steps = 0;
        bool anyDirty = false;
        for(auto part : CSM->parts())
        {
            if(part->isPartDirty())
                anyDirty = true;

            total_steps += part->countStepPairs();
        }

        if(anyDirty)
        {
            Point currentLocation(0,0,0);
            int step_index = 1;
            int start_index = -1;
            QVector<QSharedPointer<RegionBase>> previousRegions;
            for(auto part : CSM->parts())
            {
                for(QSharedPointer<Step> step : part->steps())
                {
                    QSharedPointer<Layer> layer = step.dynamicCast<Layer>();
                    layer->connectPaths(currentLocation, start_index, previousRegions);
                    currentLocation = layer->getFinalLayerLocation();

                    emit statusUpdate(StatusUpdateStepType::kPostProcess, (double)(step_index) / (double)total_steps * 100);
                    step_index++;
                }
            }
        }else
        {
            // Ensures the progress bar is marked as done
            emit statusUpdate(StatusUpdateStepType::kPostProcess, 100);
        }
    }

    void ConformalSlicer::writeGCode()
    {
        QTextStream stream(&m_temp_gcode_output_file);
        QSharedPointer<WriterBase> base = m_base;
        if(this->shouldCancel())
            return;

        // Count number of steps for status bar
        int total_steps = 0;
        for(auto part : CSM->parts())
            total_steps += part->countStepPairs();

        int layer_num = 1;
        for(auto part : CSM->parts())
        {
            for(QSharedPointer<Step> step : part->steps(StepType::kAll))
            {
                step->setDirtyBit(false);

                stream << base->writeBeforePart();

                QSharedPointer<Layer> layer = step.dynamicCast<Layer>();
                layer_num++;

                stream << base->writeLayerChange(layer_num);
                stream << base->writeBeforeLayer(layer_num, layer->getSb());
                stream << layer->writeGCode(base);
                stream << base->writeAfterLayer();

                stream << base->writeAfterPart();
                emit statusUpdate(StatusUpdateStepType::kGcodeGeneraton, (double)(layer_num - 1) / (double)total_steps * 100);
            }
        }
        stream << base->writeShutdown();
    }
}

