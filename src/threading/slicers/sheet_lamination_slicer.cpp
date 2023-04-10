// Main Module
#include "threading/slicers/sheet_lamination_slicer.h"

// Qt
#include <QSharedPointer>

// Local
#include "managers/settings/settings_manager.h"
#include "managers/session_manager.h"
#include "slicing/slicing_utilities.h"
#include "slicing/preprocessor.h"
#include "slicing/buffered_slicer.h"
#include "step/layer/island/polymer_island.h"
#include "step/layer/island/support_island.h"
#include "step/layer/regions/skin.h"
#include "utilities/mathutils.h"
#include "optimizers/multi_nozzle_optimizer.h"
#include "slicing/layer_additions.h"
#include "clipper.hpp"
#include "geometry/search_cell.h"

#include "gcode/writers/sheet_lamination_writer.h"

//#include <QMessageBox>
#include <QDebug>

namespace ORNL {

    SheetLaminationSlicer::SheetLaminationSlicer(QString gcodeLocation) : TraditionalAST(gcodeLocation) {}

    void SheetLaminationSlicer::preProcess(nlohmann::json opt_data)
    {
        m_islands.clear();
        m_layer_list.clear();
        m_island_z_values.clear();
        m_offsets.clear();
        m_layer_changes.clear();

        Preprocessor pp;

        pp.addInitialProcessing([this](const Preprocessor::Parts& parts,  const QSharedPointer<SettingsBase>& global_settings){
            // Alter settings
            global_settings->makeGlobalAdjustments();

            // Check for overlaps of settings parts and prevent them
            if(SlicingUtilities::doPartsOverlap(parts.settings_parts, Plane(Point(1,1,1), QVector3D(0, 0, 1))))
                return true; // Cancel Slicing

            if(global_settings->setting<bool>(Constants::ExperimentalSettings::SlicingAngle::kEnableMultiBranch))
                SlicingUtilities::SegmentRoot(global_settings, CSM->parts());

            return false; // No error, so continune slicing
        });

        pp.addPartProcessing([this](QSharedPointer<Part> part, QSharedPointer<SettingsBase> part_sb){
           m_saved_layer_settings.clear();

           part->clearSteps();

           return false;
        });

        pp.addMeshProcessing([this](QSharedPointer<MeshBase> mesh, QSharedPointer<SettingsBase> part_sb){
            // Clip meshes
            auto clipping_meshes = SlicingUtilities::GetMeshesByType(CSM->parts(), MeshType::kClipping);
            SlicingUtilities::ClipMesh(mesh, clipping_meshes);

            return false; // No error, so continune slicing
        });

        pp.addStepBuilder([this](QSharedPointer<BufferedSlicer::SliceMeta> next_layer_meta, Preprocessor::ActivePartMeta& meta){
            // Save settings
            m_saved_layer_settings.push_back(next_layer_meta->settings);

            // from polymer slicer if-statement
            // Create the islands from the geometry.
            QVector<PolygonList> split_geometry = next_layer_meta->geometry.splitIntoParts();

            for (const PolygonList& island_geometry : split_geometry)
            {
                m_islands.push_back(island_geometry);
                m_layer_list.push_back(next_layer_meta->number);
            }

            emit statusUpdate(StatusUpdateStepType::kPreProcess, 100);
            return false; // No error, so continune slicing
        });

        pp.addCrossSectionProcessing([this](Preprocessor::ActivePartMeta& meta){
            // If fewer layers than last slice, remove all steps from that layer onwards
            meta.part->clearStepsFromIndex(meta.last_step_count + meta.part_start);

            // Update max steps
            if(meta.part->countStepPairs() > this->getMaxSteps())
                this->setMaxSteps(meta.part->countStepPairs());

            return false; // No error, so continune slicing
        });

        pp.addStatusUpdate([this](double percentage){
            emit statusUpdate(StatusUpdateStepType::kCompute, 100);
        });

        pp.addFinalProcessing([this](const Preprocessor::Parts& parts,  const QSharedPointer<SettingsBase>& global_settings){

            return false; // No error, so continune slicing
        });

        pp.processAll();
    }



    void SheetLaminationSlicer::postProcess(nlohmann::json opt_data)
    {
        // bounds of the print table (cutting table)
        float min_x = GSM->getGlobal()->setting<float>(Constants::PrinterSettings::Dimensions::kXMin);
        float max_x = GSM->getGlobal()->setting<float>(Constants::PrinterSettings::Dimensions::kXMax);
        float min_y = GSM->getGlobal()->setting<float>(Constants::PrinterSettings::Dimensions::kYMin);
        float max_y = GSM->getGlobal()->setting<float>(Constants::PrinterSettings::Dimensions::kYMax);

        // layer height
        // This is how to get layer height from user settings. Extremely buggy set to anything other than one inch
        //float layer_height = GSM->getGlobal()->setting<float>(Constants::ProfileSettings::Layer::kLayerHeight);
        float layer_height = 25400; // set to one inch so that layers are whole numbers

        // user settings
        /* STANDARD SETTINGS: top-center gravity; no gap between shapes
        Point gravity_point(max_x/2,-999999999,0);
        Distance gap = 0 * inches;
        */
        float gravity_point_x = GSM->getGlobal()->setting<float>(Constants::ExperimentalSettings::SheetLaminationSlicing::kGravityPointX);
        float gravity_point_y = GSM->getGlobal()->setting<float>(Constants::ExperimentalSettings::SheetLaminationSlicing::kGravityPointY);
        float gap_value = GSM->getGlobal()->setting<float>(Constants::ExperimentalSettings::SheetLaminationSlicing::kGap);
        Point gravity_point(gravity_point_x/25400 * inches, gravity_point_y/25400 * inches, 0);
        Distance gap = gap_value/25400 * inches;

        // bin that we're packing the shapes into, resized whenever it fills up
        // this resizing will reset the print table for the next layer
        // we will also use this polygon as the worker to run the noFitPolygon method
        Polygon bin_poly({Point(min_x, min_y, 0), Point(min_x, max_y * 1, 0), Point(max_x, max_y * 1, 0), Point(max_x, min_y, 0)});

        // First step, pack first polygon
        QVector<float> offset; // record the offset vector of each polygon so that a robotic arm would know exactly where to stack each island, essentially reverse packing them
        float first_island_min_x = m_islands[0].getOutsidePolygons()[0].min().x();
        float first_island_min_y = m_islands[0].getOutsidePolygons()[0].min().y();
        offset.push_back(0 - first_island_min_x);
        offset.push_back(0 - first_island_min_y);
        m_offsets.push_back(offset);
        offset.clear();
        for (int i = 0, size_i = m_islands[0].length(); i < size_i; i++)
        {
            m_islands[0][i] = m_islands[0][i].translate({0 - first_island_min_x, 0 - first_island_min_y, 0});
            m_island_z_values.push_back(0);
        }

        m_resize_counter = 1;
        m_layer_changes.push_back(0); // initial layer change at index 0
        int safety_counter = 0;
        int index_of_first_island = 0; // set to whatever the m_islands index of the first shape in the current layer is (to ignore previous layers)
        for (int i = 1, size_i = m_islands.length(); i < size_i; i++)
        { // pack the rest of the polygons
            safety_counter++;
            assert (safety_counter < 2000); // assertion to avoid an infinite loop. Should be set to a reasonably large number to accommodate more complex prints

            Polygon bin_NFP = bin_poly.noFitPolygon(bin_poly, m_islands[i].getOutsidePolygons()[0], true); //inner NFP to clip the polylines
            PolygonList NFP_union;

            for (int j = index_of_first_island; j < i; j++)
            { // merge all the other nfps to the NFP_union as you generate them
                Polygon curr_NFP = bin_poly.noFitPolygon(m_islands[j].getOutsidePolygons()[0].offset(gap)[0], m_islands[i].getOutsidePolygons()[0], false);
                NFP_union += curr_NFP;
            }

            QVector<Polyline> no_fit_polylines;
            for (int j = 0, size_j = NFP_union.length(); j < size_j; j++)
            { // convert all the nfps to polylines
                no_fit_polylines.push_back(NFP_union[j].toPolyline());
            }

            assert (!no_fit_polylines.empty()); // Likely will never be hit, here just in case

            QVector<Polyline> no_fit_polylines_clipped;
            for (int j = 0, size_j = no_fit_polylines.length(); j < size_j; j++)
            { // individually clip each polyline using the bin_NFP and put the resulting polylines in a list
                QVector<Polyline> clipped_poly = bin_NFP & no_fit_polylines[j];
                for (int k = 0, size_k = clipped_poly.length(); k < size_k; k++)
                {
                    no_fit_polylines_clipped.push_back(clipped_poly[k]);
                }
            }

            if (no_fit_polylines_clipped.empty())
            { // move to next layer and ignore all previous polygons if impossible to fit shape
                m_resize_counter++; // it should work assuming the code before it also works
                index_of_first_island = i;
                m_layer_changes.push_back(i);

                first_island_min_x = m_islands[i].getOutsidePolygons()[0].min().x(); // move first polygon of this layer
                first_island_min_y = m_islands[i].getOutsidePolygons()[0].min().y();
                offset.push_back(0 - first_island_min_x);
                offset.push_back(0 - first_island_min_y);
                m_offsets.push_back(offset);
                offset.clear();
                for (int j = 0, size_j = m_islands[i].length(); j < size_j; j++)
                { // pack first polygon of layer
                    m_islands[i][j] = m_islands[i][j].translate({0 - first_island_min_x, 0 - first_island_min_y, (m_resize_counter - 1) * layer_height});
                }

                m_island_z_values.push_back((m_resize_counter - 1) * layer_height);

                continue;
            }

            // find the closest point to gravity_point out of all points in any of the polylines
            float final_x = max_x;
            float final_y = max_y;
            for (int j = 0, size_j = no_fit_polylines_clipped.length(); j < size_j; j++)
            {
                if (no_fit_polylines_clipped[j].empty())
                {
                    continue;
                }

                Point smallest_point = no_fit_polylines_clipped[j].closestPointTo(gravity_point);

                if (smallest_point.distance(gravity_point) < Point(final_x, final_y, 0).distance(gravity_point))
                {
                    final_x = smallest_point.x();
                    final_y = smallest_point.y();
                }
            }

            // m_islands[i] to that point
            float original_x = m_islands[i].getOutsidePolygons()[0][0].x();
            float original_y = m_islands[i].getOutsidePolygons()[0][0].y();
            offset.push_back(final_x - original_x);
            offset.push_back(final_y - original_y);
            m_offsets.push_back(offset);
            offset.clear();
            // move the whole island there polygon by polygon
            for (int j = 0, size_j = m_islands[i].length(); j < size_j; j++)
            {
                m_islands[i][j] = m_islands[i][j].translate({(final_x - original_x), (final_y - original_y), (m_resize_counter - 1) * layer_height});
            }

            m_island_z_values.push_back((m_resize_counter - 1) * layer_height);

        }
        m_layer_changes.push_back(m_islands.length()); // last layer change to end of m_islands

        emit statusUpdate(StatusUpdateStepType::kPostProcess, 100);
    }

    void SheetLaminationSlicer::writeGCode()
    {
        float layer_height = GSM->getGlobal()->setting<float>(Constants::ProfileSettings::Layer::kLayerHeight);
        float arm_precision = GSM->getGlobal()->setting<float>(Constants::ExperimentalSettings::SheetLaminationSlicing::kArmPrecision);
        float destination_offset_x = GSM->getGlobal()->setting<float>(Constants::ExperimentalSettings::SheetLaminationSlicing::kDestinationOffsetX);
        float destination_offset_y = GSM->getGlobal()->setting<float>(Constants::ExperimentalSettings::SheetLaminationSlicing::kDestinationOffsetY);
        float destination_offset_z = GSM->getGlobal()->setting<float>(Constants::ExperimentalSettings::SheetLaminationSlicing::kDestinationOffsetZ);

        Point destination_offset = Point(destination_offset_x, destination_offset_y, destination_offset_z);

        QTextStream stream(&m_temp_gcode_output_file);

        SheetLaminationWriter* laminator = dynamic_cast<SheetLaminationWriter*>(m_base.get());

        for (int i = 0, size_i = m_layer_changes.length() - 1; i < size_i; i++)
        {
            QVector<Point> origins;
            QVector<Point> destinations;
            QVector<float> destination_z_values;
            for (int j = m_layer_changes[i], size_j = m_layer_changes[i+1]; j < size_j; j++) {
                // have each island write its own dxf
                stream << laminator->writeIsland(m_islands[j], m_island_z_values[j]);

                // find where you can pick up the islands with a robotic arm
                // for this, we're going to add a function to PolygonList
                origins.push_back(m_islands[j].poleOfInaccessibility(arm_precision));
                destinations.push_back(origins.last() - Point(m_offsets[j][0], m_offsets[j][1], 0) + destination_offset);
                destination_z_values.push_back((m_layer_list[j] * layer_height) + destination_offset_z);
            }
            // 999 comment here
            stream << laminator->writeLayerOffsets(origins, destinations, m_island_z_values[m_layer_changes[i]], destination_z_values);
        }

        emit statusUpdate(StatusUpdateStepType::kGcodeGeneraton, 100);
        emit statusUpdate(StatusUpdateStepType::kGcodeParsing, 100);
    }
}
