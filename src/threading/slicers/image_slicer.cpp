// Main Module
#include "threading/slicers/image_slicer.h"

// Qt
#include <QSharedPointer>
#include <QDir>

// Local
#include "managers/settings/settings_manager.h"
#include "managers/session_manager.h"
#include "slicing/slicing_utilities.h"
#include "utilities/mathutils.h"
#include "cross_section/cross_section.h"

namespace ORNL {

    ImageSlicer::ImageSlicer(QString gcodeLocation) : TraditionalAST(gcodeLocation, true) {
    }

    void ImageSlicer::preProcess(nlohmann::json opt_data)
    {
        // QElapsedTimer timer;
        // timer.start();

        QVector<QSharedPointer<Part>> parts = SlicingUtilities::GetPartsByType(CSM->parts(), MeshType::kBuild);
        QVector<QSharedPointer<Part>> moreParts = SlicingUtilities::GetPartsByType(CSM->parts(), MeshType::kSupport);
        parts += moreParts;

        Distance layerHeight = GSM->getGlobal()->setting<Distance>(Constants::ProfileSettings::Layer::kLayerHeight);
        Point globalMax;

        QVector<SlicingPlaneWithLayer> slicing_planes;
        Plane lowestSlicingPlane(Point(0, 0, 0), QVector3D(0, 0, 1));

        QVector<MeshAndBounds> allMeshes;
        QVector<MeshAndBounds> supportMeshes;

        QMatrix4x4 center;
        QVector3D shift(QVector3D(GSM->getGlobal()->setting<double>(Constants::PrinterSettings::Dimensions::kXOffset), GSM->getGlobal()->setting<double>(Constants::PrinterSettings::Dimensions::kYOffset), 0));
        center.translate(shift);

        for(QSharedPointer<Part> part : parts)
        {
            MeshAndBounds currentMesh;
            Plane slicing_plane;
            Point mesh_min, mesh_max;

            QSharedPointer<MeshBase> mesh;
            auto closed_mesh = dynamic_cast<ClosedMesh*>(part->rootMesh().get());
            if(closed_mesh != nullptr)
                mesh = QSharedPointer<ClosedMesh>::create(ClosedMesh(*closed_mesh));
            else
                mesh = QSharedPointer<OpenMesh>::create(OpenMesh(*dynamic_cast<OpenMesh*>(part->rootMesh().get())));

            QMatrix4x4 trans = mesh->transformation();
            trans.translate(QVector3D(GSM->getGlobal()->setting<double>(Constants::PrinterSettings::Dimensions::kXOffset), GSM->getGlobal()->setting<double>(Constants::PrinterSettings::Dimensions::kYOffset), 0));
            mesh->setTransformation(trans);

            currentMesh.m_mesh = mesh;

            std::tie(slicing_plane, mesh_min, mesh_max) = SlicingUtilities::GetDefaultSlicingAxis(GSM->getGlobal(), mesh, nullptr);

            if(slicing_plane.point().z() < lowestSlicingPlane.point().z())
                lowestSlicingPlane = slicing_plane;

            currentMesh.m_min = mesh->min();
            currentMesh.m_max = mesh->max();

            QFileInfo fi(part->sourceFilePath());
            currentMesh.m_original_name = fi.completeBaseName();

            if(mesh_max.z() > globalMax.z())
                globalMax = mesh_max;

            if(mesh->type() == MeshType::kSupport)
            {
                currentMesh.m_id = 65535;
                supportMeshes.push_back(currentMesh);
            }
            else
                allMeshes.push_back(currentMesh);
        }

        //sort meshes to determine id: first by z, then y, then x
        std::sort(allMeshes.begin(), allMeshes.end(), [](const MeshAndBounds& a, const MeshAndBounds& b)
        {
            if (std::rint(a.m_min.z()) != std::rint(b.m_min.z())) {
                return std::rint(a.m_min.z()) < std::rint(b.m_min.z());
            }
            if (std::rint(a.m_min.y()) != std::rint(b.m_min.y())) {
                return std::rint(a.m_min.y()) < std::rint(b.m_min.y());
            }
            return std::rint(a.m_min.x()) < std::rint(b.m_min.x());
        });

        //Ids are now conveniently mesh location in vector
        for(int i = 0; i < allMeshes.size(); ++i)
            allMeshes[i].m_id = i + 1;

        //include support meshes, id was previously set to static value
        allMeshes += supportMeshes;

        //reserve enough planes for tallest object or selected planes
        int maxSlices = (globalMax.z() - lowestSlicingPlane.point().z()) / layerHeight();

        QVector<int> layers;
        QSharedPointer<SettingsBase> consoleSettings = GSM->getConsoleSettings();
        if(!consoleSettings->empty())
        {
            if(consoleSettings->contains(Constants::ConsoleOptionStrings::kSingleSliceHeight))
            {
                QVector<double> heights = QVector<double>::fromStdVector(consoleSettings->setting<std::vector<double>>(Constants::ConsoleOptionStrings::kSingleSliceHeight));
                for(double height : heights)
                {
                    int layer = qRound(height / layerHeight()) - 1;
                    if(layer < 0)
                        layer = 0;
                    layers.push_back(layer);
                }
            }

            if(consoleSettings->contains(Constants::ConsoleOptionStrings::kSingleSliceLayerNumber))
            {
                layers = QVector<int>::fromStdVector(consoleSettings->setting<std::vector<int>>(Constants::ConsoleOptionStrings::kSingleSliceLayerNumber));
            }
        }
        if(layers.size() > 0)
            slicing_planes.reserve(layers.size());
        else
            slicing_planes.reserve(maxSlices);

        for(int i = 0; i < maxSlices; ++i)
        {
            SlicingUtilities::ShiftSlicingPlane(GSM->getGlobal(), lowestSlicingPlane, layerHeight, nullptr);
            Plane next_plane = lowestSlicingPlane;
            if(layers.size() > 0)
            {
                if(i == layers[0])
                {
                    slicing_planes.push_back(SlicingPlaneWithLayer(i, next_plane));
                    layers.pop_front();
                    if(layers.size() == 0)
                        break;
                }
            }
            else
                slicing_planes.push_back(SlicingPlaneWithLayer(i, next_plane));
        }

        emit statusUpdate(StatusUpdateStepType::kPreProcess, 100);

        double xResolution = GSM->getGlobal()->setting<double>(Constants::ExperimentalSettings::ImageResolution::kImageResolutionX);
        double yResolution = GSM->getGlobal()->setting<double>(Constants::ExperimentalSettings::ImageResolution::kImageResolutionY);
        int volumeXDim = std::ceil((GSM->getGlobal()->setting<double>(Constants::PrinterSettings::Dimensions::kXMax) -
                                    GSM->getGlobal()->setting<double>(Constants::PrinterSettings::Dimensions::kXMin)) / xResolution) + 1;

        int volumeYDim = std::ceil((GSM->getGlobal()->setting<double>(Constants::PrinterSettings::Dimensions::kYMax) -
                                    GSM->getGlobal()->setting<double>(Constants::PrinterSettings::Dimensions::kYMin)) / yResolution) + 1;

        //let omp loose on the cross-sectioning and image generation
        #pragma omp parallel for
        for(int i = 0; i < slicing_planes.size(); ++i)
        {
            Plane currentPlane = slicing_planes[i].m_slicing_plane;
            Point shift_amount = Point(0, 0, 0);
            QVector3D average_normal;

            QVector<PolygonListAndColor> currentCrossSections;
            currentCrossSections.reserve(allMeshes.size());

            for(int j = 0; j < allMeshes.size(); ++j)
            {
                if(allMeshes[j].m_mesh->max().z() >= currentPlane.point().z())
                {
                    PolygonList result = CrossSection::doCrossSection(allMeshes[j].m_mesh, currentPlane, shift_amount, average_normal, GSM->getGlobal());
                    if(result.size() > 0)
                        currentCrossSections.push_back(PolygonListAndColor(result, allMeshes[j].m_id));
                    //currentCrossSections.push_back(PolygonListAndColor(result, 65535));//PolygonListAndColor(result, allMeshes[j].m_id));//PolygonListAndColor(result, (j + 1) * 10000));
                }
            }
            createImageStencilVTK(currentCrossSections, slicing_planes[i].m_layer, xResolution, yResolution, volumeXDim, volumeYDim);
            //createImageStencilITK(currentCrossSections, i, xResolution, yResolution, volumeXDim, volumeYDim);
        }

        //create companion json file
        fifojson originalName;
        int i = 0;
        for(MeshAndBounds mesh : allMeshes)
        {
            if(mesh.m_id != 65535)
                originalName[QString::number(mesh.m_id).toStdString()] = mesh.m_original_name;
            else
            {
                originalName[QString::number(mesh.m_id).toStdString()][QString::number(i).toStdString()] = mesh.m_original_name;
                ++i;
            }
        }

        QFile file(m_temp_gcode_dir.absoluteFilePath("idFileLinks.dat"));
        file.open(QIODevice::WriteOnly);
        file.write(originalName.dump().c_str());
        file.close();

        //            QString filename = "C:\\Code\\vtkexample\\totaltime.txt";
        //            QFile file(filename);
        //            if (file.open(QIODevice::ReadWrite)) {
        //                QTextStream stream(&file);
        //                stream << QString::number(timer.elapsed());
        //            }
        //            file.close();
    }

    void ImageSlicer::createImageStencilVTK(QVector<PolygonListAndColor> geometryAndColor, int layer, double xResolution, double yResolution, int gridWidth, int gridHeight)
    {
        vtkNew<vtkPolyData> blankData;

        vtkNew<vtkPolyDataToImageStencil> blankStenc;
        blankStenc->SetInputData(blankData);
        blankStenc->SetTolerance(0);
        blankStenc->SetOutputWholeExtent(0, gridWidth, 0, gridHeight, 0, 0);
        blankStenc->Update();

        vtkNew<vtkImageStencilToImage> blankImage;
        blankImage->SetInputConnection(blankStenc->GetOutputPort());
        blankImage->SetOutputScalarTypeToUnsignedShort();
        blankImage->SetOutsideValue(0);
        blankImage->Update();

        vtkImageData* fullImage = blankImage->GetOutput();

        for(PolygonListAndColor polyListAndColor : geometryAndColor)
        {
            vtkNew<vtkPoints> points;
            vtkNew<vtkCellArray> lines;

            Point minPt = Point(std::rint(polyListAndColor.m_cross_section.min().x() / xResolution), std::rint(polyListAndColor.m_cross_section.min().y() / yResolution));
            Point maxPt = Point(std::rint(polyListAndColor.m_cross_section.max().x() / xResolution), std::rint(polyListAndColor.m_cross_section.max().y() / yResolution));

            for (Polygon polygon : polyListAndColor.m_cross_section)
            {
                for (int i = 0; i < polygon.size(); ++i)
                {
                    Point p1 = polygon[i];
                    Point p2 = polygon[(i + 1) % polygon.size()];

                    vtkIdType start = points->InsertNextPoint(p1.x() / xResolution - minPt.x(), p1.y() / yResolution - minPt.y(), 0);
                    vtkIdType end = points->InsertNextPoint(p2.x() / xResolution - minPt.x(), p2.y() / yResolution - minPt.y(), 0);

                    vtkNew<vtkLine> line;
                    line->GetPointIds()->SetId(0, start);
                    line->GetPointIds()->SetId(1, end);

                    lines->InsertNextCell(line);
                }
            }

            vtkNew<vtkPolyData> polyData;
            polyData->SetPoints(points);
            polyData->SetLines(lines);

            vtkNew<vtkPolyDataToImageStencil> pol2stenc;
            pol2stenc->SetInputData(polyData);
            pol2stenc->SetTolerance(0);
            pol2stenc->SetOutputWholeExtent(0, maxPt.x() - minPt.x(), 0, maxPt.y() - minPt.y(), 0, 0);
            pol2stenc->Update();

            vtkNew<vtkImageStencilToImage> imageStencilToImage;
            imageStencilToImage->SetInputConnection(pol2stenc->GetOutputPort());
            imageStencilToImage->SetOutputScalarTypeToUnsignedShort();
            imageStencilToImage->SetInsideValue(polyListAndColor.m_color);
            imageStencilToImage->SetOutsideValue(0);
            imageStencilToImage->Update();

            for(int x = minPt.x(); x < maxPt.x(); ++x)
            {
                for(int y = minPt.y(); y < maxPt.y(); ++y)
                {
                    double val = imageStencilToImage->GetOutput()->GetScalarComponentAsDouble(x - minPt.x(), y - minPt.y(), 0, 0);
                    if(val > 0)
                        fullImage->SetScalarComponentFromDouble(x, y, 0, 0, val);
                }
            }
        }

        vtkNew<vtkPNGWriter> imageWriter;
        imageWriter->SetFileName((m_temp_gcode_dir.absoluteFilePath(QStringLiteral("%1").arg(layer, m_total_digits, 10, QLatin1Char('0'))) + ".png").toStdString().c_str());
        imageWriter->SetInputData(fullImage);
        imageWriter->Write();
    }

    void ImageSlicer::createImageStencilITK(QVector<PolygonListAndColor> geometryAndColor, int layer, double xResolution, double yResolution, int gridWidth, int gridHeight)
    {
//        using pixelType = unsigned short;
//        const int dimension = 2;
//        using ImageType = itk::Image<pixelType, dimension>;

//        ImageType::RegionType region;
////        ImageType::IndexType  start;
////        start[0] = 0;
////        start[1] = 0;

//        ImageType::SizeType size;
//        size[0] = gridWidth;
//        size[1] = gridHeight;

//        region.SetSize(size);
////        region.SetIndex(start);

//        auto fullImage = ImageType::New();
//        fullImage->SetRegions(region);
//        fullImage->Allocate();
//        fullImage->FillBuffer(0);

//        using PolygonType = itk::PolygonSpatialObject<dimension>;
//        using SpatialObjectToImageFilterType = itk::SpatialObjectToImageFilter<PolygonType, ImageType>;

//        using FilterType = itk::PasteImageFilter<ImageType, ImageType>;



////        int i = 0;
////        std::vector<FilterType*> pasteFilters;
////        std::vector<ImageType*> locals;
//        for(PolygonListAndColor polyListAndColor : geometryAndColor)
//        {
//            Point minPt = Point(std::rint(polyListAndColor.crossSection.min().x() / xResolution), std::rint((polyListAndColor.crossSection.min().y() / yResolution)));
//            Point maxPt = Point(std::rint(polyListAndColor.crossSection.max().x() / xResolution), std::rint((polyListAndColor.crossSection.max().y() / yResolution)));

//            auto fullPolygon = PolygonType::New();
//            typename PolygonType::PointType point;
//            typename PolygonType::PolygonPointType polygonPoint;


//            polyListAndColor.crossSection.removeAt(3);
//            polyListAndColor.crossSection.removeAt(3);
////            polyListAndColor.crossSection.removeAt(1);

//            int count = 0;
//            for (Polygon polygon : polyListAndColor.crossSection)
//            {
//                fullPolygon->GetPoints().reserve(fullPolygon->GetPoints().size() + polygon.size());
//                Point p1 = polygon[0];
//                Point p2 = polygon[(polygon.size() + 1) % polygon.size()];
//                qDebug() << "size" << polygon.size();// << (polygon.size() + 1) % polygon.size();
//                //qDebug() << p1.x() << p1.y() << p2.x() << p2.y();
//                if(count > 0)
//                {
//                    std::reverse(polygon.begin(), polygon.end());
//                }
//                for (int i = 0; i < polygon.size() + 1; ++i)
//                {
//                    //qDebug() << "iterator" << i << i % polygon.size();
//                    Point p = polygon[i % polygon.size()];
//                    point[0] = p.x() / xResolution - minPt.x();
//                    point[1] = p.y() / yResolution - minPt.y();
//                    qDebug() << point[0] << point[1];
//                    polygonPoint.SetPositionInObjectSpace(point);
//                    fullPolygon->GetPoints().push_back(polygonPoint);
//                }
//                count++;
//            }

//            fullPolygon->SetIsClosed(true);
//            fullPolygon->Update();

//            auto localImage = SpatialObjectToImageFilterType::New();
//            ImageType::SizeType localSize;
//            localSize[0] = maxPt.x() - minPt.x();
//            localSize[1] = maxPt.y() - minPt.y();
//            localImage->SetSize(localSize);
//            localImage->SetInput(fullPolygon);
//            localImage->SetInsideValue(polyListAndColor.color);
//            localImage->SetOutsideValue(0);
//            localImage->Update();

//            auto localImageOut = localImage->GetOutput();
//            ImageType::IndexType index;
//            index[0] = minPt.x();
//            index[1] = minPt.y();

//            auto copyImage = FilterType::New();
//            copyImage->SetSourceImage(localImageOut);
//            copyImage->SetSourceRegion(localImageOut->GetLargestPossibleRegion());
//            copyImage->SetDestinationIndex(index);
//            copyImage->SetDestinationImage(fullImage);
//            copyImage->Update();

////            locals.push_back(localImageOut);
////            pasteFilters.push_back(copyImage);


// //           QString imageName = "C:\\Code\\itkexample\\test_" + QString::number(i) + ".png";
// //           itk::WriteImage(copyImage->GetOutput(), imageName.toStdString());
//            fullImage = copyImage->GetOutput();
// //           ++i;
////            if(i == 2)
////            {
////                QString imageName = "C:\\Code\\itkexample\\test_" + QString::number(layer) + ".png";
////                itk::WriteImage(copyImage->GetOutput(), imageName.toStdString());
////            }
//        }

//        using FlipImageFilterType = itk::FlipImageFilter<ImageType>;
//        auto flipFilter = FlipImageFilterType::New();
//        flipFilter->SetInput(fullImage);
//        FlipImageFilterType::FlipAxesArrayType flipAxes;
//        flipAxes[0] = false;
//        flipAxes[1] = true;
//        flipFilter->SetFlipAxes(flipAxes);

//        //QString imageName = "C:\\Code\\itkexample\\test_" + QString::number(layer) + ".png";
//        std::string val = std::to_string(layer);
//        std::string filename = "C:\\Code\\itkexample\\" + std::string(m_total_digits - std::min(m_total_digits, (int)val.length()), '0') + val + ".png";
//        itk::WriteImage(flipFilter->GetOutput(), filename);
//        //itk::WriteImage(fullImage, imageName.toStdString());

    }

    void ImageSlicer::postProcess(nlohmann::json opt_data)
    {
        emit statusUpdate(StatusUpdateStepType::kPostProcess, 100);
    }

    void ImageSlicer::writeGCode()
    {
        emit statusUpdate(StatusUpdateStepType::kGcodeGeneraton, 100);
    }
}
