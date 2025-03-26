#ifndef IMAGESLICER_H
#define IMAGESLICER_H

#include "geometry/mesh/mesh_base.h"
#include "threading/traditional_ast.h"

#undef emit
#include <vtkCellArray.h>
#include <vtkImageData.h>
#include <vtkImageStencilToImage.h>
#include <vtkLine.h>
#include <vtkNew.h>
#include <vtkPNGWriter.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataToImageStencil.h>
#include <vtkSmartPointer.h>
#define emit

// only needed for ITK, currently on hold
// #include "itkImage.h"
// #include "itkImageFileWriter.h"
// #include "itkPoint.h"
// #include "itkPolygonSpatialObject.h"
// #include "itkSpatialObjectToImageFilter.h"
// #include "itkPasteImageFilter.h"
// #include "itkImageDuplicator.h"
// #include "itkFlipImageFilter.h"

namespace ORNL {
/*!
 * \class ImageSlicer
 * \brief Implementation of SlicingThread for Image slices.
 */
class ImageSlicer : public TraditionalAST {
  public:
    ImageSlicer(QString gcodeLocation);

  protected:
    //! \brief Creates images from cross-sections.
    //! \param opt_data: optional sensor data
    void preProcess(nlohmann::json opt_data = nlohmann::json()) override;

    //! \brief NOP
    void postProcess(nlohmann::json opt_data = nlohmann::json()) override;

    //! \brief NOP
    void writeGCode() override;

  private:
    //! \brief Struct to group meshes with their bounds and ids
    //! \param m_mesh: Mesh from each part
    //! \param m_min: minimum of mesh bounding cube
    //! \param m_max: maximum of mesh bounding cube
    //! \param m_id: id based on sorting for build meshes or simply maximum for support meshes
    //! \param m_original_name: name of original file the mesh was loaded from
    struct MeshAndBounds {
        QSharedPointer<MeshBase> m_mesh;
        Point m_min, m_max;
        ushort m_id;
        QString m_original_name;
    };

    //! \brief Struct to group cross-sections and their assigned color. Color is determined from id.
    //! \param m_cross_section: cross-section of mesh
    //! \param m_color: color of cross-section
    struct PolygonListAndColor {
        PolygonListAndColor(PolygonList pL, ushort c) {
            m_cross_section = pL;
            m_color = c;
        }
        PolygonList m_cross_section;
        ushort m_color;
    };

    //! \brief Struct to group a slicing plane with a layer id
    //! m_layer: layer id
    //! m_slicing_plane: current slicing plane for given id
    struct SlicingPlaneWithLayer {
        SlicingPlaneWithLayer(int layer, Plane slicing_plane) {
            m_layer = layer;
            m_slicing_plane = slicing_plane;
        }
        int m_layer;
        Plane m_slicing_plane;
    };

    //! \brief Create a stencil image using VTK
    //! \param geometryAndColor: Vector of polygonlists from cross-sections and associated colors
    //! \param layer: Current layer needed for file output
    //! \param xResolution: X resolution of images
    //! \param yResolution: Y resolution of images
    //! \param gridWidth: width of entire build volume
    //! \param gridHeight: height of entire build volume
    void createImageStencilVTK(QVector<PolygonListAndColor> geometryAndColor, int layer, double xResolution,
                               double yResolution, int gridWidth, int gridHeight);

    //! \brief Create a stencil image using ITK (2D optimized version of VTK)
    //! \brief Currently, this function doesn't work quite right. For some reason, ITK will not render certain images
    //! correctly when creating the stencil. \brief I need to figure out why and test speed against VTK implementation.
    //! \param geometryAndColor: Vector of polygonlists from cross-sections and associated colors
    //! \param layer: Current layer needed for file output
    //! \param xResolution: X resolution of images
    //! \param yResolution: Y resolution of images
    //! \param gridWidth: width of entire build volume
    //! \param gridHeight: height of entire build volume
    void createImageStencilITK(QVector<PolygonListAndColor> geometryAndColor, int layer, double xResolution,
                               double yResolution, int gridWidth, int gridHeight);

    //!\brief Number of digits for image file name to allow sequencing of layers
    int m_total_digits = 7;
};
} // namespace ORNL

#endif // IMAGESLICER_H
