#ifndef SKELETONSLICER_H
#define SKELETONSLICER_H

#include <threading/traditional_ast.h>
#include "step/layer/layer.h"

namespace ORNL
{
    //! \class SkeletonSlicer
    //! \brief A slicer that works with open meshes. This slicer prints only open contours. If a polygon results from cross-sectioning, it
    //!        will not be filled.
    //! \note this is a traditional slicer
    //! \note all paths in this slicer are treated as kPerimeter
    //! \note this class is backed by CGAL cross-sectioning
    //! \note no path/ island order optimization occurs in this class
    class SkeletonSlicer : public TraditionalAST
    {
        public:

            //! \brief Contructor
            //! \param gcodeLocation location to write output
            SkeletonSlicer(QString gcodeLocation);

        protected:
            //! \brief creates cross-sections and SkeletonLayers
            //! \param opt_data optional data
            void preProcess(nlohmann::json opt_data);

            //! \brief currently does nothing
            //! \param opt_data optional data
            void postProcess(nlohmann::json opt_data);

            //! \brief writes pathing to file
            void writeGCode();

        private:
            //! \typedef SkeletonLayer
            //! \brief a simple list of paths that make up a layer
            typedef QVector<Path> SkeletonLayer;

            //! \brief a list of all layers
            QVector<SkeletonLayer> m_skeleton_layers;

            //! \brief the last point visisted
            Point m_last_pos = Point(0,0,0);
    };
}

#endif // SKELETONSLICER_H
