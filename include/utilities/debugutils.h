#ifndef DEBUGUTILS_H
#define DEBUGUTILS_H

// Qt
#include <QSharedPointer>
#include <QString>

// Local
#include "geometry/mesh/closed_mesh.h"
#include "part/part.h"

namespace ORNL {
    namespace DebugUtils {
        //! \brief Load a part from a filename.
        //! \note This is intended as a debug utility and will lock the thread it runs on until the model is loaded.
        QSharedPointer<Part> getPartFromFile(QString filename, QString name, MeshType mt = MeshType::kBuild);

        //! \brief Load a mesh from a filename.
        //! \note This is intended as a debug utility and will lock the thread it runs on until the model is loaded.
        QSharedPointer<ClosedMesh> getMeshFromFile(QString filename, QString name, MeshType mt = MeshType::kBuild);
    }
}
#endif  // DEBUGUTILS_H
