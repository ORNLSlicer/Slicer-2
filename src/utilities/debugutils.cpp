#include "utilities/debugutils.h"

// Qt
#include <QObject>
#include <QEventLoop>

// Local
#include "threading/mesh_loader.h"

namespace ORNL {
    namespace DebugUtils {
        QSharedPointer<Part> getPartFromFile(QString filename, QString name, MeshType mt) {
            QSharedPointer<ClosedMesh> m = getMeshFromFile(filename, name, mt);
            QSharedPointer<Part> p = QSharedPointer<Part>::create(m);

            return p;
        }

        QSharedPointer<ClosedMesh> getMeshFromFile(QString filename, QString name, MeshType mt) {
//            // Some C here to get a void pointer of the model.
//            FILE* fptr = fopen(filename.toUtf8(), "rb");

//            fseek(fptr, 0L, SEEK_END);
//            size_t fsize = ftell(fptr);
//            fseek(fptr, 0L, SEEK_SET);

//            void* data = malloc(fsize);
//            if (data == nullptr) return nullptr;

//            int readres = fread(data, 1, fsize, fptr);
//            if (readres != fsize) return nullptr;

//            fclose(fptr);

//            // Wait on this thread for loader to finish.
//            MeshLoader* loader = new MeshLoader(data, fsize, filename, name, mt);
//            QEventLoop loop;

//            QObject::connect(loader, &MeshLoader::finished, &loop, &QEventLoop::quit);
//            loader->start();
//            loop.exec();

//            QSharedPointer<ClosedMesh> m = loader->getMesh();
//            delete loader;
//            free(data);
            QSharedPointer<ClosedMesh> m;
            return m;
        }
    }
}
