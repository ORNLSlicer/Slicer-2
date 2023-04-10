// This class is currently disabled because it is unused and requires a replacement HTTP library #if 0
#if 0
#ifndef HTTP_SERVER_H
#define HTTP_SERVER_H

//Prevent name class with boost
#undef DELETE

#include <QObject>
#include <QHttpServer>
#include <QTimer>

#include "utilities/qt_json_conversion.h"

namespace ORNL
{
    class HttpServer : public QObject
    {
        Q_OBJECT
        public:
            //! \brief Default Constructor
            HttpServer();

            //! \brief Set initial config when program starts
            //! \param path: Path to config file
            void setConfig(QString path);

        signals:
            //! \brief Signals remote connectivity window with status when restarting
            //! \param success: whether or not restart was successful
            void restartSuccess(bool success);

        public slots:
            //! \brief Set config when altered from remote connectivity window
            //! \param config: json configuration to load
            void setConfig(fifojson config);

        private:
            //! \brief Creates server based on json configuration
            //! \param endpoints: json configuration to load
            void createServer(fifojson endpoints);

            //! \brief Actual server object
            QHttpServer m_server;

            //! \brief tracker for Ids when importing an STL
            //! API must first provide a Uuid followed by import with Uuid provided
            QHash<QString, QString> m_id_name_tracker;
    };
}

#endif // HTTP_SERVER_H
#endif
