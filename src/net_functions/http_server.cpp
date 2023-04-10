// This class is currently disabled because it is unused and requires a replacement HTTP library
#if 0
#include "net_functions/http_server.h"

#include <QHttpServer>
#include <QMetaEnum>
#include <QUuid>
#include <QStandardPaths>
#include <QDir>

#include <managers/session_manager.h>

namespace ORNL
{
    HttpServer::HttpServer()
    {
        //NOP
    }

    void HttpServer::setConfig(fifojson config)
    {
        QVector<QTcpServer*> servers = m_server.servers();
        if(servers.size() > 0)
        {
            for(QTcpServer* s : servers)
                s->close();
        }

        createServer(config);
    }

    void HttpServer::setConfig(QString config)
    {
        QFile file(config);
        file.open(QIODevice::ReadOnly);
        QString settings = file.readAll();
        fifojson j = fifojson::parse(settings.toStdString());
        createServer(j);
    }

    void HttpServer::createServer(fifojson endpoints)
    {
        QList<QString> physicalEndPoints;
        for (auto& el : endpoints.at("endpoints").items())
        {
            for(auto& endpoint : el.value().items())
            {
                physicalEndPoints.push_back(endpoint.value()["physical_end"]);
            }
        }

        for (auto& el : endpoints.at("endpoints").items())
        {
            for(auto& endpoint : el.value().items())
            {
                QString key = QString::fromStdString(endpoint.key());

                QVector<QHttpServerRequest::Method> endMethods;
                QVector<QString> endPointTypes = endpoint.value()["end_type"];
                for(QString endType : endPointTypes)
                {
                    auto&& metaEnum = QMetaEnum::fromType<QHttpServerRequest::Method>();
                    endMethods.push_back(static_cast<QHttpServerRequest::Method>(metaEnum.keyToValue(endType.toStdString().c_str())));
                }
                QString physicalEndPoint = endpoint.value()["physical_end"];

                if(key == "status")
                {
                    for(QHttpServerRequest::Method method : endMethods)
                    {
                        m_server.route(physicalEndPoint, method, []() {
                            QHttpServerResponse ok_response("");
                            return ok_response;
                        });
                    }
                }
                else if(key == "api")
                {
                    for(QHttpServerRequest::Method method : endMethods)
                    {
                        m_server.route(physicalEndPoint, method, [physicalEndPoints] () {
                            return physicalEndPoints.join('\n');
                        });
                    }
                }
                else if(key == "generateUploadId")
                {
                    for(QHttpServerRequest::Method method : endMethods)
                    {
                        m_server.route(physicalEndPoint, method, [this] (QString name) {
                            QString id = QUuid::createUuid().toString(QUuid::WithoutBraces);
                            //m_previous_ids.push_back(id);

                            qDebug() << m_id_name_tracker.size();
                            m_id_name_tracker.insert(id, name);

                            QHttpServerResponse ok_response(id);
                            return ok_response;
                        });
                    }
                }
                else if(key == "uploadSTL")
                {
                    for(QHttpServerRequest::Method method : endMethods)
                    {
                        m_server.route(physicalEndPoint, method, [this] (QString id, const QHttpServerRequest &request) {

                            if(m_id_name_tracker.contains(id))
                            {
                                QString appPathStr = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
                                QDir appPath(appPathStr);
                                QString tempSTL = appPath.filePath(m_id_name_tracker[id]) + ".stl";
                                QFile stl(tempSTL);
                                stl.open(QIODevice::WriteOnly | QIODevice::Truncate);
                                stl.write(request.body());
                                stl.close();

                                CSM->loadModel(tempSTL, false);
                                m_id_name_tracker.remove(id);

                                QHttpServerResponse ok_response("");
                                return ok_response;
                            }
                            else
                            {
                                QHttpServerResponse bad_response("text/plain", "ID not found", QHttpServerResponder::StatusCode::BadRequest);
                                return bad_response;
                            }
                        });
                    }
                }
            }
        }

        bool success = true;
        const auto port = m_server.listen(QHostAddress::Any, endpoints["port"]);
        if (port == -1) {
            qDebug() << "Could not run on http://127.0.0.1";
            success = false;
        }

        emit restartSuccess(success);
    }
}
#endif
