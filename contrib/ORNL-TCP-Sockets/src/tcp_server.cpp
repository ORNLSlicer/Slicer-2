#include "include/tcp_server.h"

#include <QDataStream>

namespace ORNL
{
    TCPServer::TCPServer(QObject *parent) : QTcpServer(parent)
    {
        // Internal connection to handle start call from another thread
        connect(this, &TCPServer::startInternal, this, &TCPServer::startServer);

        this->moveToThread(&m_thread);
        m_thread.start();
    }

    TCPServer::~TCPServer()
    {
        for(auto& connection : m_connections)
        {
            emit clientDisconnected(connection);
            connection->deleteLater();
        }

        m_connections.clear();
        m_thread.terminate();
        m_thread.wait();
    }

    void TCPServer::startAsync(quint16 port)
    {
        emit startInternal(port);
    }

    bool TCPServer::startServer(quint16 port)
    {
        if(!this->listen(QHostAddress::Any, port))
        {
            qWarning() << "Failed to start server for port: " << port;
            return false;
        }
        else
        {
            //qInfo() << "Server started at port: " << port;
            return true;
        }
    }

    void TCPServer::close()
    {
        for(auto& socket : m_connections)
        {
            socket->close();
            emit clientDisconnected(socket);
            delete socket;
        }

        m_connections.clear();
    }

    TCPConnection *TCPServer::getFirstConnection()
    {
        if(!m_connections.empty())
            return m_connections.first();
        else
            return nullptr;
    }

    void TCPServer::incomingConnection(qintptr socket_descriptor)
    {
        //qInfo() << "New TCP Connection";
        TCPConnection *new_connection = new TCPConnection();

        m_connections.push_back(new_connection);

        connect(new_connection, &TCPConnection::disconnected, this, [this, new_connection] ()
        {
            m_connections.removeOne(new_connection);
            emit clientDisconnected(new_connection);
        });
        connect(new_connection, &TCPConnection::connected, this, [this, new_connection] ()
        {
            emit newClient(new_connection);
        });

        new_connection->setupExistingAsync(socket_descriptor);
    }
}

