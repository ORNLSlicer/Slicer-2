#ifndef TCPSERVER_H
#define TCPSERVER_H

#include <QThread>
#include <QTcpServer>
#include "tcp_connection.h"

namespace ORNL
{
    //! \class TCPServer
    //! \brief The class listens and establishes TCP connections
    //! \note This class runs in its own thread
    class TCPServer : public QTcpServer
    {
        Q_OBJECT
    public:
        //! \brief Contructor
        //! \param parent the parent
        TCPServer(QObject *parent = nullptr);

        //! \brief Destructor that closes all connections
        ~TCPServer();

        //! \brief queues a request in the server thread to start the server
        //! \param port the port to start the server on
        void startAsync(quint16 port = 12345);

    public slots:
        //! \brief starts the server on a given port
        //! \param port the port to start on (defaults to 12345)
        //! \return is the server could be started
        bool startServer(quint16 port = 12345);

        //! \brief closes the server
        //! \note this will automatically disconnect clients
        void close();

        //! \brief gets the first connection to this server
        //! \return the first connection or nullptr is none exist
        ORNL::TCPConnection* getFirstConnection();

    signals:
        //! \brief emitted when the server is connected to new client
        //! \param new_connection the socket that was connected
        void newClient(ORNL::TCPConnection* new_connection);

        //! \brief emitted when the server disconnects from a client
        //! \param connection the connection that was dropped
        void clientDisconnected(ORNL::TCPConnection* connection);

        //! \brief internal signal to start the server from another thread
        //! \param port the port to run the server on
        void startInternal(quint16 port);

    private slots:
        //! \brief triggered automatically when a new pending connection is available
        //! \param socket_descriptor the index of the socket
        //! \note overrides base class
        void incomingConnection(qintptr socket_descriptor);

    private:
        //! \brief the internal thread
        QThread m_thread;

        //! \brief the connections
        QVector<TCPConnection*> m_connections;
    };
}


#endif // TCPSERVER_H
