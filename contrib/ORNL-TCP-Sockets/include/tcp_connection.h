#ifndef TCPCONNECTION_H
#define TCPCONNECTION_H

#include <QObject>
#include <QThread>
#include <QTcpSocket>
#include <QAbstractSocket>
#include <QDataStream>

namespace ORNL
{
    //! \class TCPConnection
    //! \brief This class provides functions to connect to a server/ establish a connection over TCP
    //! \note This class runs in its own thread
    class TCPConnection : public QObject
    {
        Q_OBJECT
    public:
        //! \brief Constructor
        //! \param parent the parent
        TCPConnection(QObject* parent = nullptr);

        //! \brief Destructor that closes the TCP connection
        ~TCPConnection();

        //! \brief gets the ip of the connection
        //! \return the ip as a string
        QString getIP();

        //! \brief gets the port of the connection
        //! \return the port as an int
        quint16 getPort();

        //! \brief sets up a new connection to a host
        //! \param host the address to connect to
        //! \param port the port to connect to
        void setupNewAsync(QString host = "localhost", quint16 port = 9999);

        //! \brief sets up a connection to a host using a socket descriptor
        //! \note a socket descriptor is created when using the TCPServer
        //! \param socket_descriptor the connection id from the TCPServer
        void setupExistingAsync(qintptr socket_descriptor);

        //! \brief sets the timeout for this socket
        //! \param timeout the time in milliseconds
        void setConnectionTimeout(size_t timeout);

        //! \brief gets the timeout for this socket
        //! \return the timeout in milliseconds
        size_t getConnectionTimeout();

    public slots:
        //! \brief sends data over the connection
        //! \param msg the string data to send
        void write(const QString& msg);

        //! \brief closes the connection
        void close();

        //! \brief converts an incoming connection into a TCPConnection
        //! \param socket_descriptor the index of the incoming connection
        void setupExisting(qintptr socket_descriptor);

        //! \brief sets up a new connection to a host
        //! \param host the address to connect to
        //! \param port the port to connect to
        void setupNew(QString host = "localhost", quint16 port = 9999);

    signals:
        //! \brief signals when the connection has been established
        void connected();

        //! \brief signals when the connection has been terminated
        void disconnected();

        //! \brief signals when an error has occurred
        //! \param socket_error the error
        void error(QAbstractSocket::SocketError socket_error);

        //! \brief signals when a new message has arrived
        //! \param msg the new message
        void newMessage(const QString& msg);

        //! \brief signal that the connection request has timeout
        void timeout();

        //! \brief makes internal connection
        //! \note DO NOT CALL EXTERNALLY
        //! \param host the host
        //! \param port the port
        void setupInternal(QString host, quint16 port);

        //! \brief makes internal connection
        //! \note DO NOT CALL EXTERNALLY
        //! \param socket_descriptor
        void setupExistingInternal(qintptr socket_descriptor);

        //! \brief closes internal socket
        //! \note DO NOT CALL EXTERNALLY
        void closeInternalSocket();

    private:
        //! \brief runs setup
        void init();

        //! \brief the internal thread this connection runs on
        QThread m_thread;

        //! \brief the internal TCP connection
        QTcpSocket* m_socket;

        //! \brief default timeout is 15 seconds
        size_t m_connection_timeout = 15000;

    private slots:
        //! \brief handles reading new messages when they arrive
        void handleNewMessages();
    };
}


#endif // TCPCONNECTION_H
