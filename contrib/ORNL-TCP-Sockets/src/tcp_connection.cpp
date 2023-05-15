#include "include/tcp_connection.h"

#include <QHostAddress>

namespace ORNL
{
    TCPConnection::TCPConnection(QObject* parent) : QObject(parent)
    {
        connect(this, &TCPConnection::setupInternal, this, &TCPConnection::setupNew);
        connect(this, &TCPConnection::setupExistingInternal, this, &TCPConnection::setupExisting);
        connect(this, &TCPConnection::closeInternalSocket, this, &TCPConnection::close);
        this->moveToThread(&m_thread);
        m_thread.start();
    }

    TCPConnection::~TCPConnection()
    {
        emit closeInternalSocket();
        m_socket->deleteLater();
        m_thread.terminate();
        m_thread.wait();
    }

    void TCPConnection::setupExisting(qintptr socket_descriptor)
    {
        init();

        if (!m_socket->setSocketDescriptor(socket_descriptor))
            return;

        if(m_socket->isOpen())
            emit connected();
    }

    void TCPConnection::setupNew(QString host, quint16 port)
    {
        init();

        m_socket->connectToHost(host, port);

        if (m_socket->waitForConnected(m_connection_timeout))
            qInfo() << "Connected to server at " << host << ":" << QString::number(port);
        else
        {
            qWarning() << "Could not connect to server";
            delete m_socket;
            m_socket = nullptr;
            emit timeout();
        }
    }

    void ORNL::TCPConnection::init()
    {
        m_socket = new QTcpSocket();

        connect(m_socket, &QTcpSocket::connected, this, [this](){emit connected();});
        connect(m_socket, &QTcpSocket::disconnected, this, [this](){emit disconnected();});
        connect(m_socket, &QTcpSocket::errorOccurred, this, [this](QAbstractSocket::SocketError e){emit error(e);});
        connect(m_socket, &QTcpSocket::readyRead, this, &TCPConnection::handleNewMessages);
    }

    void TCPConnection::write(const QString& msg)
    {
        if(m_socket == nullptr)
        {
            qCritical() << "Thread is not initialized yet!";
        }else
        {
            QByteArray byte_array = msg.toUtf8();
            qint64 bytes_written = m_socket->write(byte_array);
            qDebug() << bytes_written << "\t" << byte_array.size();
            if (bytes_written == -1)
                qCritical() << "An error occured during socket write.";
        }
    }

    void TCPConnection::close()
    {
        m_socket->close();
    }

    void TCPConnection::handleNewMessages()
    {
        qint64 num = m_socket->bytesAvailable();
        char* buffer = new char[num];
        qint64 read_bytes = m_socket->read(buffer, num);
        qDebug() << read_bytes;

        QString message = QString::fromUtf8(buffer, read_bytes);
        emit newMessage(message);
        delete[] buffer;
    }

    QString TCPConnection::getIP()
    {
        if(m_socket != nullptr)
        {
            auto ipv6addr = m_socket->peerAddress();
            QHostAddress ip4_addr(ipv6addr.toIPv4Address());
            return ip4_addr.toString();
        }
        else
            return "";
    }

    quint16 TCPConnection::getPort()
    {
        if(m_socket != nullptr) return m_socket->peerPort();
        else return 0;
    }

    void TCPConnection::setupNewAsync(QString host, quint16 port)
    {
        emit setupInternal(host, port);
    }

    void TCPConnection::setupExistingAsync(qintptr socket_descriptor)
    {
        emit setupExistingInternal(socket_descriptor);
    }

    void TCPConnection::setConnectionTimeout(size_t timeout)
    {
        m_connection_timeout = timeout;
    }

    size_t TCPConnection::getConnectionTimeout()
    {
        return m_connection_timeout;
    }
}

