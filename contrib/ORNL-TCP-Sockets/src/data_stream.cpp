#include "include/data_stream.h"

namespace ORNL
{
    DataStream::DataStream(TCPConnection *connection, QObject* parent) : QObject(parent)
    {
        m_connection = connection;
        connect(m_connection, &TCPConnection::newMessage, this, &DataStream::handleNewMessage);
        connect(this, &DataStream::write, m_connection, &TCPConnection::write);
    }

    DataStream::~DataStream()
    {
        delete m_connection;
    }

    bool DataStream::isEmpty()
    {
        return m_data.isEmpty();
    }

    QString DataStream::getNextMessage()
    {
        return m_data.dequeue();
    }

    TCPConnection *DataStream::getConnection()
    {
        return m_connection;
    }

    void DataStream::send(const QString& msg)
    {
        emit write(msg);
    }

    void DataStream::handleNewMessage(const QString& msg)
    {
        m_data.enqueue(msg);
        emit newData();
    }
}

