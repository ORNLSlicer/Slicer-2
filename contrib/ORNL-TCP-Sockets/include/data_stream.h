#ifndef DATASTREAM_H
#define DATASTREAM_H

#include <QObject>
#include <QQueue>

#include "tcp_server.h"
#include "tcp_connection.h"

namespace ORNL
{
    //! \class DataStream
    //! \brief Provides easy access to receive/ send functions from a TCPConnection. This class maintains an
    //!        incoming message FIFO queue from the TCP Connection
    class DataStream : public QObject
    {
    Q_OBJECT
    public:
        //! \brief Constructor
        //! \param connection the TCPConnection to wrap
        //! \param parent the parent widget
        DataStream(TCPConnection *connection, QObject* parent = nullptr);

        //! \brief Destructor
        ~DataStream();

        //! \brief are there any messages in the queue
        //! \return if there are any messages in the queue
        bool isEmpty();

        //! \brief removes and returns the next message in the queue
        //! \return a string object message
        //! \note this assumes there is a message in the queue
        QString getNextMessage();

        //! \brief gets the internal TCP connection
        //! \return a TCPConnection
        TCPConnection* getConnection();

    public slots:
        //! \brief sends a message to the remote client
        //! \param msg the message to send
        void send(const QString& msg);

    signals:
        //! \brief emitted when new data is available in the stream
        void newData();

        //! \brief internal signal used to tell the TCPConnection to write
        //! \param msg the message to write
        //! \note DO NOT directly call this \see send()
        void write(const QString& msg);

    private:
        //! \brief handles incoming messages from the TCPConnection
        //! \param msg the message
        void handleNewMessage(const QString&);

        //! \brief the TCP connection
        TCPConnection *m_connection;

        //! \brief FIFO queue to hold incoming messages
        QQueue<QString> m_data;
    };
}

#endif // DATASTREAM_H
