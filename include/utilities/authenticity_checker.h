#ifndef AUTHENTICITYCHECKER_H
#define AUTHENTICITYCHECKER_H

#include <QWidget>
#include <QDateTime>
#include <QUdpSocket>

namespace ORNL
{
    /*!
     * \class AuthenticityChecker
     * \brief checks registry and time to make sure the program is operating in a valid environment
     */
    class AuthenticityChecker : public QObject
    {
        Q_OBJECT
    public:
        //! \brief Constructor
        //! \param parent: parent to draw error dialogs on
        AuthenticityChecker(QWidget* parent);

        //! \brief starts the check
        void startCheck();

    signals:
        //! \brief Signal to indicate that the check is done
        //! \param ok if it succeed or not
        void done(bool ok);

    private:
        //! \brief checks the registry for settings
        void checkRegistry();

        //! \brief starts the nist clock check request
        void startNISTCheck();

        //! \brief handles the nist clock check request
        void nistCheckDone();

        //! \brief parent widget
        QWidget* m_parent;

        //! \brief Separate timeout to override default OS timeout for http request
        QTimer* m_duration_timeout;

        //! \brief Parsed response from nist request for date/time
        QDateTime m_nist_time;

        //! \brief Socket to connect to NIST servers for time check (if necessary)
        QUdpSocket* m_upd_socket;

        //! \brief Current program version
        QString m_current_version;
    };
}

#endif // AUTHENTICITYCHECKER_H
