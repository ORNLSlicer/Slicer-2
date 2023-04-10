// Header
#include "utilities/msg_handler.h"

// Local
#include "windows/main_window.h"

namespace ORNL {
    void msgHandler(QtMsgType type, const QMessageLogContext& context, const QString& msg) {
        QString formattedMessage = qFormatLogMessage(type, context, msg);
        QByteArray localMsg = msg.toLocal8Bit();

        if (formattedMessage.isNull()) return;

        // TODO: Make behavior more complex. I.E. tie into preferences

        switch (type) {
            case QtDebugMsg:
            case QtInfoMsg:
            case QtWarningMsg:
            case QtCriticalMsg:
                // Output to main window terminal.
                //MWIN->getCmdOut() << msg;

                // Output to console.
                fprintf(stderr, "%s\n", formattedMessage.toLocal8Bit().constData());
                fflush(stderr);

                break;
            case QtFatalMsg:
                fprintf(stderr, "Fatal: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
                abort();
        }
    }
}
