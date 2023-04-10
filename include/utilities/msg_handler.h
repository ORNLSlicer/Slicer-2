#ifndef MSG_HANDLER_H
#define MSG_HANDLER_H

#include <QtGlobal>

namespace ORNL {
    //! \brief Function that handles output through Qt's output protocols (qDebug, qWarning, etc.)
    //!
    //! This function was intended to tee output to both MainWindow's terminal and stderr. However, this ended up causing a couple issues:
    //!     1. Debug output was very slow when large ammounts of text needed to be printed.
    //!     2. On Windows OS, the reference to the singleton of MainWindow somehow caused a double instantiation of the singleton.
    //!
    //! \todo Either fix the issues with the function or toss it (along with CmdWidget).
    //! \note Message handler is set in main.cpp.
    void msgHandler(QtMsgType type, const QMessageLogContext &context, const QString &msg);

} // Namespace ORNL

#endif // MSG_HANDLER_H
