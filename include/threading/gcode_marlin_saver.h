#ifndef GCODE_MARLIN_SAVER_H
#define GCODE_MARLIN_SAVER_H

// Qt
#include <QThread>

#include "gcode/gcode_meta.h"

namespace ORNL
{
/*!
     * \class GCodeMarlinSaver
     * \brief Threaded class that provides data output from the Marlin syntax.
     */
class GCodeMarlinSaver : public QThread {
    Q_OBJECT
public:
    //! \brief Constructor
    //! \param tempLocation: location of gcode file
    //! \param path: path to output
    //! \param filename: filename to output
    //! \param text: current gcode
    //! \param meta: meta used to generate gcode
    GCodeMarlinSaver(QString tempLocation, QString path, QString filename, QString text, GcodeMeta meta);

    //! \brief Function that is run when start is called on this thread.
    void run() override;

private:
    //! \brief Temporary file location, output path, output filename, and text to output
    QString m_temp_location, m_path, m_filename, m_text;

    //! \brief Meta info determined from file
    GcodeMeta m_selected_meta;

};  // class GCodeMarlinSaver
}  // namespace ORNL
#endif  // GCODE_MARLIN_SAVER_H
