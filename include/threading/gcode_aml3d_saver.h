#ifndef GCODE_AML3D_SAVER_H
#define GCODE_AML3D_SAVER_H

// Qt
#include <QThread>

#include "gcode/gcode_meta.h"

namespace ORNL
{
/*!
     * \class GCodeAML3DSaver
     * \brief Threaded class that provides additional gcode processing.  Currently for AML3D
     */
class GCodeAML3DSaver : public QThread {
    Q_OBJECT
public:
    //! \brief Constructor
    //! \param tempLocation: location of gcode file
    //! \param path: path to output
    //! \param filename: filename to output
    //! \param text: current gcode
    //! \param meta: meta used to generate gcode
    GCodeAML3DSaver(QString tempLocation, QString path, QString filename, QString text, GcodeMeta meta);

    //! \brief Function that is run when start is called on this thread.
    void run() override;

private:
    //! \brief Temporary file location, output path, output filename, and text to output
    QString m_temp_location, m_path, m_filename, m_text;

    //! \brief Meta info determined from file
    GcodeMeta m_selected_meta;

};  // class GCodeAML3DSaver
}  // namespace ORNL
#endif  // GCODE_AML3D_SAVER_H
