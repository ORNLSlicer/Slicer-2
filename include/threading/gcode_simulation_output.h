#ifndef GCODE_SIMULATION_OUTPUT_H
#define GCODE_SIMULATION_OUTPUT_H

// Qt
#include <QThread>

#include "gcode/gcode_meta.h"
#include "utilities/mathutils.h"

namespace ORNL
{
/*!
     * \class GCodeSimulationOutput
     * \brief Threaded class that provides additional gcode processing for user with ABAQUS.
     */
class GCodeSimulationOutput : public QThread {
    Q_OBJECT
public:
    //! \brief Constructor
    //! \param tempLocation: location of gcode file
    //! \param path: path to output
    //! \param filename: filename to output
    //! \param text: current gcode
    //! \param meta: meta used to generate gcode
    GCodeSimulationOutput(QString tempLocation, QString path, QString filename, QString text, GcodeMeta meta);

    //! \brief Function that is run when start is called on this thread.
    void run() override;

    //! \brief Calculate the time needed for the current move
    //! \param X: the new X position
    //! \param Y: the new Y position
    //! \param Z: the new Z position
    //! \param W: the new W position
    //! \param F: the new velocity command
    void calculateTime(QString X, QString Y, QString Z, QString W, QString F);

private:
    //! \brief Temporary file location, output path, output filename, and text to output
    QString m_temp_location, m_path, m_filename, m_text;

    //! \brief Current X, Y, Z, and W locations
    Distance currentX, currentY, currentZ, currentW;

    //! \brief Defines whether the syntax uses metric units
    bool useMetric;

    //! \brief Defines the move as G0 and not requiring an F
    bool isG0;

    //! \brief Current elapsed time
    Time currentTime;

    //! \brief Meta info determined from file
    GcodeMeta m_selected_meta;

};  // class GCodeSimulationOutput
}  // namespace ORNL
#endif // GCODE_SIMULATION_OUTPUT_H
