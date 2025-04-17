#ifndef INGERSOLLPARSER_H
#define INGERSOLLPARSER_H

#include "common_parser.h"
#include "gcode/gcode_meta.h"

namespace ORNL {
/*!
 * \class IngersollParser
 * \brief This class implements the GCode parsing configuration for the
 * Ingersoll 3D printer(s). \note The current commands that this class
 * implements are:
 *          - M Commands:
 *              -
 *          - Tool Change Commands:
 *              - All T commands are checked for syntax errors but otherwise
 * do nothing. The commands inherited from the CommonParser superclass are:
 *          - G Commands:
 *              - G0, G1, G2, G3, G4
 *          - Line Comment Delimiter:
 *              - ;
 *
 */
class IngersollParser : public CommonParser {
  public:
    //! \brief Standard constructor that specifies meta type and whether or not
    //! to alter layers for minimal layer time
    //! \param meta GcodeMeta struct that includes information about units and
    //! delimiters
    //! \param allowLayerAlter bool to determinw whether or not to alter layers to
    //! adhere to minimal layer times
    //! \param lines Original lines of the gcode file
    //! \param upperLines Uppercase lines of the original used for ease of parsing/comparison
    IngersollParser(GcodeMeta meta, bool allowLayerAlter, QStringList& lines, QStringList& upperLines);

    virtual void config() override;

    //! \brief Modify layer time by adjusting feedrate for motions for that layer
    //! \param modifer Amount to modify feedrate from 0 to 1
    void AdjustFeedrate(double modifier) override;

  protected:
    //! \brief Handler for 'G0' command for motion
    //! G0 must be overridden due to wire feed, formatting is adjusted and then
    //! referred to base G0 handler
    //! \param params Parameters of current interest include: X, Y, Z, speed
    void G0Handler(QVector<QString> params) override;

    //! \brief Handler for 'G1' command for motion
    //! G1 must be overridden due to wire feed, formatting is adjusted and then
    //! referred to base G1 handler
    //! \param params Parameters of current interest include: X, Y, Z, speed
    void G1Handler(QVector<QString> params) override;

  private:
    //! \brief UA and H parameters to ignore in G1's
    QString m_ua_parameter;
    QChar m_h_parameter;

    //! \brief Copy of G-Code meta from Common Parser
    GcodeMeta m_meta_copy;

    //! \brief Copies of all lines and all uppercase lines from Common Parser
    QStringList &m_lines_copy, &m_upper_lines_copy;
};
} // namespace ORNL
#endif // INGERSOLPARSER_H
