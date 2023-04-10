#ifndef AEROBASICPARSER_H
#define AEROBASICPARSER_H

#include "common_parser.h"

namespace ORNL
{
    /*!
     * \class AeroBasicParser
     * \brief This class implements the GCode parsing configuration for the
     * AeroBasic 3D printer(s).
     */
    class AeroBasicParser : public CommonParser
    {
        public:
            //! \brief Standard constructor that specifies meta type and whether or not
            //! to alter layers for minimal layer time
            //! \param meta GcodeMeta struct that includes information about units and
            //! delimiters
            //! \param allowLayerAlter bool to determinw whether or not to alter layers to
            //! adhere to minimal layer times
            //! \param lines Original lines of the gcode file
            //! \param upperLines Uppercase lines of the original used for ease of parsing/comparison
            AeroBasicParser(GcodeMeta meta, bool allowLayerAlter, QStringList& lines, QStringList& upperLines);

            //! \brief forwards aliases to common parser version
            //! \param params gcode-params
            void G0Handler(QVector<QStringRef> params) override;

            //! \brief forwards aliases to common parser version
            //! \param params gcode-params
            void G1Handler(QVector<QStringRef> params) override;

            //! \brief forwards aliases to common parser version
            //! \param params gcode-params
            void G2Handler(QVector<QStringRef> params) override;

            //! \brief forwards aliases to common parser version
            //! \param params gcode-params
            void G3Handler(QVector<QStringRef> params) override;

            //! \brief forwards aliases to common parser version
            //! \param params gcode-params
            void G4Handler(QVector<QStringRef> params) override;

            //! \brief Function to initialize syntax specific handlers
            virtual void config() override;
    };
}  // namespace ORNL

#endif // AEROPARSER_H
