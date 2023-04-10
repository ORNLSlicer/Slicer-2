//Gudel dedicated parser currently unnecessary as its only current handlers are M3/M5
//M3/M5 has been moved to the common parser as all current syntaxes utilize same format
#if 0
#ifndef GUDEL_PARSER_H
#define GUDEL_PARSER_H
#include "common_parser.h"

namespace ORNL
{
    /*!
     * \class GudelParser
     * \brief This class implements the GCode parsing configuration for the
     * Gudel 3D printer(s). \note The current commands that this class
     * implements are:
     *          - M Commands:
     *              - M3, M5
     */
    class GudelParser : public CommonParser
    {
        public:
            //! \brief Standard constructor that specifies meta type and whether or not
            //! to alter layers for minimal layer time
            //! \param meta GcodeMeta struct that includes information about units and
            //! delimiters
            //! \param allowLayerAlter bool to determinw whether or not to alter layers to
            //! adhere to minimal layer times
            GudelParser(GcodeMeta meta, bool allowLayerAlter);

            //! \brief Function to initialize syntax specific handlers
            virtual void config();

        protected:
            //! \brief Handler for 'M3' command for turning on extruder
            //! \param params Accepted by function for formatting check, but are not
            //! used for the command
            virtual void M3Handler(QVector<QStringRef> params);

            //! \brief Handler for 'M5' command for turning off extruder
            //! \param params Accepted by function for formatting check, but are not
            //! used for the command
            virtual void M5Handler(QVector<QStringRef> params);

    };
}

#endif // GUDEL_PARSER_H
#endif
