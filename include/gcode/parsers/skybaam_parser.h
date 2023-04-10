//Skybaam dedicated parser currently unnecessary as it conforms to cincinnati parser
//only current difference is that G4 uses S instead of P parameter, G4 currently accepts
//multiple valid parameters
#if 0
#ifndef SKYBAAM_H
#define SKYBAAM_H

//! \file skybaam_parser.h

#include "common_parser.h"

namespace ORNL
{
    /*!
     * \class SkyBaamParser
     * \brief This class implements the GCode parsing configuration for the
     * SkyBAAM 3D printer(s). \note The current commands that this class
     * implements are:
     *          - M Commands:
     *              - M3, M5, M30, M68
     *          - Tool Change Commands:
     *              - All T commands are checked for syntax errors but otherwise
     * do nothing.
     *          - Block Comment Delimiter:
     *              - ()
     *      The commands inherited from the CommonParser superclass are:
     *          - G Commands:
     *              - G0, G1, G2, G3, G4
     * invalid in for the parser.
     *
     */
    class SkyBaamParser : public CommonParser
    {
    public:
        SkyBaamParser();

        virtual void config();

    protected:

        //! \brief Function handler for the 'M3' command for turning the extruder on
        //! and setting the spindle speed rate.
        //!        The function handler accepts the following parameters:
        //!             - S: Sets the spindle speed rate for the extruder.
        //!        The S command must exist or an exception is thrown.
        //! \param params List of parameters that have been split up to be
        //! parsed. \throws IllegalParameterException This occurs during either
        //! a conversion error, a parameter missing a value,
        //!                                   a duplicate parameter, if a
        //!                                   required parameter is not passed,
        //!                                   or an illegal parameter passed.
        virtual void M3Handler(QStringList& params);

        //! \brief Function handler for the 'M5' command for turning off the
        //! extruder.
        //!        This function handler accepts no parameters but takes them
        //!        for error checking purposes.
        //! \throws IllegalParameterException This occurs when another parameter
        //! is passed to this function.
        virtual void M5Handler(QStringList& params);

        //! \brief Function handler for the 'M30' command for stating the end of
        //! Gcode commands within a Gcode file..
        //!        This function handler accepts no parameters but takes them
        //!        for error checking purposes.
        //! \throws IllegalParameterException This occurs when another parameter
        //! is passed to this function.
        virtual void M30Handler(QStringList& params);

        //! \brief Function Handler for the 'M68' command that sends the
        //! extruder to the park position.
        //!        The function accepts no parameters but takes them for error
        //!        checking purposes.
        //! \throws IllegalParameterException This occurs when another parameter
        //! is passed to this function.
        virtual void M68Handler(QStringList& params);


        //! \brief Funciton handler for tool changes in GCode. For now its just
        //! a placeholder function.
        virtual void ToolChangeHandler(QStringList& params);

    private:
        bool m_voltage_control = true;
        double m_voltage_control_value;
    };
}  // namespace ORNL
#endif  // SKYBAAM_H
#endif
