//not currently supported, may include in the future
#if 0
#ifndef WOLFPARSER_H
#define WOLFPARSER_H

//! \file wolfparser.h

#include "common_parser.h"


namespace ORNL
{
    /*!
     * \class WolfParser
     * \brief This class implements the GCode parsing configuration for the Wolf
     * printer(s).
     */
    class WolfParser : public CommonParser
    {
    public:
        WolfParser();
        virtual void config();

    protected:
        //! \brief Function handler for the 'M101' command for turning on extruder
        //!        This function handler accepts no parameters but takes them
        //!        for error checking purposes.
        //! \throws IllegalParameterException This occurs when another parameter
        //! is passed to this function.
        virtual void M101Handler(QStringList& params);
        virtual void M103Handler(QStringList& params);

        virtual void T1Handler(QStringList& params);
        virtual void M1Handler(QStringList& params);
        virtual void M2Handler(QStringList& params);
        virtual void M5Handler(QStringList& params);
        virtual void M6Handler(QStringList& params);


    private:
    };
}  // namespace ORNL
#endif  // WOLFPARSER_H
#endif
