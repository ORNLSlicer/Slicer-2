#if 0
#ifndef GCODE_HIGHLIGHTING_RULES_H
#define GCODE_HIGHLIGHTING_RULES_H

// Qt
#include <QRegularExpression>
#include <QTextCharFormat>

namespace ORNL
{

    class GcodeHighlightingRules
    {
    public:
        GcodeHighlightingRules();

        QTextCharFormat getFormat(QString text);
        QColor getColor(QString text);

    private:
        QRegularExpressionMatch m_match;
        QTextCharFormat m_default_format;

        //! \struct GcodeHighlightingRules
        //! \brief A POD struct to hold a regular expression to search for and
        //! the color
        //!        corresponding to the type of move or build.
        struct GcodeHighlightingRule
        {
            QRegularExpression pattern;
            QTextCharFormat formatting;
        };

        QVector< GcodeHighlightingRule > m_path_type_highlighting_rules;
        QVector< GcodeHighlightingRule > m_path_modifier_highlighting_rules;

        //! \brief Helper function to setup the Perimiter path type highlighting
        //! format and regex strings.
        void setupPerimeterHighlighting();

        //! \brief Helper function to setup the Infill path type highlighting
        //! format and regex strings.
        void setupInfillHighlighting();

        //! \brief Helper function to setup the Skin path type highlighting
        //! format and regex strings.
        void setupSkinHighlighting();

        //! \brief Helper function to setup the Inset path type highlighting
        //! format and regex strings.
        void setupInsetHighlighting();

        //! \brief Helper function to setup the Travel move highlighting format
        //! and regex strings.
        void setupTravelHighlighting();

        //! \brief Helper function to setup the Support path type highlighting
        //! format and regex strings.
        void setupSupportHighlighting();

        //! \brief Helper function to setup the Support Roof path type
        //! highlighting format and regex strings.
        void setupSupportRoofHighlighting();

        //! \brief Helper function to setup the Skeleton path type highlighting
        //! format and regex strings.
        void setupSkeletonHighlighting();

        //! \brief Helper function to setup the Skirt path type highlighting
        //! format and regex strings.
        void setupSkirtHighlighting();

        //! \brief Helper function to setup the Prestart highlighting format and
        //! regex strings.
        void setupPrestartHighlighting();

        //! \brief Helper function to setup the Skeleton path type highlighting
        //! format and regex strings.
        void setupInitalStartupHighlighting();

        //! \brief Helper function to setup the Skeleton path type highlighting
        //! format and regex strings.
        void setupSlowDownHighlighting();

        //! \brief Helper function to setup the Forward Tip Wipe path type
        //! highlighting format and regex strings.
        void setupForwardTipWipeHighlighting();

        //! \brief Helper function to setup the Reverse Tip Wipe path type
        //! highlighting format and regex strings.
        void setupReverseTipWipeHighlighting();

        //! \brief Helper function to setup the Coasting path type highlighting
        //! format and regex strings.
        void setupCoastingHighlighting();

        //! \brief Helper function to setup the Spiral Lift path type
        //! highlighting format and regex strings.
        void setupSpiralLiftHighlighting();

        //! \brief Helper function to setup an Unknown path type highlighting
        //! format and regex strings.
        void setupUnknownHighlighting();

        //! \brief Helper function to setup the Raft path type highlighting
        //! format and regex strings.
        void setupRaftHighlighting();

        //! \brief Helper function to setup the Brim path type highlighting
        //! format and regex strings.
        void setupBrimHighlighting();

        //! \brief Helper function to setup the IR Camera path type highlighting
        //! format and regex strings.
        //! added by Nicholas Miller
        void setupIRCameraHighlighting();

        //! \brief Helper function to setup the Laser Scanner path type highlighting
        //! format and regex strings.
        //! added by Nicholas Miller
        void setupLaserScannerHighlighting();
    };



}
#endif // GCODE_HIGHLIGHTING_RULES_H
#endif
