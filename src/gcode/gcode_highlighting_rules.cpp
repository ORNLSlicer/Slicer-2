#if 0
// Local
#include <gcode/gcode_highlighting_rules.h>
#include <utilities/constants.h>

namespace ORNL
{
    GcodeHighlightingRules::GcodeHighlightingRules()
    {
        m_default_format.setForeground(Constants::Colors::kBlack);

        setupBrimHighlighting();
        setupCoastingHighlighting();
        setupInfillHighlighting();
        setupInitalStartupHighlighting();
        setupInsetHighlighting();
        setupPerimeterHighlighting();
        setupPrestartHighlighting();
        setupRaftHighlighting();
        setupSkeletonHighlighting();
        setupSkinHighlighting();
        setupSkirtHighlighting();
        setupSlowDownHighlighting();
        setupSpiralLiftHighlighting();
        setupSupportHighlighting();
        setupSupportRoofHighlighting();
        setupForwardTipWipeHighlighting();
        setupReverseTipWipeHighlighting();
        setupTravelHighlighting();
        setupUnknownHighlighting();
    };

    QTextCharFormat GcodeHighlightingRules::getFormat(QString text)
    {
        //Check modifiers...
        for (GcodeHighlightingRule rule : m_path_modifier_highlighting_rules)
        {
            m_match = rule.pattern.match(text);
            if (m_match.hasMatch())
            {
                return rule.formatting;
            }
        }

        //...then check regular path types
        for (GcodeHighlightingRule rule : m_path_type_highlighting_rules)
        {
            m_match = rule.pattern.match(text);
            if (m_match.hasMatch())
            {
                return rule.formatting;
            }
        }

        //If nothing is found, don't highlight in any color
        return m_default_format;

    }

    QColor GcodeHighlightingRules::getColor(QString text)
    {
        return this->getFormat(text).foreground().color();
    }

    void GcodeHighlightingRules::setupPerimeterHighlighting()
    {
        QRegularExpression re(Constants::RegionTypeStrings::kPerimeter);
        re.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
        QTextCharFormat format;

        format.setForeground(Constants::Colors::kPerimeter);

        GcodeHighlightingRule temp;
        temp.pattern    = re;
        temp.formatting = format;

        m_path_type_highlighting_rules.push_back(temp);
    }


    void GcodeHighlightingRules::setupInfillHighlighting()
    {
        QRegularExpression re(Constants::RegionTypeStrings::kInfill);
        re.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
        QTextCharFormat format;

        format.setForeground(Constants::Colors::kInfill);

        GcodeHighlightingRule temp;
        temp.pattern    = re;
        temp.formatting = format;

        m_path_type_highlighting_rules.push_back(temp);
    }


    void GcodeHighlightingRules::setupSkinHighlighting()
    {
        QRegularExpression re(Constants::RegionTypeStrings::kSkin);
        re.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
        QTextCharFormat format;

        format.setForeground(Constants::Colors::kSkin);

        GcodeHighlightingRule temp;
        temp.pattern    = re;
        temp.formatting = format;

        m_path_type_highlighting_rules.push_back(temp);
    }


    void GcodeHighlightingRules::setupInsetHighlighting()
    {
        QRegularExpression re(Constants::RegionTypeStrings::kInset);
        re.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
        QTextCharFormat format;

        format.setForeground(Constants::Colors::kInset);

        GcodeHighlightingRule temp;
        temp.pattern    = re;
        temp.formatting = format;

        m_path_type_highlighting_rules.push_back(temp);
    }

    void GcodeHighlightingRules::setupTravelHighlighting()
    {
        QRegularExpression re(Constants::RegionTypeStrings::kTravel);
        re.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
        QTextCharFormat format;

        format.setForeground(Constants::Colors::kTravel);

        GcodeHighlightingRule temp;
        temp.pattern    = re;
        temp.formatting = format;

        m_path_type_highlighting_rules.push_back(temp);
    }

    void GcodeHighlightingRules::setupSupportHighlighting()
    {
        QRegularExpression re(Constants::RegionTypeStrings::kSupport);
        re.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
        QTextCharFormat format;

        format.setForeground(Constants::Colors::kSupport);

        GcodeHighlightingRule temp;
        temp.pattern    = re;
        temp.formatting = format;

        m_path_type_highlighting_rules.push_back(temp);
    }

    void GcodeHighlightingRules::setupSupportRoofHighlighting()
    {
        QRegularExpression re(Constants::RegionTypeStrings::kSupportRoof);
        re.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
        QTextCharFormat format;

        format.setForeground(Constants::Colors::kSupportRoof);

        GcodeHighlightingRule temp;
        temp.pattern    = re;
        temp.formatting = format;

        m_path_type_highlighting_rules.push_back(temp);
    }

    void GcodeHighlightingRules::setupSkeletonHighlighting()
    {
        // TODO: Figure out string and color for skeleton, none exist
    }

    void GcodeHighlightingRules::setupSkirtHighlighting()
    {
        // TODO: Figure out string and color for skirt, none exists.
    }

    void GcodeHighlightingRules::setupPrestartHighlighting()
    {
        QRegularExpression re(Constants::PathModifierStrings::kPrestart);
        QTextCharFormat format;
        re.setPatternOptions(QRegularExpression::CaseInsensitiveOption);

        format.setForeground(Constants::Colors::kPrestart);

        GcodeHighlightingRule temp;
        temp.pattern    = re;
        temp.formatting = format;

        m_path_modifier_highlighting_rules.push_back(temp);
    }

    void GcodeHighlightingRules::setupInitalStartupHighlighting()
    {
        QRegularExpression re(Constants::PathModifierStrings::kInitialStartup);
        re.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
        QTextCharFormat format;

        format.setForeground(Constants::Colors::kInitialStartup);

        GcodeHighlightingRule temp;
        temp.pattern    = re;
        temp.formatting = format;

        m_path_modifier_highlighting_rules.push_back(temp);
    }

    void GcodeHighlightingRules::setupSlowDownHighlighting()
    {
        QRegularExpression re(Constants::PathModifierStrings::kSlowDown);
        re.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
        QTextCharFormat format;

        format.setForeground(Constants::Colors::kSlowDown);

        GcodeHighlightingRule temp;
        temp.pattern    = re;
        temp.formatting = format;

        m_path_modifier_highlighting_rules.push_back(temp);
    }

    void GcodeHighlightingRules::setupForwardTipWipeHighlighting()
    {
        QRegularExpression re(Constants::PathModifierStrings::kForwardTipWipe);
        re.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
        QTextCharFormat format;

        format.setForeground(Constants::Colors::kForwardTipWipe);

        GcodeHighlightingRule temp;
        temp.pattern    = re;
        temp.formatting = format;

        m_path_modifier_highlighting_rules.push_back(temp);
    }

    void GcodeHighlightingRules::setupReverseTipWipeHighlighting()
    {
        QRegularExpression re(Constants::PathModifierStrings::kReverseTipWipe);
        re.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
        QTextCharFormat format;

        format.setForeground(Constants::Colors::kReverseTipWipe);

        GcodeHighlightingRule temp;
        temp.pattern    = re;
        temp.formatting = format;

        m_path_modifier_highlighting_rules.push_back(temp);
    }

    void GcodeHighlightingRules::setupCoastingHighlighting()
    {
        QRegularExpression re(Constants::PathModifierStrings::kCoasting);
        re.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
        QTextCharFormat format;

        format.setForeground(Constants::Colors::kCoasting);

        GcodeHighlightingRule temp;
        temp.pattern    = re;
        temp.formatting = format;

        m_path_modifier_highlighting_rules.push_back(temp);
    }

    void GcodeHighlightingRules::setupSpiralLiftHighlighting()
    {
        QRegularExpression re(Constants::PathModifierStrings::kSpiralLift);
        re.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
        QTextCharFormat format;

        format.setForeground(Constants::Colors::kSpiralLift);

        GcodeHighlightingRule temp;
        temp.pattern    = re;
        temp.formatting = format;

        m_path_modifier_highlighting_rules.push_back(temp);
    }

    void GcodeHighlightingRules::setupUnknownHighlighting()
    {
        // TODO: Figure out what is meant by 'unknown'
        QRegularExpression re(Constants::RegionTypeStrings::kUnknown);
        re.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
        QTextCharFormat format;

        format.setForeground(Constants::Colors::kUnknown);

        GcodeHighlightingRule temp;
        temp.pattern    = re;
        temp.formatting = format;

        m_path_type_highlighting_rules.push_back(temp);
    }

    void GcodeHighlightingRules::setupRaftHighlighting()
    {
        // TODO: Figure out string and color for raft
    }

    void GcodeHighlightingRules::setupBrimHighlighting()
    {
        // TODO: Figure out string and color for brim
    }
} //namespace ORNL
#endif
