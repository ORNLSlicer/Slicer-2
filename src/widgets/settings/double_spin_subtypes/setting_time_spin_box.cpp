#include "widgets/settings/double_spin_subtypes/setting_time_spin_box.h"
#include "managers/preferences_manager.h"

namespace ORNL
{
    SettingTimeSpinBox::SettingTimeSpinBox(SettingTab* parent, QSharedPointer<SettingsBase> sb, QString key, fifojson json, QGridLayout* layout, int index)
         : SettingDoubleSpinBox(parent, sb, key, json, layout, index)
    {
        Time cur;
        m_warn = false;
        if (sb->contains(key)) cur = sb->setting<Time>(key);
        else
        {
            cur = json[Constants::Settings::Master::kDefault].get<Time>();
            sb->setSetting(key, cur);
        }

        Time unit = PM->getTimeUnit();

        this->setMaximum(Constants::Limits::Maximums::kMaxTime.to(unit));
        this->setDecimals(m_precision);
        this->setValue(cur.to(unit));

        //Set setting units
        m_unit_label->setText(PM->getTimeUnitText());
    }

    SettingRowBase* SettingTimeSpinBox::createInstance(SettingTab* parent, QSharedPointer<SettingsBase> sb, QString key, fifojson json, QGridLayout* layout, int index)
    {
        return new SettingTimeSpinBox(parent, sb, key, json, layout, index);
    }

    void SettingTimeSpinBox::valueChanged(QVariant val)
    {
        if(m_warn)
            emit warnParent(-1); //if a value is changed, it changes for all selected settings bases, so remove a warning.
        m_warn = false;
        Time base_value;
        base_value.from(val.toDouble(), PM->getTimeUnit());
        SettingDoubleSpinBox::valueChanged(base_value());
    }

    void SettingTimeSpinBox::reloadValue()
    {
        this->blockSignals(true);
        Time unit = PM->getTimeUnit();
        this->setMaximum(Constants::Limits::Maximums::kMaxTime.to(unit));

        m_unit_label->setText(PM->getTimeUnitText());

        bool consistent = true;
        Time cur(reloadValueHelper<double>(consistent));
        if(consistent)
             setValue(cur.to(unit));

        this->blockSignals(false);
        emit modified(m_key);

        // give (1) warning if there is a mismatch, otherwise give no (0) warning
        if (!consistent) {
            emit warnParent(1);
            m_warn = true;
        }
        else
            emit warnParent(0);
    }
} // namespace ORNL
