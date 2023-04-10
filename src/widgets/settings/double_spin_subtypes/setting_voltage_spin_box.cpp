#include "widgets/settings/double_spin_subtypes/setting_voltage_spin_box.h"
#include "managers/preferences_manager.h"

namespace ORNL
{
    SettingVoltageSpinBox::SettingVoltageSpinBox(SettingTab* parent, QSharedPointer<SettingsBase> sb, QString key, fifojson json, QGridLayout* layout, int index)
        : SettingDoubleSpinBox(parent, sb, key, json, layout, index)
    {
        Voltage cur;
        m_warn = false;
        if (sb->contains(key)) cur = sb->setting<Voltage>(key);
        else
        {
            cur = json[Constants::Settings::Master::kDefault].get<Voltage>();
            sb->setSetting(key, cur);
        }

        Voltage unit = PM->getVoltageUnit();

        this->setMaximum(Constants::Limits::Maximums::kMaxVoltage.to(unit));
        this->setDecimals(m_precision);
        this->setValue(cur.to(unit));

        //Set setting units
        m_unit_label->setText(PM->getVoltageUnitText());
    }

    SettingRowBase* SettingVoltageSpinBox::createInstance(SettingTab* parent, QSharedPointer<SettingsBase> sb, QString key, fifojson json, QGridLayout* layout, int index)
    {
        return new SettingVoltageSpinBox(parent, sb, key, json, layout, index);
    }

    void SettingVoltageSpinBox::valueChanged(QVariant val)
    {
        if(m_warn)
            emit warnParent(-1); //if a value is changed, it changes for all selected settings bases, so remove a warning.
        m_warn = false;
        Voltage base_value;
        base_value.from(val.toDouble(), PM->getVoltageUnit());
        SettingDoubleSpinBox::valueChanged(base_value());
    }

    void SettingVoltageSpinBox::reloadValue()
    {
        this->blockSignals(true);
        Voltage unit = PM->getVoltageUnit();
        this->setMaximum(Constants::Limits::Maximums::kMaxVoltage.to(unit));

        m_unit_label->setText(PM->getVoltageUnitText());

        bool consistent = true;
        Voltage cur(reloadValueHelper<double>(consistent));
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
