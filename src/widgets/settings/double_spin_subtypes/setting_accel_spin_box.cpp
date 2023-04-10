#include "widgets/settings/double_spin_subtypes/setting_accel_spin_box.h"
#include "managers/preferences_manager.h"

namespace ORNL
{
    SettingAccelSpinBox::SettingAccelSpinBox(SettingTab* parent, QSharedPointer<SettingsBase> sb, QString key, fifojson json, QGridLayout* layout, int index)
        : SettingDoubleSpinBox(parent, sb, key, json, layout, index)
    {
        Acceleration cur;
        m_warn = false;
        if (sb->contains(key)) cur = sb->setting<Acceleration>(key);
        else
        {
            cur = json[Constants::Settings::Master::kDefault].get<Acceleration>();
            m_sb->setSetting(key, cur);
        }

        Acceleration unit = PM->getAccelerationUnit();

        this->setMaximum(Constants::Limits::Maximums::kMaxAccel.to(unit));
        this->setDecimals(m_precision);
        this->setValue(cur.to(unit));

        //Set setting units
        m_unit_label->setText(PM->getAccelerationUnitText());
    }

    SettingRowBase* SettingAccelSpinBox::createInstance(SettingTab* parent, QSharedPointer<SettingsBase> sb, QString key, fifojson json, QGridLayout* layout, int index)
    {
        return new SettingAccelSpinBox(parent, sb, key, json, layout, index);
    }

    void SettingAccelSpinBox::valueChanged(QVariant val)
    {
        if(m_warn)
            emit warnParent(-1); //if a value is changed, it changes for all selected settings bases, so remove a warning.
        m_warn = false;
        Acceleration base_value;
        base_value.from(val.toDouble(), PM->getAccelerationUnit());
        SettingDoubleSpinBox::valueChanged(base_value());
    }

    void SettingAccelSpinBox::reloadValue()
    {
        this->blockSignals(true);
        Acceleration unit = PM->getAccelerationUnit();
        this->setMaximum(Constants::Limits::Maximums::kMaxAccel.to(unit));
        m_unit_label->setText(PM->getAccelerationUnitText());

        bool consistent = true;
        Acceleration cur(reloadValueHelper<double>(consistent));
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
