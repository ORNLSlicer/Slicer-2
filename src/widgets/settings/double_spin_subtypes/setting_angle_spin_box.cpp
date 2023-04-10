#include "widgets/settings/double_spin_subtypes/setting_angle_spin_box.h"
#include "managers/preferences_manager.h"

namespace ORNL
{
    SettingAngleSpinBox::SettingAngleSpinBox(SettingTab* parent, QSharedPointer<SettingsBase> sb, QString key, fifojson json, QGridLayout* layout, int index)
        : SettingDoubleSpinBox(parent, sb, key, json, layout, index)
    {
        Angle cur;
        m_warn = false;
        if (sb->contains(key)) cur = sb->setting<Angle>(key);
        else
        {
            cur = json[Constants::Settings::Master::kDefault].get<Angle>();
            sb->setSetting(key, cur);
        }

        Angle unit = PM->getAngleUnit();

        this->setMaximum(Constants::Limits::Maximums::kMaxAngle.to(unit));
        this->setMinimum(Constants::Limits::Minimums::kMinAngle.to(unit));
        this->setDecimals(m_precision);
        this->setValue(cur.to(unit));

        //Set setting units
        m_unit_label->setText(PM->getAngleUnitText());
    }

    SettingRowBase* SettingAngleSpinBox::createInstance(SettingTab* parent, QSharedPointer<SettingsBase> sb, QString key, fifojson json, QGridLayout* layout, int index)
    {
        return new SettingAngleSpinBox(parent, sb, key, json, layout, index);
    }

    void SettingAngleSpinBox::valueChanged(QVariant val)
    {
        if(m_warn)
            emit warnParent(-1); //if a value is changed, it changes for all selected settings bases, so remove a warning.
        m_warn = false;
        Angle base_value;
        base_value.from(val.toDouble(), PM->getAngleUnit());
        SettingDoubleSpinBox::valueChanged(base_value());
    }

    void SettingAngleSpinBox::reloadValue()
    {
        this->blockSignals(true);
        Angle unit = PM->getAngleUnit();
        this->setMaximum(Constants::Limits::Maximums::kMaxAngle.to(unit));
        this->setMinimum(Constants::Limits::Minimums::kMinAngle.to(unit));
        m_unit_label->setText(PM->getAngleUnitText());

        bool consistent = true;
        Angle cur(reloadValueHelper<double>(consistent));
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
