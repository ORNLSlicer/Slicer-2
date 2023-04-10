#include "widgets/settings/double_spin_subtypes/setting_area_spin_box.h"
#include "managers/preferences_manager.h"

namespace ORNL
{
    SettingAreaSpinBox::SettingAreaSpinBox(SettingTab* parent, QSharedPointer<SettingsBase> sb, QString key, fifojson json, QGridLayout* layout, int index)
        : SettingDoubleSpinBox(parent, sb, key, json, layout, index)
    {
        Area cur;
        m_warn = false;
        if (sb->contains(key)) cur = sb->setting<Area>(key);
        else
        {
            cur = json[Constants::Settings::Master::kDefault].get<Area>();
            sb->setSetting(key, cur);
        }

        Area unit = PM->getDistanceUnit() * PM->getDistanceUnit();

        this->setMaximum(Constants::Limits::Maximums::kMaxArea.to(unit));
        this->setAlignment(Qt::AlignRight);
        this->setDecimals(m_precision);
        this->setValue(cur.to(unit));

        //Set setting units
        m_unit_label->setText(PM->getDistanceUnitText() % "²");
    }

    SettingRowBase* SettingAreaSpinBox::createInstance(SettingTab* parent, QSharedPointer<SettingsBase> sb, QString key, fifojson json, QGridLayout* layout, int index)
    {
        return new SettingAreaSpinBox(parent, sb, key, json, layout, index);
    }

    void SettingAreaSpinBox::valueChanged(QVariant val)
    {
        if(m_warn)
            emit warnParent(-1); //if a value is changed, it changes for all selected settings bases, so remove a warning.
        m_warn = false;
        Area base_value;
        base_value.from(val.toDouble(), PM->getDistanceUnit() * PM->getDistanceUnit());
        SettingDoubleSpinBox::valueChanged(base_value());
    }

    void SettingAreaSpinBox::reloadValue()
    {
        this->blockSignals(true);
        Area unit = PM->getDistanceUnit() * PM->getDistanceUnit();
        this->setMaximum(Constants::Limits::Maximums::kMaxArea.to(unit));
        m_unit_label->setText(PM->getDistanceUnitText() % "²");

        bool consistent = true;
        Area cur(reloadValueHelper<double>(consistent));
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
