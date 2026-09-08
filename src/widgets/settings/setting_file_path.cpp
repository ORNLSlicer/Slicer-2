#include "widgets/settings/setting_file_path.h"

#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QToolTip>

#include <qgridlayout.h>
#include <qlabel.h>
#include <qlineedit.h>
#include <qnamespace.h>
#include <qobject.h>
#include <qsharedpointer.h>
#include <qtoolbutton.h>
#include <qvariant.h>

#include "configs/settings_base.h"
#include "utilities/constants.h"
#include "utilities/qt_json_conversion.h"
#include "widgets/settings/setting_row_base.h"

namespace ORNL {

SettingFilePath::SettingFilePath(SettingTab* parent, QSharedPointer<SettingsBase> sb, QString key, fifojson json,
                                 QGridLayout* layout, int index)
    : QWidget(parent), SettingRowBase(parent, sb, key, json, layout, index) {
    QString cur;
    m_warn = false;
    if (m_sb->contains(m_key))
        cur = m_sb->setting<QString>(m_key);
    else {
        cur = json.operator[](Constants::Settings::Master::kDefault).get<QString>();
        m_sb->setSetting(m_key, cur);
    }

    m_line_edit.reset(new QLineEdit(this));
    m_line_edit->setAlignment(Qt::AlignRight);
    m_line_edit->setText(cur);

    m_browse_button.reset(new QToolButton(this));
    m_browse_button->setText("...");
    m_browse_button->setToolTip("Select file");
    m_browse_button->setFixedWidth(28);

    QHBoxLayout* row_layout = new QHBoxLayout(this);
    row_layout->setContentsMargins(0, 0, 0, 0);
    row_layout->setSpacing(4);
    row_layout->addWidget(m_line_edit.get());
    row_layout->addWidget(m_browse_button.get());

    connect(m_line_edit.get(), &QLineEdit::textChanged, this, &SettingFilePath::valueChanged);
    connect(m_browse_button.get(), &QToolButton::clicked, this, &SettingFilePath::selectFile);
    connect(this, &SettingFilePath::modified, parent, &SettingTab::keyModified);
    connect(this, &SettingFilePath::warnParent, parent, &SettingTab::headerWarning);

    layout->addWidget(this, index, 1, Qt::AlignRight);

    m_unit_label.reset(new QLabel(""));
    layout->addWidget(m_unit_label.get(), index, 2, Qt::AlignLeft);
    registerRowWidget(this);
}

SettingRowBase* SettingFilePath::createInstance(SettingTab* parent, QSharedPointer<SettingsBase> sb, QString key,
                                                fifojson json, QGridLayout* layout, int index) {
    return new SettingFilePath(parent, sb, key, json, layout, index);
}

void SettingFilePath::setEnabled(bool enabled) {
    SettingRowBase::setEnabled(enabled);
    applyWidgetState(this);
}

void SettingFilePath::setNotification(QString msg) {
    this->setStyleFromFile(m_line_edit.get(), m_theme_path + "setting_rows_warning.qss");
    QToolTip::showText(this->mapToGlobal(QPoint(0, 0)), msg, nullptr, QRect(), 30000);
}

void SettingFilePath::clearNotification() {
    this->setStyleFromFile(m_line_edit.get(), m_theme_path + "setting_rows_normal.qss");
    m_line_edit->setToolTip("");
}

void SettingFilePath::hide() {
    SettingRowBase::hide();
    applyWidgetState(this);
}

void SettingFilePath::show() {
    SettingRowBase::show();
    applyWidgetState(this);
}

void SettingFilePath::valueChanged(QVariant val) {
    if (m_warn)
        emit warnParent(-1);  // if a value is changed, it changes for all selected settings bases, so remove a warning.
    m_warn = false;
    valueChangedHelper<QString>(val.toString());
    emit modified(m_key);
}

void SettingFilePath::reloadValue() {
    m_line_edit->blockSignals(true);
    bool consistent = true;
    QString cur     = reloadValueHelper<QString>(consistent);
    if (consistent) m_line_edit->setText(cur);

    m_line_edit->blockSignals(false);
    emit modified(m_key);
    emit warnParent(warningCountDelta(!consistent, m_warn));
}

void SettingFilePath::selectFile() {
    const QFileInfo current_file(m_line_edit->text());
    const QString start_directory = current_file.dir().exists() ? current_file.absolutePath() : QDir::homePath();
    const QString file_path =
        QFileDialog::getOpenFileName(this, tr("Select File"), start_directory, tr("NC Files (*.nc);;All Files (*)"));

    if (!file_path.isEmpty()) { m_line_edit->setText(QDir::toNativeSeparators(file_path)); }
}
}  // Namespace ORNL
