#ifndef TEMPLATELAYERDIALOG_H
#define TEMPLATELAYERDIALOG_H

// Qt Libraries
#include <QDialog>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QListWidget>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QGroupBox>
#include <QToolButton>
#include <QLineEdit>
#include <QFileDialog>
#include <QAbstractButton>
#include <QMessageBox>
#include <QVector>
#include <QSharedPointer>
#include <QPair>

//Local
#include "widgets/programmatic_check_box.h"
#include "units/unit.h"
#include "configs/range.h"
#include "widgets/settings/setting_bar.h"
#include "configs/settings_base.h"

namespace ORNL{

/*!
 * \class TemplateLayerSave
 * \brief Dialog for saving layer bar template.
 */

    class TemplateLayerDialog : public QDialog
    {
    public:
        Q_OBJECT
        public:
            //! \brief Default Constructor.
            explicit TemplateLayerDialog(QWidget* parent = nullptr);

            //! \brief Constructor from a selected layer bar template.
            TemplateLayerDialog(QWidget* parent, QVector<SettingsRange> selected_template, QString template_name);

        private slots:

            //! \brief File dialog for selecting a file.
            void fileDialog();

        private:
            //! \brief Setup the static widgets and their layouts.
            void setupUi();

            //! \brief 1. Setup the window properties.
            void setupWindow();

            //! \brief 2. Setup the widgets.
            void setupWidgets();

            //! \brief 3. Setup the layouts and insert their children.
            void setupLayouts();

            //! \brief 4. Setup the events for the various widgets.
            void setupEvents();

            //! \brief 5. Add new layer to list.
            void addNewLayer();

            //! \brief 6. Delete existing layer from list
            void deleteLayer();

            //! \brief 7. Adds setting bar to right side.
            void addSettingBar();

            //! \brief 8. Set settings fro currently selected layer.
            void setCurrentLayerSettings(QListWidgetItem* item);

            //! \brief 9. Returns index of range from array of ranges.
            int findRange(SettingsRange new_range);

            //! \brief Sort ranges in list by bottom element in range.
            static bool sortByMin(SettingsRange& a, SettingsRange& b);

            //! \brief Save Layer bar template to s2l file.
            void saveLayerTemplate();

            // Layout
            QHBoxLayout* m_layout;
            QHBoxLayout* m_file_layout;

            // Container
            QWidget* m_file_container;

            // Button container
            QDialogButtonBox* m_button_container;

            // Widgets
            QVBoxLayout* m_leftside;
            QVBoxLayout* m_rightside;
            QHBoxLayout* m_right_buttons;
            QHBoxLayout* m_left_buttons;
            QListWidget* m_ranges_list;
            QLabel* m_range_label;
            QLabel* m_edit_label;
            QLabel* m_new_label;
            QLabel* m_save_label;
            QLabel* m_name_label;
            QToolButton* m_browse_btn;
            QLineEdit* m_edit;
            QLineEdit* m_name_edit;
            QPushButton* m_new;
            QPushButton* m_delete;
            QPushButton* m_save;
            QPushButton* m_cancel;
            SettingBar* settingbar;

            //! \brief Result filename.
            QString m_filename;

            //! \brief Array of ranges and their corresponding settings bases.
            QVector<SettingsRange> m_settings_ranges;

    };
}

#endif // TEMPLATELAYERDIALOG_H
