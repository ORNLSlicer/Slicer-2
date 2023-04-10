#ifndef EMBOSSDIALOG_H
#define EMBOSSDIALOG_H

// Qt
#include <QWidget>
#include <QDialog>

// Local
#include "part/part.h"
#include "widgets/xyz_input.h"

// Forward
class QGridLayout;
class QHBoxLayout;
class QToolButton;
class QComboBox;
class QLabel;
class QTableWidget;
class QLineEdit;
class QRadioButton;
class QDoubleSpinBox;
class QDialogButtonBox;
class QListWidget;

namespace ORNL {
    class PreviewView;
    class EmbossView;

    /*!
     * \brief Dialog that allows user to setup an emboss build.
     */
    class EmbossDialog : public QDialog {
        Q_OBJECT
        public:
            //! \brief Constructor
            //! \param session_set: Parts to show for selection as base.
            //! \param parent: Parent window.
            EmbossDialog(QSet<QSharedPointer<Part>> session_set, QWidget* parent = nullptr);

            //! \brief Result of setup.
            QSharedPointer<Part> resultPart();

        public slots:
            //! \brief Loads a base object from a file.
            void baseObjectFromFile();
            //! \brief Responds to mesh loader.
            //! \todo Actually use this instead of Debug call.
            void baseObjectLoadCompleted(QSharedPointer<Part> p);

            //! \brief Uses a base object in the session.
            void baseObjectFromSession(QString name);

            //! \brief Uses a base object from a mesh generator.
            void baseObjectFromGenerator();

            //! \brief Loads emboss object from a file.
            void embossObjectFromFile();
            //! \brief Responds to mesh loader.
            //! \todo Actually use this instead of Debug call.
            void embossObjectLoadCompleted();

        private slots:
            //! \brief Accepts the dialog.
            void accept();
            //! \brief Rejects the dialog.
            void reject();

            //! \brief Selects a part by name.
            void selectPart(QString name);

            //! \brief Clears the import dialog.
            void clear();

        private:
            // Base object
            QSharedPointer<Part> m_base_part;
            // Embossing objects
            QVector<QSharedPointer<Part>> m_emboss_parts;
            // Session objects
            QSet<QSharedPointer<Part>> m_session_parts;
            // Scale map
            QMap<QString, QVector3D> m_scales;

            // Selected name
            QString m_selected_name;

            // ----------
            // UI Objects
            // ----------

            //! \brief Setup the static widgets and their layouts.
            void setupUi();

            //! \brief 1. Setup the window properties.
            void setupWindow();
            //! \brief 2. Setup the widgets.
            void setupWidgets();
            //! \brief 3. Setup the layouts and insert their children.
            void setupLayouts();
            //! \brief 4. Setup the insertions for all UI elements.
            void setupInsert();
            //! \brief 5. Setup the events for the various widgets.
            void setupEvents();

            // Layouts
            QGridLayout* m_layout;
            QGridLayout* m_load_layout;
            QHBoxLayout* m_dsb_layout;

            // Labels
            QLabel* m_build_label;
            QLabel* m_emboss_label;
            QLabel* m_base_label;
            QLabel* m_preview_label;

            // Emboss View
            EmbossView* m_preview;

            // Table
            QListWidget* m_part_list;

            // Buttons
            QToolButton* m_add_button;
            QToolButton* m_browse_button;

            // File Select
            QLineEdit* m_file_line;

            // Radios
            QRadioButton* m_load_radio;
            QRadioButton* m_generate_radio;
            QRadioButton* m_session_radio;

            // Parts list
            QListWidget* m_session_list;

            // Scales
            XYZInputWidget* m_base_scale;
            XYZInputWidget* m_part_scale;

            // Button container
            QDialogButtonBox* m_button_container;
    };
} // namespace ORNL

#endif // EMBOSSDIALOG_H
