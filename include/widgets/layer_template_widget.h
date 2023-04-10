#ifndef LAYERTEMPLATEWIDGET_H
#define LAYERTEMPLATEWIDGET_H

// Qt Libraries
#include <QDialog>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QComboBox>
#include <QLabel>
#include <QPushButton>
#include <QLineEdit>
#include <QFileDialog>
#include <QVector>
#include <QSharedPointer>
#include <QPair>
#include <QWidget>
#include <QVector>
#include <QMap>

//Local
#include "managers/settings/settings_manager.h"
#include "windows/dialogs/template_layer_save.h"

namespace ORNL
{

    /*!
     *  \class LayerTemplateWidget
     *  \brief Widget to allow user to select layer bar templates from dropdown, edit and add new templates
     */
    class LayerTemplateWidget : public QWidget
    {
        Q_OBJECT
    public:
        //! \brief Default Constructor
        explicit LayerTemplateWidget(QWidget *parent = nullptr);

        //! \brief Add layer bar templates from settings manager to layer bar.
        void loadLayerBar();

        //! \brief Set additional path for template files.
        void setCurrentFolder(QString path);

        //! \brief Change to a different template.
        void changeLayerBarSettings(QString layer_template);

        //! \brief Opens dialog to create and save new template
        void addNewTemplate();

        //! \brief Opens dialog to edit and save existing template.
        void editTemplate();

    signals:   
        //! \brief Signal that template changed while part selected.
        void layerbarTemplateChanged();

    private:
        //! \brief Setup the static widgets and their layouts.
        void setupUi();

        //! \brief 1. Setup the window properties.
        void setupWindow();

        //! \brief 2. Setup the widgets.
        void setupWidgets();

        //! \brief 3. Setup the layouts and insert their children.
        void setupLayouts();

        //! \brief 6. Setup the events for the various widgets.
        void setupEvents();       

            // -- QActions --
        QVBoxLayout* m_layout;
        QLineEdit* m_search;
        QComboBox* m_templates;
        QLabel* m_current_folder;
        QHBoxLayout* m_buttons;
        QPushButton* m_new;
        QPushButton* m_edit;
        QSpacerItem* m_space;
        QString m_most_recently_edited_template;
    };
}

#endif // LAYERTEMPLATEWIDGET_H
