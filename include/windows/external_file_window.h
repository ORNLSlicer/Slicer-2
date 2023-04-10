#ifndef EXTERNALFILEWINDOW_H
#define EXTERNALFILEWINDOW_H

#include <QObject>
#include <QWidget>
#include <QGridLayout>
#include <QLabel>
#include <QPushButton>
#include <QProgressBar>
#include <QVBoxLayout>
#include <QTableWidget>
#include <QFrame>

#include "external_files/external_grid.h"

namespace ORNL
{
    /*!
     * \class ExternalFileWindow
     * \brief Window that displays simple load/processing for external grid files
     */
    class ExternalFileWindow : public QWidget
    {
        Q_OBJECT

        public:
            //! \brief Standard widget constructor
            //! \param parent: Pointer to parent window
            ExternalFileWindow(QWidget* parent);

        signals:
            //! \brief Forward computed grid to relevant manager
            //! \param gridInfo: Structure that holds grid and relevant dimension information
            void forwardGridInfo(ExternalGridInfo gridInfo);

        private slots:
            //! \brief Handler for load button click.  Loads file, starts processing class,
            //! and adjusts UI.
            void loadFile();

        private:
            //! \brief Layout for widget
            QGridLayout *m_layout;

            //! \brief Labels and separators for file info and various grid statistics
            QLabel* m_display_label;
            QLabel* m_grid_info_label, *m_grid_size_label, *m_grid_cell_label;
            QLabel* m_x_cell_label, *m_x_size_label, *m_y_cell_label, *m_y_size_label, *m_z_cell_label, *m_z_size_label;
            QLabel* m_recipe_label;
            QFrame* m_horizontal_sep1, *m_horizontal_sep2;

            //! \brief Table to show recipe mapping
            QTableWidget* m_recipe_table;

            //! \brief Load button
            QPushButton* m_load_button;

            //! \brief Progress bar
            QProgressBar* m_progress_bar;
    };
}

#endif // EXTERNALFILEWINDOW_H
