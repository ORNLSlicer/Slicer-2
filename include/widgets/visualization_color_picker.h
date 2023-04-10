#ifndef VISUALIZATION_COLOR_PICKER_H
#define VISUALIZATION_COLOR_PICKER_H

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QGridLayout>
#include <QColorDialog>
#include "managers/preferences_manager.h"

namespace ORNL {
    class VisualizationColorPicker : public QWidget
    {
        Q_OBJECT

        public:
            //! \brief Constructor
            //! \param name of color / type
            //! \param color as QColor
            //! \param parent
            explicit VisualizationColorPicker(QString name, QColor color, QWidget *parent = nullptr);

        private:
            //! \brief Mouse press event
            void mousePressEvent (QMouseEvent*);

            //! \brief Select and set new color value
            void selectSetColor();

            //! \brief Update controls on display
            void updateDisplay();

            //! \brief variable to hold the name
            QString  name;

            //! \brief variable to hold the color
            QColor   color;

            //! \brief UI display control label,
            //! displays the regions name for which color is chosen
            QLabel*      colorTypeLbl;

            //! \brief UI display control label,
            //! displays (This is how it looks!!!) for visulization of color picked
            QLabel*      sampleTextLbl;

            //! \brief UI display button control for reseting color to app default color
            QPushButton* colorResetBtn;

            //! \brief UI display button control for selecting and setting color
            QPushButton* colorSelectorBtn;
    };
}

#endif // VISUALIZATION_COLOR_PICKER_H
