#ifndef THEME_MANAGER_H
#define THEME_MANAGER_H

#include <QWidget>
#include <QColor>

#include "utilities/enums.h"

namespace ORNL
{
    /*!
     * \class Theme
     * \brief holds the path of the theme styles as well as colors for drawn objects
     */
    class Theme
    {
    public:
        //! \brief Constructor
        Theme(int themeNum);

        //! \brief sets the values of the current theme according to theme choice
        //! \param themeNum: number corresponding to desired theme (should be taken from ThemeName enumerator)
        void chooseTheme(int themeNum);

        //! \brief sets the path of the current theme
        //! \param path: the path to the current theme's folder
        void setPath(QString path);

        //! \brief sets the dot color of the current theme
        //! \param r: red value of desired color (0-255)
        //! \param g: green value of desired color (0-255)
        //! \param b: blue valued of desired color (0-255)
        //! \param a: alpha value of desired color (0-255)
        //! \param color: a QColor; can use preset color names
        void setDotColor(int r, int g, int b, int a = 255);
        void setDotColor(QColor color);

        //! \brief sets the selected dot color of the current theme
        //! \param r: red value of desired color (0-255)
        //! \param g: green value of desired color (0-255)
        //! \param b: blue valued of desired color (0-255)
        //! \param a: alpha value of desired color (0-255)
        //! \param color: a QColor; can use preset color names
        void setDotSelectedColor(int r, int g, int b, int a = 255);
        void setDotSelectedColor(QColor color);

        //! \brief sets the dot hover color of the current theme
        //! \param r: red value of desired color (0-255)
        //! \param g: green value of desired color (0-255)
        //! \param b: blue valued of desired color (0-255)
        //! \param a: alpha value of desired color (0-255)
        //! \param color: a QColor; can use preset color names
        void setDotHoverColor(int r, int g, int b, int a = 255);
        void setDotHoverColor(QColor color);

        //! \brief sets the dot prio color of the current theme
        //! \param r: red value of desired color (0-255)
        //! \param g: green value of desired color (0-255)
        //! \param b: blue valued of desired color (0-255)
        //! \param a: alpha value of desired color (0-255)
        //! \param color: a QColor; can use preset color names
        void setDotPrioColor(int r, int g, int b, int a = 255);
        void setDotPrioColor(QColor color);

        //! \brief sets the color of the line connecting paired dots in the current theme
        //! \param r: red value of desired color (0-255)
        //! \param g: green value of desired color (0-255)
        //! \param b: blue valued of desired color (0-255)
        //! \param a: alpha value of desired color (0-255)
        //! \param color: a QColor; can use preset color names
        void setDotPairedColor(int r, int g, int b, int a = 255);
        void setDotPairedColor(QColor color);

        //! \brief sets the grouped dot color of the current theme
        //! \param r: red value of desired color (0-255)
        //! \param g: green value of desired color (0-255)
        //! \param b: blue valued of desired color (0-255)
        //! \param a: alpha value of desired color (0-255)
        //! \param color: a QColor; can use preset color names
        void setDotGroupedColor(int r, int g, int b, int a = 255);
        void setDotGroupedColor(QColor color);

        //! \brief sets the dot label color of the current theme
        //! \param r: red value of desired color (0-255)
        //! \param g: green value of desired color (0-255)
        //! \param b: blue valued of desired color (0-255)
        //! \param a: alpha value of desired color (0-255)
        //! \param color: a QColor; can use preset color names
        void setDotLabelColor(int r, int g, int b, int a = 255);
        void setDotLabelColor(QColor color);

        //! \brief sets the view background color of t he current theme
        //! \param r: red value of desired color (0.00 - 1.00; extended RGB )
        //! \param g: green value of desired color (0.00 - 1.00; extended RGB )
        //! \param b: blue valued of desired color (0.00 - 1.00; extended RGB )
        //! \param a: alpha value of desired color (0.00 - 1.00; extended RGB )
        //! \param color: a QColor; can use preset color names
        void setBgColor(double r, double g, double b, double a = 1.0);

        //! \brief sets the layerbar major line color of the current theme
        //! \param r: red value of desired color (0-255)
        //! \param g: green value of desired color (0-255)
        //! \param b: blue valued of desired color (0-255)
        //! \param a: alpha value of desired color (0-255)
        //! \param color: a QColor; can use preset color names
        void setLayerbarMajorColor(int r, int g, int b, int a = 255);
        void setLayerbarMajorColor(QColor color);

        //! \brief sets the layerbar minor line color of the current theme
        //! \param r: red value of desired color (0-255)
        //! \param g: green value of desired color (0-255)
        //! \param b: blue valued of desired color (0-255)
        //! \param a: alpha value of desired color (0-255)
        //! \param color: a QColor; can use preset color names
        void setLayerbarMinorColor(int r, int g, int b, int a = 255);
        void setLayerbarMinorColor(QColor color);

        //! \brief return color values
        QVector<QColor> getDotColors();
        QColor getDotPrioColor();
        QColor getDotPairedColor();
        QColor getLayerbarMajorColor();
        QColor getLayerbarMinorColor();
        QVector<double> getBgColor();

        //! \brief returns the folder path of the current theme
        QString getFolderPath();

    signals:


    private:
        //! folder path
        QString m_folder_path;

        //! Colors
        QVector<double> m_bgColor;
        QVector<QColor> m_dotColors;
        QColor m_dotColor_paired;
        QColor m_dotColor_prio;
        QColor m_lineColor_major;
        QColor m_lineColor_minor;
    };
}

#endif // THEME_MANAGER_H
