#ifndef SETTINGS_POLYGON_H
#define SETTINGS_POLYGON_H

// Locals
#include "geometry/polygon_list.h"
#include "geometry/plane.h"
#include "configs/settings_base.h"

namespace ORNL
{
    /*!
     * \class SettingsPolygon
     * \brief is a polygon that contains a settings. It is a cross-section of a settings mesh with part range info
     */
    class SettingsPolygon : public PolygonList
    {
        public:
            //! \brief Constructor
            SettingsPolygon() = default;

            //! \brief Constructor to cross section a setting's part with a given plane and save it's SettingsBase
            //! \param geometery: cross-section of the settings mesh
            //! \param sb: the settings
            SettingsPolygon(QVector<Polygon> geometry, QSharedPointer<SettingsBase>& sb);

            //! \brief clips a line with this polygon
            //! \param start: the start of the line
            //! \param end: the end of the line
            //! \return a list of points were the line intersected the polygon
            QVector<Point> clipLine(Point start, Point end);

            //! \brief gets the settings base
            //! \return a pointer to the settings
            QSharedPointer<SettingsBase> getSettings();

        private:
            //! \brief the settings
            QSharedPointer<SettingsBase> m_sb;
    };
}

#endif // SETTINGS_POLYGON_H
