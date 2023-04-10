#ifndef EXTERNALGRID_H
#define EXTERNALGRID_H

#include <QVector>
#include "geometry/point.h"

namespace ORNL
{
    //! \brief A plain old data structure for recipe maps
    //! Provides each layer a copy of the recipe map used for altering density
    struct RecipeMap
    {
            RecipeMap(int id, double min, double max)
                : m_id(id), m_min(min), m_max(max) {}

            RecipeMap() {}


            //! \brief Id and min/max range for each recipe
            int m_id;
            double m_min;
            double m_max;
    };

    //! \brief A plain old data structure for external grid information.
    //! Holds each layer's individual grid and as well as necessary min and step information for
    //! interpolation into the grid
    struct SingleExternalGridInfo
    {
            SingleExternalGridInfo() {}

            //! \brief Identifies whether or not current layer has had additional parameters set
            //! \return boolean to indicate whether set or not set
            bool isSet() { return m_set; }

            //! \brief Sets additional parameters for a particular layer
            //! \param xMin: min x value
            //! \param xStep: x step size
            //! \param xMax: max x value
            //! \param yMin: min y value
            //! \param yStep: y step size
            //! \param yMax: max y value
            void setParams(double xMin, double xStep, double xMax, double yMin, double yStep, double yMax, QVector<RecipeMap> rMaps)
            {
                m_x_min = xMin;
                m_x_step = xStep;
                m_y_min = yMin;
                m_y_step = yStep;
                m_recipe_maps = rMaps;
                m_set = true;
                int columnSize = (xMax - xMin) / xStep + 1;
                m_grid.resize(columnSize);
                int rowCount = (yMax - yMin) / yStep + 1;
                for(int i = 0; i < columnSize; ++i)
                    m_grid[i].resize(rowCount);
            }

            //! \brief Variables to hold each layer's individual grid, x/y parameters, and recipe map
            QVector<QVector<double>> m_grid;
            double m_x_min, m_y_min;
            double m_x_step, m_y_step;
            QVector<RecipeMap> m_recipe_maps;

            //! \brief Boolean to indicate whether or not above params are set
            bool m_set = false;

            //! \brief Holds object origin to allow alignment between grid and object
            //! Filled in later during slicing when objects relative position is known
            Point m_object_origin;
    };

    //! \brief A plain old data structure for external grid information.
    //! Holds the layer set of grids as well as necessary min and step information for
    //! interpolation into the grid
    struct ExternalGridInfo
    {
        ExternalGridInfo() {}
        ExternalGridInfo(double zMin, double zStep) : m_z_min(zMin), m_z_step(zStep) {}

        //! \brief Vector of grid layers
        QVector<SingleExternalGridInfo> m_grid_layers;

        //! \brief min and step sizes for z to find appropriate layer
        double m_z_min;
        double m_z_step;
    };
}
#endif  // EXTERNALGRID_H
