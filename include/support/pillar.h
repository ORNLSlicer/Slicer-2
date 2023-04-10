#ifndef PILLAR_H
#define PILLAR_H

#include <QSharedPointer>
#include <QString>
#include <QVector>

//#include "geometry/mesh.h"
//#include "regions/island.h"
//#include "regions/layer.h"
//#include "units/unit.h"

//namespace ORNL
//{
//    class Island;

//    /**
//     * \class Pillar
//     *
//     * \brief Contains the properties of a pillar in the support structure. A
//     * support structure can have multiple pillars.
//     */
//    class Pillar
//    {
//    private:
//        int tappering = 0;
//        Distance threshhold_distance;


//        /********review and change********/
//        // int area;
//        /*TODO TODO add other other properties TODO TODO*/

//        //Mesh* m_parent;
//        Layer* m_start;
//        Layer* m_end;
//        Pillar* top;
//        Pillar* bottom;
//        QVector< QSharedPointer< Island > > islands;

//    public:
//        Area threshArea;

//        /**
//         * \brief Constructor
//         */
//        Pillar(Layer* layer, QString pillarType, Distance distance);//Mesh* mesh, Distance distance);

//        /**
//         * \brief Add an island layer to the pillar
//         * \param1 Island to be added
//         * \param2 Layer where the island is to be added.
//         */
//        void addIsland(PolygonList poly, Layer* below_layer);
//        /**
//         * \brief add a pillar on top of the current pillar
//         * \param pillar to be added on top
//         */
//        void addTop(Pillar* p, Layer* layer);
//        /**
//         * \brief add a pillar at bottom of the current pillar
//         * \param pillar to be added on bottom
//         */
//        void addBottom(Pillar* p, Layer* layer);
//    };
//}  // namespace ORNL
#endif  // PILLAR_H
