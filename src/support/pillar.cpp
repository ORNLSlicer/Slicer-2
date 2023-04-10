#include "support/pillar.h"

#include <QSharedPointer>
#include <QString>

//#include "regions/island.h"


namespace ORNL
{
//    Pillar::Pillar(Layer* layer,
//                   QString pillarType,
//                   //Mesh* mesh,
//                   Distance distance)
//    {
//        //m_parent            = mesh;
//        m_start             = layer;
//        m_end               = layer;
//        top                 = NULL;
//        bottom              = NULL;
//        threshhold_distance = distance;
//        if (tappering ||
//            !QString::compare(pillarType, "tappering", Qt::CaseInsensitive))
//            tappering = 1;
//    }

//    void Pillar::addIsland(PolygonList poly, Layer* below_layer)
//    {
//        if (tappering)
//            if (poly.offset(-0.5 * threshhold_distance, ClipperLib2::jtRound)
//                    .outerArea() > threshArea)
//                if (this->islands.size()>0)
//                    /// Start tappering after first support layer
//                    poly = poly.offset(-0.5 * threshhold_distance,
//                                       ClipperLib2::jtRound);
//        //TODO: may not work with skins
//        QSharedPointer< Island > support_island =
//            QSharedPointer< Island >(new Island(below_layer, poly, 0));
//        support_island->is_support = 1;
//        support_island->m_pillar   = this;
//        below_layer->append(support_island);
//        islands.append(support_island);
//        m_end = below_layer;
//    }
//    void Pillar::addTop(Pillar* p, Layer* layer)
//    {
//        top     = p;
//        m_start = layer;
//    }

//    void Pillar::addBottom(Pillar* p, Layer* layer)
//    {
//        bottom = p;
//        m_end  = layer;
//    }
}  // namespace ORNL
