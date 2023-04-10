#include "support/support.h"

#include <QtMath>
#include <cmath>
#include <iostream>

#include "geometry/point.h"
#include "geometry/polygon.h"
#include "geometry/polygon_list.h"
#include "step/layer/layer.h"

namespace ORNL
{
//    Support::Support(QString str, QSharedPointer< PartBase> m, float angle, Distance layer_height, Area thresh_Area, float ratio)
//    {
//        supportType = str;
//        mesh        = m;
//        layerHeight = layer_height;
//        threshhold_distance = layer_height * qTan(qDegreesToRadians(angle));
//        thresholdArea = thresh_Area;
//        area_ratio = ratio;
//        findOverhang();
//    }

//    void Support::findOverhang()
//    {
//        int total_layer = mesh->getNumLayers(), layer_nr;
//        QSharedPointer< Layer >* layers = mesh->getLayersPointer();
//        for (layer_nr = total_layer - 1; layer_nr > 0; layer_nr--)
//        {
//            // compute the layer indices.
//            int cur_layer = layer_nr;
//            int below_layer = layer_nr - 1;

//            // Extract Details for layer below the current layer
//            QSharedPointer< Island >* islands = layers[below_layer]->data();
//            int islands_below_layer = layers[below_layer]->size();
//            PolygonList below_layer_outline;
//            int island_nr = 0;

//            // compute the outline for the below layer
//            for (island_nr = 0; island_nr < islands_below_layer; island_nr++)
//            {
//                PolygonList poly = (islands[island_nr].data())->getOutlines();
//                //offset below layer outline by threshold distance.
//                poly = poly.offset(0.5*threshhold_distance, ClipperLib2::jtRound);
//                below_layer_outline = below_layer_outline + (poly);
//            }

//            // Extract Details for the  current layer
//            islands = layers[cur_layer]->data();
//            int islands_cur_layer = layers[cur_layer]->size();
//            PolygonList cur_layer_outline;
//            QVector< QSharedPointer< Island > > cur_layer_support;

//            // Calculate the outline for the current layer
//            for (island_nr = 0; island_nr < islands_cur_layer; island_nr++)
//            {
//                // calculate otlines separately for original model and support
//                // outlines. This will help in deciding whether a new pillar
//                // is to be added or the new support island can be added
//                // to an existing island.
//                if (islands[island_nr]->is_support == 1)
//                    cur_layer_support.append(islands[island_nr]);
//                else
//                {
//                    PolygonList poly = islands[island_nr].data()->getOutlines();
//                    cur_layer_outline = cur_layer_outline + (poly);
//                }
//            }
//            // calculate overhang for the original islands of the layer in the
//            // mesh
//            PolygonList original_common = cur_layer_outline & below_layer_outline;
//            PolygonList original_overhang = (cur_layer_outline) - (original_common);

//            QVector< QSharedPointer< Island > > support_overhang_island;
//            QVector< PolygonList > support_overhang;

//            // calculate the overhang for support islands in current layer
//            // separately.
//            // After certain layers, support structure might intersect/touch
//            // a below layer outline and might no longer need support. Hence,
//            // overhang area is calculated for them too.
//            for (QSharedPointer< Island > island : cur_layer_support)
//            {
//                PolygonList poly = island.data()->getOutlines();
//                poly = (poly - (poly & below_layer_outline));
//                // If overhang is present
//                if (!poly.isEmpty())
//                {
//                    // Store the support island having overhang and corresponding
//                    // area in diffrent areas.
//                    // Note: unable to create QMap for the datas tructures
//                    // Hence using two arrays.
//                    support_overhang_island.append(island);
//                    support_overhang.append(poly);
//                }
//            }
//            QVector< PolygonList > pl = original_overhang.splitIntoParts();
//            layers[cur_layer].data()->OverHangArea = pl;

//            if (!QString::compare(supportType, "tree", Qt::CaseInsensitive))
//            {
//                QVector< PolygonList > pl_temp;
//                // First generate patches
//                for (PolygonList plt : pl)
//                {
//                    pl_temp += generatePatch(plt, true);
//                }
//                pl = pl_temp;
//                // process the tree
//                for (treePoint* h : heads)
//                    processTree(h);
//                // generate the tree
//                generateTree(below_layer_outline, layers[below_layer]);
//                layers[cur_layer].data()->OverHangArea = pl;
//            }
//            else if (!QString::compare(supportType, "forest", Qt::CaseInsensitive))
//            {
//                QVector< PolygonList > pl_temp;
//                // First generate patches
//                for (PolygonList plt : pl)
//                {
//                    pl_temp += generatePatch(plt, false);
//                }
//                pl = pl_temp;
//                layers[cur_layer].data()->OverHangArea = pl;
//                // Generate dropdown supports for the patches generated
//                generateSupport(layers[cur_layer], layers[below_layer], support_overhang_island, support_overhang);
//            }
//            else if (!QString::compare(supportType, "tappering", Qt::CaseInsensitive))
//                generateSupport(layers[cur_layer], layers[below_layer], support_overhang_island, support_overhang);
//            // DropDown Support structure
//            else
//                generateSupport(layers[cur_layer], layers[below_layer], support_overhang_island, support_overhang);
//        }
//    }
//    void Support::generateTree(PolygonList layer_outline, QSharedPointer< Layer > layer)
//    {
//        treePoint* par;
//        QVector< treePoint* > delete_leaf;
//        QVector< PolygonList > poly;
//        PolygonList p;
//        QVector< treePoint* > parents;
//        // Loop until there are leaves in the tree.
//        for (treePoint* t : leaves)
//        {
//            par = t->parent;

//            if (par != nullptr)
//            {
//                if (std::abs(t->p.distance(par->p)()) ==0)
//                {
//                    // Once merged prepare the leaf for deletion from the tree
//                    delete_leaf.append(t);
//                    par->pl += t->pl;
//                }
//                // If the distance to shift is less than actual distance between
//                // the points, adjust the distance
//                if (t->shift.distance(Distance2D(0, 0))() > std::abs(t->p.distance(par->p)()))
//                {
//                    t->shift = (par->p - t->p);
//                }
//            }

//            float intrsct = t->pl.commonArea(layer_outline);
//            // Manage tree structure outiline if it intersects with the original
//            //  outline of mesh
//            if (intrsct < 0.9999)
//            {
//                t->pl = t->pl - (t->pl & layer_outline);
//            }
//            else
//            {
//                // If completely intersects with the outline, prepare leaf
//                // for deletion
//                delete_leaf.append(t);
//                continue;
//            }
//            // calculate the polygonlist outline of the islands to be added
//            // to the layer
//            p += t->pl;
//            t->pl = t->pl.shift(t->shift);
//            t->p  = t->p + t->shift;

//        }
//        for (treePoint* i : delete_leaf)
//        {// Delete leaf and make its parent the new leaf node for shifting and
//         // merging at subsequent branching point
//            if (i->parent && leaves.indexOf(i->parent) == -1)
//            {
//                leaves.append(i->parent);
//                i->parent->child.remove(i->parent->child.indexOf(i));
//            }
//            leaves.remove(leaves.indexOf(i));
//        }
//        delete_leaf.clear();

//        poly = p.splitIntoParts();
//        // add Islands to the layer
//        for (PolygonList pl : poly)
//        {
//            (new Pillar(layer.data(), "tree", threshhold_distance))->addIsland(pl, layer.data());
//        }
//    }

//    void Support::generateSupport(QSharedPointer< Layer > cur_layer, QSharedPointer< Layer > below_layer,
//        QVector< QSharedPointer< Island > > support_island_overhang, QVector< PolygonList > support_overhang)
//    {
//        // generate support for overhang in islands present in the original
//        // outline of the current layer. add a new pillar.
//        if (!cur_layer->OverHangArea.isEmpty())
//        {
//            QVector< PolygonList > pl = cur_layer->OverHangArea;
//            for (PolygonList poly : pl)
//        // Add new pillar which would internally add a support Island.
//                addNewPillar(poly.offset(-0.5*threshhold_distance,ClipperLib2::jtRound),below_layer);
//        }
//        int itr = support_island_overhang.size(), i;
//        for (i = 0; i < itr; i++)
//        {
//            Island* cur_support = support_island_overhang[i].data();
//            Pillar* cur_support_pillar = cur_support->m_pillar;
//            PolygonList o_hang = support_overhang[i];
//            QVector< PolygonList > hang = o_hang.splitIntoParts();
//            // If the current support island is the only overhanging part.
//            if (hang.size() == 1)
//                // if it is a complete overhang with one overhanging add it
//                // to the current pillar
//                if (o_hang == cur_support->getOutlines())
//                    cur_support->m_pillar->addIsland(o_hang, below_layer.data());
//                // if it is a partial overhang of the original support island
//                // add a new pillar
//                else
//                    addNewPillar(o_hang, below_layer, cur_support_pillar);
//            // if it has a lot of partial overhanging areas.
//            // add multiple new pillars
//            else if (hang.size() > 1)
//                for (PolygonList& part_hang : hang)
//                    addNewPillar(part_hang, below_layer, cur_support_pillar);
//        }
//    }


//    void Support::addNewPillar(PolygonList supportAreas, QSharedPointer< Layer > below_layer,
//                               Pillar* cur_support_pillar)
//    {
//        Pillar* p = new Pillar(below_layer.data(), this->supportType, threshhold_distance);
//        /// In case of tappering support structure set the threshold area.
//        if (!QString::compare(supportType, "tappering", Qt::CaseInsensitive))
//            p->threshArea = area_ratio * supportAreas.outerArea();

//        p->addIsland(supportAreas, below_layer.data());
//        pillars.append(QSharedPointer< Pillar >(p));
//        if (cur_support_pillar)
//        {
//            cur_support_pillar->addBottom(p, below_layer.data());
//            p->addTop(cur_support_pillar, below_layer.data());
//        }
//    }

//    void Support::processTree(treePoint* head)
//    {
//        int ind = 0;
//        QVector< treePoint* > q;
//        q.append(head);
//        q.append(nullptr);
//        int add    = 0;
//        int chk    = 1;
//        float maxd = 0;
//        float max_time;
//        float u_max_time;
//        // Future work: add feature for splitting trees in case of support present after
//        // certain layer
//        while (chk)
//        {
//            treePoint* t = q[ind];
//            if (t != nullptr)
//            {
//                if (t->parent)
//                {   // If it is not the root of the tree. Update the distance it
//                    // has to shift every loop to merge at branching point at
//                    // the same time as other leaves that merge at the point
//                    float speed   = (t->p.distance(t->parent->p)() / u_max_time);
//                    Point diff    = t->parent->p - t->p;
//                    float vecx    = diff.x();
//                    float vecy    = diff.y();
//                    float factor  = 1 / (qSqrt(qPow(vecx, 2) + qPow(vecy, 2)));
//                    //dist_vec is the unit vector in the direction of shifting
//                    Point dist_vec(speed * vecx * factor, speed * vecy * factor);
//                    t->shift = dist_vec;
//                }
//                for (treePoint* p : t->child)
//                    if (p != nullptr)
//                    {   // Determine the leaf with the maximum distance from the
//                        // branching point
//                        if (maxd < p->p.distance(t->p)())
//                            maxd = p->p.distance(t->p)();
//                        q.append(p);
//                        add = 1;
//                    }
//                    // Move maximum of half of the threshold distance
//                max_time = maxd / (0.5 * threshhold_distance)();
//            }
//            else
//            {
//                u_max_time = max_time;
//                maxd       = maxd * 0.0;
//                if (add)
//                {
//                    q.append(nullptr);
//                    add = 0;
//                }
//                else
//                    chk = 0;
//            }
//            ind++;
//        }
//    }

//    QVector< PolygonList > Support::generatePatch(PolygonList pList, bool tree_support)
//    {
//        Polygon outer;
//        QVector< Polygon > inner;
//        QVector< Polygon > selected_inner;
//        // find the inner and outer polygons
//        for (Polygon pgon : pList)
//            if (pgon.orientation())
//                outer = pgon;
//            else
//                inner.append(pgon);

//        QVector< PolygonList > pl;
//        QVector< treePoint* > nodes;
//        PolygonList copy_plist = pList;
//        // Find Extremities of the bounding box of the polygon
//        Point lb               = pList.min();
//        Point ub               = pList.max();
//        int x1                 = lb.x();
//        int x2                 = ub.x();
//        int y1                 = lb.y();
//        int y2                 = ub.y();
//        float i, j;
//        float ratio = 1;
//        float area  = qCeil(qSqrt(thresholdArea()));
//        float thres = threshhold_distance();
//        //float dim = area  + thres;

//        float dim = 2 * area  +  thres;

//        // Loop over the grid of the bounding box
//        for (j = y1 + thres; j <= y2; j+=dim)
//        {
//            for (i = x1 + thres ; i <= x2; i++)
//            {
//                // Create the extremities of a possible patch
//                QVector< Point > p;
//                p.append(Distance2D(i, j));
//                p.append(Distance2D(i + dim, j));
//                p.append(Distance2D(i + dim, j + dim));
//                p.append(Distance2D(i, j + dim));

//                Polygon temp_p(p);
//                PolygonList temp_pl;
//                temp_pl += temp_p;
//                PolygonList intrsct = pList & temp_pl;
//                // Select the patch if it has intersection with the original polygon
//                // of more than 50%

//               // if ((temp_pl - (intrsct)).isEmpty() )
//                if ((temp_pl - (intrsct)).isEmpty() ||
//                    (intrsct).netArea() > 0.5 * (temp_pl).netArea())
//                {

//                    pl.append(intrsct);
//                    // Fill in treePoint Data structure if tree structure is enabled
//                    if (tree_support)
//                    {
//                        treePoint* n = new treePoint();
//                        n->p         = temp_p.centerOfMass();
//                        n->parent    = nullptr;
//                        n->leaf      = 1;
//                        n->pl        = intrsct.offset(-0.5 * threshhold_distance());
//                        leaves.append(n);
//                        nodes.append(n);
//                    }
//                    i = i + dim - 1;
//                    pList -= intrsct;
//                }
//            }
//            // Create the tree from leaf nodes if tree support structure
//            // is enabled
//            if (tree_support && nodes.size() > 0)
//            {
//                QVector< treePoint* > tmp;
//                while (nodes.size() > 1)
//                {
//                    int itr, limit = nodes.size();
//                    int dif = limit % 2;
//                    limit -= dif;
//                   // std::cout << dif << "here2too" << limit << "\n";
//                    for (itr = 0; itr < limit; itr += 2)
//                    {
//                        treePoint* par = new treePoint();
//                        treePoint* a   = nodes[itr];
//                        treePoint* b   = nodes[itr + 1];
//                        a->parent      = par;
//                        par->child.append(a);
//                        b->parent = par;
//                        par->child.append(b);
//                        par->p      = (a->p + b->p) / 2;
//                        par->parent = nullptr;
//                        tmp.append(par);
//                    }
//                    if (dif)
//                    {
//                        treePoint* par = new treePoint();
//                        treePoint* a   = nodes[limit - 1];
//                        treePoint* b   = nodes[limit];
//                        b->parent      = par;
//                        par->child.append(b);
//                        par->p      = (a->p + b->p) / 2;
//                        par->parent = nullptr;
//                        tmp.append(par);
//                    }
//                    nodes.clear();
//                    nodes = tmp;
//                    tmp.clear();
//                   // std::cout << "limit" << nodes.size() << "\n";
//                }
//                if (nodes.size() == 1)
//                    heads.append(nodes[0]);
//                nodes.clear();
//            }
//        }
//        return pl;
//    }

}  // end namespace ORNL
