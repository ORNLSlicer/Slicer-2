#if 0
//
// FelkelStraightSkeleton.cpp: implementation of the vtStraightSkeleton class.
//
// Copyright (c) 2003-2009 Virtual Terrain Project
// Free for all uses, see license.txt for details.
//
// Straight skeleton algorithm and original implementation
// courtesy of Petr Felkel and Stepan Obdrzalek (petr.felkel@tiani.com)
// Re-implemented for the Virtual Terrain Project (vterrain.org)
// by Roger James (www.beardandsandals.co.uk)
//

#include "FelkelStraightSkeleton.h"
#include "../Skeleton.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
namespace StraightSkeleton
{
	/**
	 *	Method to determine the intersection point between a ray and a line segment. The function return true and sets intersection if an intersection is found. 
	 *  The method was copied from http://pastebin.com/f28510ac9/.
	 */
	bool RayToLineSegmentIntersection(ClipperLib::DoublePoint rayBegin, ClipperLib::DoublePoint rayEnd, ClipperLib::DoublePoint lineBegin, ClipperLib::DoublePoint lineEnd, ClipperLib::DoublePoint &intersection)
	{
		float r, s;
		float d;

		//Make sure the lines aren't parallel
		ClipperLib::DoublePoint vertex1to2 = rayEnd - rayBegin;
		ClipperLib::DoublePoint vertex3to4 = lineEnd - lineBegin;
		if ((double)vertex1to2.Y / vertex1to2.X != (double)vertex3to4.Y / vertex3to4.X)
		{
			d = (float) (vertex1to2.X * vertex3to4.Y - vertex1to2.Y * vertex3to4.X);
			if (d != 0)
			{
				ClipperLib::DoublePoint vertex3to1 = rayBegin - lineBegin;
				r = (float) ((vertex3to1.Y * vertex3to4.X - vertex3to1.X * vertex3to4.Y) / d);
				s = (float) ((vertex3to1.Y * vertex1to2.X - vertex3to1.X * vertex1to2.Y) / d);

				if (r >= 0)
				{
					if (s >= 0 && s <= 1)
					{
						intersection = rayBegin + (rayEnd - rayBegin) * r;
						return true;
					}
				}
			}
		}
		return false;
	}

	vtStraightSkeleton::vtStraightSkeleton()
	{

	}

	vtStraightSkeleton::~vtStraightSkeleton()
	{

	}

	CSkeleton& vtStraightSkeleton::MakeSkeleton(ContourVector &contours)
	{
		try
		{
			while (m_iq.size())
				m_iq.pop();
			m_vl.erase(m_vl.begin(), m_vl.end());
			m_skeleton.erase(m_skeleton.begin(), m_skeleton.end());
			m_boundaryedges.erase(m_boundaryedges.begin(), m_boundaryedges.end());

			for (size_t ci = 0; ci < contours.size(); ci++)
			{
				Contour &points = contours[ci];

				Contour::iterator first = points.begin();
				if (first == points.end())
					break;

				Contour::iterator next = first;

				while (++next != points.end())
				{
					if (*first == *next)
						points.erase(next);
					else
						first = next;
					next = first;
				}

				int s = (int) points.size();
				CVertexList::iterator start = m_vl.end();
				CVertexList::iterator from = start;
				CVertexList::iterator to = start;

				for (int f = 0; f <= s; f++)
				{
					if (0 == f)
					{
						m_vl.push_back(CVertex(points[0].m_Point, points[s - 1].m_Point, points[s - 1].m_Slope, points[1].m_Point, points[0].m_Slope));
						to = m_vl.end();
						to--;
						start = to;
					}
					else if (f == s)
					{
						from = to;
						to = start;
						m_boundaryedges.push_front(CSkeletonLine(*from, *to));
					}
					else
					{
						from = to;
						m_vl.push_back(CVertex(points[f].m_Point, points[f - 1].m_Point, points[f - 1].m_Slope, points[(f + 1) % s].m_Point, points[f].m_Slope));
						to = m_vl.end();
						to--;
						m_boundaryedges.push_front(CSkeletonLine(*from, *to));
					}
				}
			}

			m_NumberOfBoundaryVertices = (int) m_vl.size();
			m_NumberOfBoundaryEdges = (int) m_boundaryedges.size();

			if (m_vl.size() < 3)
			{
				std::string str = "Polygon too small\n";
				throw str;
			}

			CVertexList::iterator i;

			size_t vn = 0, cn = 0;

			CVertexList::iterator contourBegin;

			for (i = m_vl.begin(); i != m_vl.end(); i++)
			{
				(*i).m_prevVertex = &*m_vl.prev(i);
				(*i).m_nextVertex = &*m_vl.next(i);
				(*i).m_leftVertex = &*i;
				(*i).m_rightVertex = &*i;
				if (vn == 0)
					contourBegin = i;
				if (vn == contours[cn].size() - 1)
				{
					(*i).m_nextVertex = &*contourBegin;
					(*contourBegin).m_prevVertex = &*i;
					vn = 0;
					cn++;
				}
				else
					vn++;
			}


#ifdef FELKELDEBUG
			VTLOG("Building initial intersection queue\n");
#endif
			for (i = m_vl.begin(); i != m_vl.end(); i++)
			{
				if (!(*i).m_done)
				{
					CIntersection is(m_vl, *i);
					if (is.m_height != CN_INFINITY)
						m_iq.push(is);
				}
			}

#ifdef FELKELDEBUG
			VTLOG("Processing intersection queue\n");
#endif
			while (m_iq.size())
			{
				CIntersection i = m_iq.top();

				m_iq.pop();

#ifdef FELKELDEBUG
				VTLOG("Processing %d %d left done %d right done %d\n",
					i.m_leftVertex->m_ID, i.m_rightVertex->m_ID, i.m_leftVertex->m_done, i.m_rightVertex->m_done);
#endif
				if ((NULL == i.m_leftVertex) || (NULL == i.m_rightVertex))
				{
					string str = "Invalid intersection queue entry\n";
					throw str;
				}
				if (i.m_leftVertex->m_done && i.m_rightVertex->m_done)
					continue;
				if (i.m_leftVertex->m_done || i.m_rightVertex->m_done)
				{
					if (!i.m_leftVertex->m_done)
						m_iq.push(CIntersection(m_vl, *i.m_leftVertex));
					if (!i.m_rightVertex->m_done)
						m_iq.push(CIntersection(m_vl, *i.m_rightVertex));
					continue;
				}

#ifdef FELKELDEBUG
				if (!(i.m_leftVertex->m_prevVertex != i.m_rightVertex))
					VTLOG("%s %d Assert failed\n", __FILE__, __LINE__);
				if (!(i.m_rightVertex->m_nextVertex != i.m_leftVertex))
					VTLOG("%s %d Assert failed\n", __FILE__, __LINE__);
#endif
				if (i.m_type == CIntersection::CONVEX)
					if (i.m_leftVertex->m_prevVertex->m_prevVertex == i.m_rightVertex || i.m_rightVertex->m_nextVertex->m_nextVertex == i.m_leftVertex)
						i.ApplyLast3(m_skeleton, m_vl);
					else
						i.ApplyConvexIntersection(m_skeleton, m_vl, m_iq);
				if (i.m_type == CIntersection::NONCONVEX)
					i.ApplyNonconvexIntersection(m_skeleton, m_vl, m_iq, cn == 1);
			}

#ifdef FELKELDEBUG
			Dump();
#endif

			FixSkeleton();

#ifdef FELKELDEBUG
			Dump();
#endif
		}
		catch (string str)
		{
			m_skeleton.erase(m_skeleton.begin(), m_skeleton.end());
			fprintf(stderr, "%s", str.c_str());
		}

		return m_skeleton;
	}

	CSkeleton& vtStraightSkeleton::MakeSkeleton(Contour &points)
	{
		ContourVector vv;

		vv.push_back(points);

		return MakeSkeleton(vv);
	}

	ORNLengine::Skeleton *vtStraightSkeleton::MakeSkeleton(ORNLengine::PolygonRef& polygon, bool connectToPoly)
	{
		ORNLengine::Polygons polygons;

		polygons.add(polygon);
		
		return MakeSkeleton(polygons);
	}

	ORNLengine::Skeleton *vtStraightSkeleton::MakeSkeleton(ORNLengine::Polygons polygons, bool connectToPoly)
	{
		ORNLengine::Skeleton *skeleton;
		ContourVector contours;

		// Create the contours from the polygons
		for (unsigned int polygonindex = 0; polygonindex < polygons.size(); polygonindex++)
		{
			Contour conversion;
			ORNLengine::PolygonRef polygon = polygons[polygonindex];

			for (int i = polygon.size() - 1; i >= 0; i--)
			{
				conversion.push_back(CEdge((double) polygon[i].X, 0, (double) polygon[i].Y, M_PI / 4));
			}

			contours.push_back(conversion);
		}

		// Create the skeleton to return
		skeleton = new ORNLengine::Skeleton();
		CSkeleton cSkeleton = MakeSkeleton(contours);

		string textLine = "";

		for (CSkeletonLine line : cSkeleton)
		{
			textLine += "\\draw[blue] (";

			textLine += to_string(line.m_lower.m_vertex->m_point.m_x / 1000.0) + ", " +
				to_string(line.m_lower.m_vertex->m_point.m_z / 1000.0) + ") -- (" +
				to_string(line.m_higher.m_vertex->m_point.m_x / 1000.0) + ", " +
				to_string(line.m_higher.m_vertex->m_point.m_z / 1000.0) + ");";

			textLine += "\n";
		}

		textLine;

		// Add the lines from the old skeleton to the new skeleton
		for (CSkeletonLine line : cSkeleton)
		{
			Point p1;
			Point p2;
			bool intersect;
			bool containedP1;
			bool containedP2;

			// Get the point data from the line
			p1.X = (ClipperLib::cInt) line.m_lower.m_vertex->m_point.m_x;
			p1.Y = (ClipperLib::cInt) line.m_lower.m_vertex->m_point.m_z;
			p2.X = (ClipperLib::cInt) line.m_higher.m_vertex->m_point.m_x;
			p2.Y = (ClipperLib::cInt) line.m_higher.m_vertex->m_point.m_z;

			// Check to see if the line borders the polygon
			intersect = false;
			containedP1 = false;
			containedP2 = false;
			for (unsigned int polygonindex = 0; polygonindex < polygons.size(); polygonindex++)
			{
				ORNLengine::PolygonRef polygon = polygons[polygonindex];

				// Determine if the point intersects with the polygon.
				for (unsigned int i = 0; i < polygon.size() && !intersect; i++)
				{
					if ((polygon[i].X == p1.X && polygon[i].Y == p1.Y) || (polygon[i].X == p2.X && polygon[i].Y == p2.Y))
					{
						intersect = true;
					}
				}

				// Determine if the point is contained in the polygon.
				if (polygon.inside(p1)) containedP1 = true;
				if (polygon.inside(p2)) containedP2 = true;

			}
			// Only add lines that do not border the polygon
			if (!intersect && containedP1 && containedP2)
			{
				skeleton->AddSkeletonEdge(p1, p2);
			}
		}

		// Add connection from the skeleton to the polygon
		if (connectToPoly)
		{
			// Remove small segments
			skeleton->CombineShortEdges();

			DoublePoint nearestIntersection;
			int64_t intersectionDistance = 0;
			bool intersectionFound = false;
			int nodeNum = (int) skeleton->m_nodes.size();

			// For each of the endpoints
			for (int idx = 0; idx < nodeNum; idx++)
			{
				if (skeleton->m_nodes[idx]->getIsEdge())
				{
					intersectionFound = false;

					for (unsigned int polygonindex = 0; polygonindex < polygons.size(); polygonindex++)
					{
						ORNLengine::PolygonRef polygon = polygons[polygonindex];

						for (unsigned int polygonSeg = 0; polygonSeg < polygon.size(); polygonSeg++)
						{
							DoublePoint intersection;
							if (skeleton->m_nodes[idx]->m_edges.size() == 0) continue;
							if (RayToLineSegmentIntersection(skeleton->m_nodes[idx]->m_edges[0]->m_point.to_doublePoint(), skeleton->m_nodes[idx]->m_point.to_doublePoint(), polygon[polygonSeg].to_doublePoint(), polygon[(polygonSeg + 1) % polygon.size()].to_doublePoint(), intersection))
							{
								// Calculate the distance to calculate the nearest intersection.
								int64_t distance = vSize(skeleton->m_nodes[idx]->m_point - intersection.to_IntPoint());

								// If the intersection has not been found yet or the intersection is closer then use then point
								if (!intersectionFound || distance < intersectionDistance)
								{
									nearestIntersection = intersection;
									intersectionFound = true;
									intersectionDistance = distance;
								}
							}
						}
					}

					// Add connection from skeleton to polygon
					if (intersectionFound)
						skeleton->AddSkeletonEdge(skeleton->m_nodes[idx]->m_point, nearestIntersection.to_IntPoint());
				}
			}
		}

		return skeleton;
	}

	CSkeleton vtStraightSkeleton::CompleteWingedEdgeStructure(ContourVector &contours)
	{
		// Save current skeleton
		int iOldSize = (int) m_skeleton.size();
		int i;
		CSkeleton::iterator si;

		for (size_t ci = 0; ci < contours.size(); ci++)
		{
			Contour& points = contours[ci];
			for (size_t pi = 0; pi < points.size(); pi++)
			{
				C3DPoint& LowerPoint = points[pi].m_Point;
				C3DPoint& HigherPoint = points[(pi + 1) % points.size()].m_Point;


				// Find a matching empty lower left
				for (i = 0, si = m_skeleton.begin(); i < iOldSize; i++, si++)
				{
					CSkeletonLine& Line = *si;
					if ((Line.m_lower.m_vertex->m_point == LowerPoint) && (Line.m_lower.LeftID() == -1))
						break;
				}
				if (i == iOldSize)
				{
					fprintf(stderr, "CompleteWingedEdgeStructure - Failed to find matching empty lower left\n");
					return CSkeleton();
				}
				CSkeletonLine& OldLowerLeft = *si;

				// Find a matching empty lower right
				for (i = 0, si = m_skeleton.begin(); i < iOldSize; i++, si++)
				{
					CSkeletonLine& Line = *si;
					if ((Line.m_lower.m_vertex->m_point == HigherPoint) && (Line.m_lower.RightID() == -1))
						break;
				}
				if (i == iOldSize)
				{
					fprintf(stderr, "CompleteWingedEdgeStructure - Failed to find matching empty lower right\n");
					return CSkeleton();
				}
				CSkeletonLine& OldLowerRight = *si;

				m_skeleton.push_back(CSkeletonLine(*OldLowerLeft.m_lower.m_vertex, *OldLowerRight.m_lower.m_vertex));

				CSkeletonLine& NewEdge = m_skeleton.back();

				NewEdge.m_lower.m_right = &OldLowerLeft;
				OldLowerLeft.m_lower.m_left = &NewEdge;
				NewEdge.m_higher.m_left = &OldLowerRight;
				OldLowerRight.m_lower.m_right = &NewEdge;
			}
		}
#ifdef FELKELDEBUG
		Dump();
#endif
		return m_skeleton;
	}

	void vtStraightSkeleton::FixSkeleton()
	{
		// Search the skeleton list for consecutive pairs of incorrectly linked lines
		CSkeleton::iterator s1 = m_skeleton.begin();
		for (unsigned int i = 0; i < m_skeleton.size() - 2; i++, s1++)
		{
			CSkeletonLine& Lower = *s1++;
			CSkeletonLine& Higher1 = *s1++;
			CSkeletonLine& Higher2 = *s1;

			if ((Higher1.m_higher.RightID() == -1) &&
				(Higher1.m_lower.LeftID() == -1) &&
				(Higher2.m_higher.LeftID() == -1) &&
				(Higher2.m_lower.RightID() == -1) &&
				(Higher1.m_higher.VertexID() == Higher2.m_lower.VertexID()) &&
				(Higher1.m_lower.VertexID() == Higher2.m_higher.VertexID())) // I don't think I cam make this test much tighter !!!
			{
				CSkeletonLine* pLeft = Lower.m_higher.m_left;
				CSkeletonLine* pRight = Lower.m_higher.m_right;
				const CVertex* pVertex = Lower.m_higher.m_vertex;
				if ((NULL == pLeft) || (NULL == pRight) || (NULL == pVertex))
				{
					string str = "Problem fixing skeleton\n";
					throw str;
				}
				// Fix up the left side
				if ((pLeft->m_lower.VertexID() == pVertex->m_ID) || (pLeft->m_lower.VertexID() == pVertex->m_ID + 1))
				{
					// Fix up lower end
					pLeft->m_lower.m_vertex = pVertex;
					pLeft->m_lower.m_left = pRight;
					if (pLeft->m_lower.RightID() != Lower.m_ID)
					{
						string str = "Left Lower Right ID != Lower ID\n";
						throw str;
					}
				}
				else if ((pLeft->m_higher.VertexID() == pVertex->m_ID) || (pLeft->m_higher.VertexID() == pVertex->m_ID + 1))
				{
					// Fix up upper end
					pLeft->m_higher.m_vertex = pVertex;
					pLeft->m_higher.m_left = pRight;
					if (pLeft->m_higher.RightID() != Lower.m_ID)
					{
						string str = "Left Higher Right ID != Lower ID\n";
						throw str;
					}
				}
				else
				{
					string str = "Problem fixing left side\n";
					throw str;
				}
				// Fix up the right side
				if ((pRight->m_lower.VertexID() == pVertex->m_ID) || (pRight->m_lower.VertexID() == pVertex->m_ID + 1))
				{
					// Fix up lower end
					pRight->m_lower.m_vertex = pVertex;
					pRight->m_lower.m_right = pLeft;
					if (pRight->m_lower.LeftID() != Lower.m_ID)
					{
						string str = "Right Lower Left ID != Lower ID\n";
						throw str;
					}
				}
				else if ((pRight->m_higher.VertexID() == pVertex->m_ID) || (pRight->m_higher.VertexID() == pVertex->m_ID + 1))
				{
					// Fix up upper end
					pRight->m_higher.m_vertex = pVertex;
					pRight->m_higher.m_right = pLeft;
					if (pRight->m_higher.LeftID() != Lower.m_ID)
					{
						string str = "Right Higher Left ID != Lower ID\n";
						throw str;
					}
				}
				else
				{
					string str = "FixSkeleton - Problem fixing right side\n";
					throw str;
				}
			}
			s1--;
			s1--;
		}
	}

#ifdef FELKELDEBUG
	void vtStraightSkeleton::Dump()
	{
		int i;

		VTLOG("Skeleton:\n");

		i = 0;
		for (CSkeleton::iterator s1 = m_skeleton.begin(); s1 != m_skeleton.end(); s1++)
		{
			CSkeletonLine& db = (*s1);
			VTLOG("ID: %d lower leftID %d rightID %d vertexID %d (%f %f %f)\nhigher leftID %d rightID %d vertexID %d (%f %f %f)\n",
				db.m_ID,
				db.m_lower.LeftID(),
				db.m_lower.RightID(),
				db.m_lower.VertexID(), db.m_lower.m_vertex->m_point.m_x, db.m_lower.m_vertex->m_point.m_y, db.m_lower.m_vertex->m_point.m_z,
				db.m_higher.LeftID(),
				db.m_higher.RightID(),
				db.m_higher.VertexID(), db.m_higher.m_vertex->m_point.m_x, db.m_higher.m_vertex->m_point.m_y, db.m_higher.m_vertex->m_point.m_z);
		}
	}
#endif
}
#endif
