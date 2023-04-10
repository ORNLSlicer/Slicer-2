#if 0
//
// FelkelIntersection.cpp: implementation of the CIntersection class.
//
// Copyright (c) 2003-2011 Virtual Terrain Project
// Free for all uses, see license.txt for details.
//
// Straight skeleton algorithm and original implementation
// courtesy of Petr Felkel and Stepan Obdrzalek (petr.felkel@tiani.com)
// Re-implemented for the Virtual Terrain Project (vterrain.org)
// by Roger James (www.beardandsandals.co.uk)
//


#include "FelkelIntersection.h"

namespace StraightSkeleton
{
	//////////////////////////////////////////////////////////////////////
	// Construction/Destruction
	//////////////////////////////////////////////////////////////////////

	CIntersection::CIntersection(CVertexList &vl, CVertex &v)
	{
#if VTDEBUG
		if (!(v.m_prevVertex == NULL || v.m_leftLine.FacingTowards (v.m_prevVertex -> m_rightLine)))
			VTLOG("%s %d Assert failed\n", __FILE__, __LINE__);
		if (!(v.m_nextVertex == NULL || v.m_rightLine.FacingTowards (v.m_nextVertex -> m_leftLine)))
			VTLOG("%s %d Assert failed\n", __FILE__, __LINE__);
#endif

		CVertex &l = *v.m_prevVertex;
		CVertex &r = *v.m_nextVertex;

#if VTDEBUG
		if (!(v.m_leftLine.m_Angle == v.m_leftVertex -> m_leftLine.m_Angle))
			VTLOG("%s %d Assert failed\n", __FILE__, __LINE__);
		if (!(v.m_rightLine.m_Angle == v.m_rightVertex -> m_rightLine.m_Angle))
			VTLOG("%s %d Assert failed\n", __FILE__, __LINE__);
#endif

		CNumber al = v.m_axis.m_Angle - l.m_axis.m_Angle;
		al.NormalizeAngle();

		CNumber ar = v.m_axis.m_Angle - r.m_axis.m_Angle;
		ar.NormalizeAngle();

#ifdef FELKELDEBUG
		VTLOG("New Intersection i1\n");
#endif
		C3DPoint i1 = v.m_axis.FacingTowards(l.m_axis) ? C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY) : v.m_axis.Intersection(l.m_axis);
		i1.m_y = v.m_leftLine.Dist(i1) * fabs(tan(v.m_leftLine.m_Slope));
#ifdef FELKELDEBUG
		VTLOG("New Intersection i1\nm_Origin(x %e y %e z %e) m_Angle %e m_Slope %e\na.m_Origin(x %e y %e z %e) a.m_Angle %e a.m_Slope %e\n",
			v.m_axis.m_Origin.m_x, v.m_axis.m_Origin.m_y, v.m_axis.m_Origin.m_z, v.m_axis.m_Angle, v.m_axis.m_Slope,
			l.m_axis.m_Origin.m_x, l.m_axis.m_Origin.m_y, l.m_axis.m_Origin.m_z, l.m_axis.m_Angle, l.m_axis.m_Slope);
#endif
#ifdef FELKELDEBUG
		VTLOG("New Intersection i2\n");
#endif
		C3DPoint i2 = v.m_axis.FacingTowards(r.m_axis) ? C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY) : v.m_axis.Intersection(r.m_axis);
		i2.m_y = v.m_rightLine.Dist(i2) * fabs(tan(v.m_rightLine.m_Slope));
#ifdef FELKELDEBUG
		VTLOG("m_Origin(x %e y %e z %e) m_Angle %e m_Slope %e\na.m_Origin(x %e y %e z %e) a.m_Angle %e a.m_Slope %e\n",
			v.m_axis.m_Origin.m_x, v.m_axis.m_Origin.m_y, v.m_axis.m_Origin.m_z, v.m_axis.m_Angle, v.m_axis.m_Slope,
			r.m_axis.m_Origin.m_x, r.m_axis.m_Origin.m_y, r.m_axis.m_Origin.m_z, r.m_axis.m_Angle, r.m_axis.m_Slope);
#endif

#if VTDEBUG
		CNumber Oldi1y = i1.m_y;
		CNumber Oldi2y = i2.m_y;
#endif
		// I need to check why this code is here !!!!!!!!!
		// I must of put it here bu I cannot remember why
		// Getting a slope of exactly PI/2 must be rare
		// but could arise from many different edge slopes and vertex angles
		if (SIMILAR(v.m_axis.m_Slope, CN_PI / 2))
		{
			i1.m_y = C3DPoint(i1 - l.m_point).LengthXZ() * fabs(tan(l.m_axis.m_Slope)) + l.m_point.m_y;
			i2.m_y = C3DPoint(i2 - r.m_point).LengthXZ() * fabs(tan(r.m_axis.m_Slope)) + r.m_point.m_y;
		}
		else
		{
			i1.m_y = C3DPoint(i1 - v.m_point).LengthXZ() * fabs(tan(v.m_axis.m_Slope)) + v.m_point.m_y;
			i2.m_y = C3DPoint(i2 - v.m_point).LengthXZ() * fabs(tan(v.m_axis.m_Slope)) + v.m_point.m_y;
		}
		//	assert ((Oldi1y == i1.m_y) && (Oldi2y == i2.m_y));

		CNumber d1 = v.m_point.DistXZ(i1);
		CNumber d2 = v.m_point.DistXZ(i2);
		//	CNumber d1 = i1.m_y;
		//	CNumber d2 = i2.m_y;


		CVertex *leftPointer, *rightPointer;
		C3DPoint p;
		CNumber d3 = CN_INFINITY;
		CNumber av = v.m_leftLine.m_Angle - v.m_rightLine.m_Angle;

		av.NormalizeAngle();
		if ((av >= 0.0 || av == -CN_PI) && (v.m_leftLine.Intersection(v.m_rightLine) == v.m_point || v.m_leftLine.Intersection(v.m_rightLine) == C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY)))
			d3 = v.NearestIntersection(vl, &leftPointer, &rightPointer, p);
		//	d3 = p.m_y;

#ifdef FELKELDEBUG
		VTLOG("New Intersection i1\nm_Origin(x %e y %e z %e) m_Angle %e m_Slope %e\na.m_Origin(x %e y %e z %e) a.m_Angle %e a.m_Slope %e\n",
			v.m_axis.m_Origin.m_x, v.m_axis.m_Origin.m_y, v.m_axis.m_Origin.m_z, v.m_axis.m_Angle, v.m_axis.m_Slope,
			l.m_axis.m_Origin.m_x, l.m_axis.m_Origin.m_y, l.m_axis.m_Origin.m_z, l.m_axis.m_Angle, l.m_axis.m_Slope);
		VTLOG("New Intersection i2\nm_Origin(x %e y %e z %e) m_Angle %e m_Slope %e\na.m_Origin(x %e y %e z %e) a.m_Angle %e a.m_Slope %e\n",
			v.m_axis.m_Origin.m_x, v.m_axis.m_Origin.m_y, v.m_axis.m_Origin.m_z, v.m_axis.m_Angle, v.m_axis.m_Slope,
			r.m_axis.m_Origin.m_x, r.m_axis.m_Origin.m_y, r.m_axis.m_Origin.m_z, r.m_axis.m_Angle, r.m_axis.m_Slope);
		VTLOG("New Intersection\n al %e ar %e\ni1.x %e i1.y %e i1.z %e\ni2.x %e i2.y %e i2.z %e\np.m_x %e p.m_y %e p.m_z %e\nd1 %e d2 %e d3 %e\n",
			al, ar, i1.m_x, i1.m_y, i1.m_z, i2.m_x, i2.m_y, i2.m_z, p.m_x, p.m_y, p.m_z, d1, d2, d3);
#endif

		if (d3 <= d1 && d3 <= d2)
		{
			m_poi = p;
			m_leftVertex = m_rightVertex = &v;
			m_type = NONCONVEX;
			if (v.InvalidIntersection(vl, *this))
			{
				d3 = CN_INFINITY;
				m_poi = C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY);
			}
		}

		if (d1 <= d2 && d1 <= d3)
		{
			m_leftVertex = &l;
			m_rightVertex = &v;
			m_poi = i1;
			m_type = CONVEX;
		}
		else if (d2 <= d1 && d2 <= d3)
		{
			m_leftVertex = &v;
			m_rightVertex = &r;
			m_poi = i2;
			m_type = CONVEX;
		}

		if (m_poi == C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY))
			m_height = CN_INFINITY;
		else
			m_height = m_poi.m_y;

#ifdef FELKELDEBUG
		VTLOG("New %s Intersection %d %d x %e y %e z %e height %e\n",
			m_type == CONVEX ? "CONVEX" : "NONCONVEX",
			m_leftVertex->m_ID, m_rightVertex->m_ID, m_poi.m_x, m_poi.m_y, m_poi.m_z, m_height);
#endif
	}

	void CIntersection::ApplyNonconvexIntersection(CSkeleton &skeleton, CVertexList &vl, IntersectionQueue &iq, bool bCheckVertexinCurrentContour)
	{
#ifdef FELKELDEBUG
		VTLOG("ApplyNonconvexIntersection\n");
#endif

#if VTDEBUG
		// Left and right vertices must always be the same point
		if (!(m_leftVertex == m_rightVertex))
			VTLOG("%s %d Assert failed\n", __FILE__, __LINE__);
		// Check to see of they are the same data structure RFJ !!!
		if (!(m_leftVertex->m_ID == m_rightVertex->m_ID))
			VTLOG("%s %d Assert failed\n", __FILE__, __LINE__);
#endif

		CVertex *leftPointer, *rightPointer;
		C3DPoint p;
		CNumber d3 = CN_INFINITY;

		d3 = m_leftVertex->NearestIntersection(vl, &leftPointer, &rightPointer, p);
		if (d3 == CN_INFINITY)
			return;

		if (p != m_poi)
			return;

		if (!m_leftVertex->VertexInCurrentContour(*leftPointer))
		{
			if (bCheckVertexinCurrentContour) // Temporary hack to disable checking in multiple contour buildings !!!! NEEDS FIXING
				return;
			else
				fprintf(stderr, "Vertex in current contour check failed - needs to be fixed - this check is not valid if one (or both?) of the contours is clockwise\n");
		}
		// Left and right vertex are actually the same in this case
		if (!m_rightVertex->VertexInCurrentContour(*rightPointer))
		{
			if (bCheckVertexinCurrentContour) // Temporary hack to disable checking in multiple contour buildings !!!! NEEDS FIXING
				return;
			else
				fprintf(stderr, "Vertex in current contour check failed - needs to be fixed - this check is not valid if one (or both?) of the contours is clockwise\n");
		}

#ifdef FELKELDEBUG
		VTLOG("left vertex %d left ptr %d right ptr %d right vertex %d\n",
			m_leftVertex->m_ID,
			leftPointer->m_ID,
			rightPointer->m_ID,
			m_rightVertex->m_ID);
#endif

		// Treat as a split event
		CVertex v1(p, *rightPointer, *m_rightVertex);
		CVertex v2(p, *m_leftVertex, *leftPointer);

#if VTDEBUG
		if (!(v1.m_point != C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY)))
			VTLOG("%s %d Assert failed\n", __FILE__, __LINE__);
		if (!(v2.m_point != C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY)))
			VTLOG("%s %d Assert failed\n", __FILE__, __LINE__);
#endif

		m_leftVertex->m_done = true;
		//  i.rightVertex -> done = true;

		CVertex *newNext1 = m_rightVertex->m_nextVertex;
		CVertex *newPrev1 = leftPointer->Highest();
		v1.m_prevVertex = newPrev1;
		v1.m_nextVertex = newNext1;
		vl.push_back(v1);

		CVertex *v1Pointer = &vl.back();

		newPrev1->m_nextVertex = v1Pointer;
		newNext1->m_prevVertex = v1Pointer;
		m_rightVertex->m_higher = v1Pointer;

		CVertex *newNext2 = rightPointer->Highest();
		CVertex *newPrev2 = m_leftVertex->m_prevVertex;
		v2.m_prevVertex = newPrev2;
		v2.m_nextVertex = newNext2;
		vl.push_back(v2);

		CVertex *v2Pointer = &vl.back();

		newPrev2->m_nextVertex = v2Pointer;
		newNext2->m_prevVertex = v2Pointer;
		m_leftVertex->m_higher = v2Pointer;

		skeleton.push_back(CSkeletonLine(*m_rightVertex, *v1Pointer));

		CSkeletonLine *linePtr = &skeleton.back();

		skeleton.push_back(CSkeletonLine(*v1Pointer, *v2Pointer));

		CSkeletonLine *auxLine1Ptr = &skeleton.back();

		skeleton.push_back(CSkeletonLine(*v2Pointer, *v1Pointer));

		CSkeletonLine *auxLine2Ptr = &skeleton.back();

		linePtr->m_lower.m_right = m_leftVertex->m_leftSkeletonLine;
		linePtr->m_lower.m_left = m_leftVertex->m_rightSkeletonLine;

		v1Pointer->m_rightSkeletonLine = v2Pointer->m_leftSkeletonLine = linePtr;
		v1Pointer->m_leftSkeletonLine = auxLine1Ptr;
		v2Pointer->m_rightSkeletonLine = auxLine2Ptr;

		auxLine1Ptr->m_lower.m_right = auxLine2Ptr;
		auxLine2Ptr->m_lower.m_left = auxLine1Ptr;

		if (m_leftVertex->m_leftSkeletonLine)
			m_leftVertex->m_leftSkeletonLine->m_higher.m_left = linePtr;
		if (m_leftVertex->m_rightSkeletonLine)
			m_leftVertex->m_rightSkeletonLine->m_higher.m_right = linePtr;
		m_leftVertex->m_advancingSkeletonLine = linePtr;

		if (newNext1 == newPrev1)
		{
			v1Pointer->m_done = true;
			newNext1->m_done = true;
			skeleton.push_back(CSkeletonLine(*v1Pointer, *newNext1));
			CSkeletonLine *linePtr = &skeleton.back();
			linePtr->m_lower.m_right = v1Pointer->m_leftSkeletonLine;
			linePtr->m_lower.m_left = v1Pointer->m_rightSkeletonLine;
			linePtr->m_higher.m_right = newNext1->m_leftSkeletonLine;
			linePtr->m_higher.m_left = newNext1->m_rightSkeletonLine;

			if (v1Pointer->m_leftSkeletonLine)
				v1Pointer->m_leftSkeletonLine->m_higher.m_left = linePtr;
			if (v1Pointer->m_rightSkeletonLine)
				v1Pointer->m_rightSkeletonLine->m_higher.m_right = linePtr;
			if (newNext1->m_leftSkeletonLine)
				newNext1->m_leftSkeletonLine->m_higher.m_left = linePtr;
			if (newNext1->m_rightSkeletonLine)
				newNext1->m_rightSkeletonLine->m_higher.m_right = linePtr;
		}
		else
		{
			CIntersection i1(vl, *v1Pointer);
			if (i1.m_height != CN_INFINITY)
				iq.push(i1);
		}

		if (newNext2 == newPrev2)
		{
			v2Pointer->m_done = true;
			newNext2->m_done = true;
			skeleton.push_back(CSkeletonLine(*v2Pointer, *newNext2));
			CSkeletonLine *linePtr = &skeleton.back();
			linePtr->m_lower.m_right = v2Pointer->m_leftSkeletonLine;
			linePtr->m_lower.m_left = v2Pointer->m_rightSkeletonLine;
			linePtr->m_higher.m_right = newNext2->m_leftSkeletonLine;
			linePtr->m_higher.m_left = newNext2->m_rightSkeletonLine;

			if (v2Pointer->m_leftSkeletonLine)
				v2Pointer->m_leftSkeletonLine->m_higher.m_left = linePtr;
			if (v2Pointer->m_rightSkeletonLine)
				v2Pointer->m_rightSkeletonLine->m_higher.m_right = linePtr;
			if (newNext2->m_leftSkeletonLine)
				newNext2->m_leftSkeletonLine->m_higher.m_left = linePtr;
			if (newNext2->m_rightSkeletonLine)
				newNext2->m_rightSkeletonLine->m_higher.m_right = linePtr;
		}
		else
		{
			CIntersection i2(vl, *v2Pointer);
			if (i2.m_height != CN_INFINITY)
				iq.push(i2);
		}
	}

	void CIntersection::ApplyConvexIntersection(CSkeleton &skeleton, CVertexList &vl, IntersectionQueue &iq)
	{
#ifdef FELKELDEBUG
		VTLOG("ApplyConvexIntersection\n");
#endif
		// create new vertex and link into current contour
		CVertex vtx(m_poi, *m_leftVertex, *m_rightVertex);
#if VTDEBUG
		if(!(vtx.m_point != C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY)))
			VTLOG("%s %d Assert failed\n", __FILE__, __LINE__);
#endif

		// Link vertex into overall chain
		CVertex *newNext = m_rightVertex->m_nextVertex;
		CVertex *newPrev = m_leftVertex->m_prevVertex;

		vtx.m_prevVertex = newPrev;
		vtx.m_nextVertex = newNext;

		vl.push_back(vtx);

		CVertex *vtxPointer = &vl.back();

		newPrev->m_nextVertex = vtxPointer;
		newNext->m_prevVertex = vtxPointer;

		// Set this vertex as the higher skeleton point for the vertices which have been
		// removed from the active contour
		m_leftVertex->m_higher = vtxPointer;
		m_rightVertex->m_higher = vtxPointer;

		// mark vertices as inactive
		m_leftVertex->m_done = true;
		m_rightVertex->m_done = true;

		CIntersection newI(vl, *vtxPointer);

		if (newI.m_height != CN_INFINITY)
			iq.push(newI);

		skeleton.push_back(CSkeletonLine(*m_leftVertex, *vtxPointer));

		CSkeletonLine *lLinePtr = &skeleton.back();

		skeleton.push_back(CSkeletonLine(*m_rightVertex, *vtxPointer));

		CSkeletonLine *rLinePtr = &skeleton.back();

		lLinePtr->m_lower.m_right = m_leftVertex->m_leftSkeletonLine;
		lLinePtr->m_lower.m_left = m_leftVertex->m_rightSkeletonLine;
		lLinePtr->m_higher.m_right = rLinePtr;
		rLinePtr->m_lower.m_right = m_rightVertex->m_leftSkeletonLine;
		rLinePtr->m_lower.m_left = m_rightVertex->m_rightSkeletonLine;
		rLinePtr->m_higher.m_left = lLinePtr;

		if (m_leftVertex->m_leftSkeletonLine)
			m_leftVertex->m_leftSkeletonLine->m_higher.m_left = lLinePtr;
		if (m_leftVertex->m_rightSkeletonLine)
			m_leftVertex->m_rightSkeletonLine->m_higher.m_right = lLinePtr;

		if (m_rightVertex->m_leftSkeletonLine)
			m_rightVertex->m_leftSkeletonLine->m_higher.m_left = rLinePtr;
		if (m_rightVertex->m_rightSkeletonLine)
			m_rightVertex->m_rightSkeletonLine->m_higher.m_right = rLinePtr;

		vtxPointer->m_leftSkeletonLine = lLinePtr;
		vtxPointer->m_rightSkeletonLine = rLinePtr;

		m_leftVertex->m_advancingSkeletonLine = lLinePtr;
		m_rightVertex->m_advancingSkeletonLine = rLinePtr;
	}

	void CIntersection::ApplyLast3(CSkeleton &skeleton, CVertexList &vl)
	{
#ifdef FELKELDEBUG
		VTLOG("ApplyLast3\n");
#endif

		CVertex &v1 = *m_leftVertex;
		CVertex &v2 = *m_rightVertex;
		CVertex &v3 = *m_leftVertex->m_prevVertex;

		v1.m_done = true;
		v2.m_done = true;
		v3.m_done = true;


		C3DPoint is = m_poi;

		CVertex v(is);

		v.m_done = true;
		vl.push_back(v);
		CVertex *vtxPointer = &vl.back();

		skeleton.push_back(CSkeletonLine(v1, *vtxPointer));

		CSkeletonLine *line1Ptr = &skeleton.back();

		skeleton.push_back(CSkeletonLine(v2, *vtxPointer));

		CSkeletonLine *line2Ptr = &skeleton.back();

		skeleton.push_back(CSkeletonLine(v3, *vtxPointer));

		CSkeletonLine *line3Ptr = &skeleton.back();

		line1Ptr->m_higher.m_right = line2Ptr;	// zapojeni okridlenych hran
		line2Ptr->m_higher.m_right = line3Ptr;
		line3Ptr->m_higher.m_right = line1Ptr;

		line1Ptr->m_higher.m_left = line3Ptr;
		line2Ptr->m_higher.m_left = line1Ptr;
		line3Ptr->m_higher.m_left = line2Ptr;

		line1Ptr->m_lower.m_left = v1.m_rightSkeletonLine;
		line1Ptr->m_lower.m_right = v1.m_leftSkeletonLine;

		line2Ptr->m_lower.m_left = v2.m_rightSkeletonLine;
		line2Ptr->m_lower.m_right = v2.m_leftSkeletonLine;

		line3Ptr->m_lower.m_left = v3.m_rightSkeletonLine;
		line3Ptr->m_lower.m_right = v3.m_leftSkeletonLine;

		if (v1.m_leftSkeletonLine)
			v1.m_leftSkeletonLine->m_higher.m_left = line1Ptr;
		if (v1.m_rightSkeletonLine)
			v1.m_rightSkeletonLine->m_higher.m_right = line1Ptr;

		if (v2.m_leftSkeletonLine)
			v2.m_leftSkeletonLine->m_higher.m_left = line2Ptr;
		if (v2.m_rightSkeletonLine)
			v2.m_rightSkeletonLine->m_higher.m_right = line2Ptr;

		if (v3.m_leftSkeletonLine)
			v3.m_leftSkeletonLine->m_higher.m_left = line3Ptr;
		if (v3.m_rightSkeletonLine)
			v3.m_rightSkeletonLine->m_higher.m_right = line3Ptr;

		v1.m_advancingSkeletonLine = line1Ptr;
		v2.m_advancingSkeletonLine = line2Ptr;
		v3.m_advancingSkeletonLine = line3Ptr;
	}

}
#endif
