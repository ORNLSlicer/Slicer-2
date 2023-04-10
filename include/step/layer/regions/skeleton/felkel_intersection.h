#if 0
/** Filename:FelkelIntersection.h
	Description: interface for the Cintersection class
	Author:Roger James */

#ifndef FELKELINTERSECTIONH
#define FELKELINTERSECTIONH

#include <queue>
#include <functional>
#include "FelkelComponents.h"
namespace StraightSkeleton
{
	class CSkeleton;

	typedef priority_queue <CIntersection, deque <CIntersection>, greater <CIntersection> > IntersectionQueue;
	/*
	* Class:CIntersection
	* Description:
	*/
	class CIntersection
	{
	public:
		CIntersection(void) { };
		CIntersection(CVertexList &vl, CVertex &v);

		void ApplyNonconvexIntersection(CSkeleton &skeleton, CVertexList &vl, IntersectionQueue &iq, bool bCheckVertexinCurrentContour);
		void ApplyConvexIntersection(CSkeleton &skeleton, CVertexList &vl, IntersectionQueue &iq);
		void ApplyLast3(CSkeleton &skeleton, CVertexList &vl);

		C3DPoint m_poi;
		CVertex *m_leftVertex, *m_rightVertex;
		CNumber m_height;
		enum Type { CONVEX, NONCONVEX } m_type;

		bool operator > (const CIntersection &i) const
		{
			// Do exact comparison for intersection queue
			// Using CNumber will also test for != which is implemented as !SIMILAR
			double d1 = m_height;
			double d2 = i.m_height;
			return d1 > d2;
		}
		bool operator == (const CIntersection &i) const
		{
			return m_poi == i.m_poi;
		}
	};
}
#endif // FELKELINTERSECTIONH
#endif
