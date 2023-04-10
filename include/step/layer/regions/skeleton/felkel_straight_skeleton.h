#if 0
/** Filename:FelkelStraightSkeleton.h
	Description:interface for the vtStraightSkeleton class
	Author:Roger James */

#ifndef STRAIGHTSKELETONH
#define STRAIGHTSKELETONH

#include "FelkelComponents.h"
#include "FelkelIntersection.h"

// Engine Additions
#include "../utils/polygon.h"
#include "../Skeleton.h"
#include "clipper.hpp"

namespace StraightSkeleton
{
	bool RayToLineSegmentIntersection(ClipperLib::DoublePoint rayBegin, ClipperLib::DoublePoint rayEnd, ClipperLib::DoublePoint lineBegin, ClipperLib::DoublePoint lineEnd, ClipperLib::DoublePoint &intersection);

	/*
	* Class:vtStraightSkeleton
	* Description:This class implements a Straight skeleton algorithm, used for generating geometry for building roofs when the
	*	footprint is complicated (more than just a regular rectangle).
	*/
	class vtStraightSkeleton
	{
	public:
		vtStraightSkeleton();
		virtual ~vtStraightSkeleton();

		CSkeleton& MakeSkeleton(ContourVector &contours);
		CSkeleton& MakeSkeleton(Contour &points);
		ORNLengine::Skeleton *MakeSkeleton(ORNLengine::PolygonRef& polygon, bool connectToPoly = false);
		ORNLengine::Skeleton *MakeSkeleton(ORNLengine::Polygons polygons, bool connectToPoly = false);
		CSkeleton CompleteWingedEdgeStructure(ContourVector &contours);
#ifdef FELKELDEBUG
		void Dump();
#endif

		IntersectionQueue m_iq;
		CVertexList m_vl;
		CSkeleton m_skeleton;
		CSkeleton m_boundaryedges;
		int m_NumberOfBoundaryVertices;
		int m_NumberOfBoundaryEdges;
	private:
		void FixSkeleton(); // Clean up the unlinked skeleton lines caused by non-convex intersections
	};
}
#endif // STRAIGHTSKELETONH
#endif
