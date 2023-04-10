#if 0
/** Filename:Skeleton.h
	Description:A skeleton is a single bead nonclosed pass extrusion move.
	Author:Andrew Messing*/

#ifndef SKELETON_H
#define SKELETON_H

#include <vector>
#include <stack>
#include <map>
#include <algorithm>
#include "utils\intpoint.h"
#include "settings.h"

using namespace std;

namespace ORNLengine
{
	/*
	* Class:SkeletNode
	* Description:Node of the Skeleton Graph.
	*/
	class SkeletonNode
	{
	public:
		// *** Variables ***
		Point m_point;
		vector<SkeletonNode *> m_edges;
		bool m_visited;

		// *** Methods ***
		SkeletonNode();
		bool getIsEdge();	
	};

	/*
	* Class:Skeleton
	* Description:graph containing polygon skeleton
	*/
	class Skeleton
	{
	public:
		// *** Variables ***
		vector<SkeletonNode *> m_nodes;

		// *** Methods ***
		~Skeleton();
		void AddSkeletonEdge(Point p1, Point p2);
		void ResetVisited();
		SkeletonNode *GetSkeletonNodeFromPoint(Point point);
		vector<vector<Point> >GetPathsFromSkeleton(Point startPoint);
		void CombineShortEdges();
	};
}
#endif//SKELETON_H
#endif
