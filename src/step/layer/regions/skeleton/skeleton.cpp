#if 0
#define Verify_Skeleton_Integrity

#include "Skeleton.h"

namespace ORNLengine
{

	SkeletonNode::SkeletonNode()
	{
		m_visited = false;
	}

	/**
	 * Return true if the node is an edge of the skeleton and false if the edge is not an edge.
	 */
	bool SkeletonNode::getIsEdge()
	{
		return m_edges.size() <= 1;
	}

	Skeleton::~Skeleton()
	{
		for (SkeletonNode *node : m_nodes)
		{
			delete node;
		}
	}

	/**
	 * Method to add an edge of the skeleton to the skeleton graph.
	*/
	void Skeleton::AddSkeletonEdge(Point p1, Point p2)
	{
		SkeletonNode *p1Node; 
		SkeletonNode *p2Node;

		// Do not allow a edge of zero distance to be added
		if (p1 == p2)
			return;

		// Try to find the p1 node
		p1Node = GetSkeletonNodeFromPoint(p1);

		// Try to find the p2 node
		p2Node = GetSkeletonNodeFromPoint(p2);

		// If either node does not exist create it.
		if (p1Node == NULL)
		{
			p1Node = new SkeletonNode();
			m_nodes.push_back(p1Node);
			p1Node->m_point = p1;

		}
		if (p2Node == NULL)
		{
			p2Node = new SkeletonNode();
			m_nodes.push_back(p2Node);
			p2Node->m_point = p2;
		}

		// Create the edge from p1 to p2
		// Only add the edge if it doesn't already exist
		if (find(p1Node->m_edges.begin(), p1Node->m_edges.end(), p2Node) == p1Node->m_edges.end())
			p1Node->m_edges.push_back(p2Node);

		// Create the edge from p2 to p1
		// Only add the edge if it doesn't already exist
		if (find(p2Node->m_edges.begin(), p2Node->m_edges.end(), p1Node) == p2Node->m_edges.end())
			p2Node->m_edges.push_back(p1Node);
	}

	/**
	 * Reset the visited field for all the nodes.
	 */
	void Skeleton::ResetVisited()
	{
		for (SkeletonNode *n : m_nodes)
		{
			n->m_visited = false;
		}
	}

	/**
	 * Return the skeleton nodes that corresponds to a point.
	 */
	SkeletonNode *Skeleton::GetSkeletonNodeFromPoint(Point point)
	{
		for (SkeletonNode *node : m_nodes)
		{
			if (node->m_point == point)
			{
				return node;
			}
		}

		return NULL;
	}

	/**
	 *	Get paths from the skeleton.
	 */
	vector<vector<Point> >Skeleton::GetPathsFromSkeleton(Point startPoint)
	{
		vector<vector<Point> > paths;
		vector<Point> path;
		stack <SkeletonNode *> nstack;
		stack <SkeletonNode *> pstack;
		SkeletonNode *startNode = nullptr;
		SkeletonNode *node;
		SkeletonNode *pnode;
		bool startNewBranch = false;
		set<string> edges;
		string edge;

		// Reset the visited fields
		ResetVisited();

		int64_t mindist = 0;

		// Find the end point of the skeleton that is closest to the last position
		for (SkeletonNode *n : m_nodes)
		{			
			int64_t distance;
			if (n->getIsEdge())
			{
				// Calculate the distance from the node to the previous position.
				distance = vSize(startPoint - n->m_point);

				if (startNode == nullptr || distance < mindist)
				{
					startNode = n;
					mindist = distance;
				}
			}
		}

		// Return if no start node was found
		if (startNode == nullptr)
			return paths;

		// Add the first node to the stack
		nstack.push(startNode);
		pstack.push(nullptr);

		// Traverse the skeleton and the points to the paths
		while (!nstack.empty())
		{
			// Get the node off the stack
			node = nstack.top();
			nstack.pop();

			// Get the parent node off the stack
			pnode = pstack.top();
			pstack.pop();

			// Add the parent point if new branch
			if (startNewBranch)
			{
				path.push_back(pnode->m_point);
			}

			edge = "";
			// Create the edge string
			if (pnode != nullptr)
			{
				if (pnode->m_point.to_string().compare(node->m_point.to_string()) <= 0)
				{
					edge = pnode->m_point.to_string() + ", " + node->m_point.to_string();
				}
				else
				{
					edge = node->m_point.to_string() + ", " + pnode->m_point.to_string();
				}
			}

			// Add the point to the path only if the edge has not been added previously
			if (edge.compare("") == 0)
			{
				path.push_back(node->m_point);
			}
			else if (edges.find(edge) == edges.end())
			{
				path.push_back(node->m_point);
				edges.insert(edge);
			}

			startNewBranch = true;
			// If not visited push children
			if (!node->m_visited)
			{
				// Push the children 
				for (vector<SkeletonNode *>::reverse_iterator rit = node->m_edges.rbegin(); rit != node->m_edges.rend(); rit++)
				{

					// Only add children that are not the parent
					if (pnode != *rit)
					{
						// Add child to the stack
						nstack.push(*rit);

						// Add the parent to the stack
						pstack.push(node);

						// Set new branch to false
						startNewBranch = false;
					}
				}
			}

			// Set the node as visited
			node->m_visited = true;

			// If start new branch then push the path to paths
			if (startNewBranch)
			{
				// If the path is longer than 2 then add it to paths
				if (path.size() >= 2)
					paths.push_back(path);

				// Clear path for next path
				path.clear();
			}

			if (nstack.empty())
			{
				if (path.size() >= 2)
					paths.push_back(path);

				path.clear();
				startNewBranch = false;

				for (SkeletonNode *node : m_nodes)
				{
					if (!node->m_visited)
					{
						nstack.push(node);
						pstack.push(nullptr);
						break;
					}
				}
			}
		}

		// If the path is longer than 2 then add it to paths
		if (path.size() >= 2)
			paths.push_back(path);

#ifdef _DEBUG
#ifdef Verify_Skeleton_Integrity
		// Check to make sure the skeleton was a single graph.
		for (SkeletonNode *node : m_nodes)
		{
			if (!node->m_visited)
				throw exception("Not all nodes of the graph are connected.");
		}
#endif
#endif

		// Return the paths
		return paths;
	}

	/**
	 *	Combine the short segments in the skeleton
	 */
	void Skeleton::CombineShortEdges()
	{
		stack <SkeletonNode *> nstack;
		SkeletonNode *node;
		SkeletonNode *childNode;
		SkeletonNode *grandChildNode;
		int64_t length;
		vector<SkeletonNode *>::iterator nodeit;

		// If there are not nodes on the skeleton then return
		if (m_nodes.size() == 0)
			return;

		// Reset the visited fields
		ResetVisited();

		// If there are no nodes then return
		if (m_nodes.size() == 0)
			return;

		// Add the first node to the graph
		nstack.push(m_nodes[0]);

		// While the stack is not empty pop off the node and process the children
		while (!nstack.empty())
		{
			node = nstack.top();
			nstack.pop();

			// If already visited continue
			if (node->m_visited)
				continue;

			// Mark the node as visited
			node->m_visited = true;

			// Check the length of all the child nodes
			for (unsigned int idx = 0; idx < node->m_edges.size(); idx++)
			{
				childNode = node->m_edges[idx];
				length = vSize(node->m_point - childNode->m_point);

				// If the size is small then combined points
				if (length < SettingsBase::config->getInt("minimalExtrudeLength"))
				{
					for (unsigned int idx2 = 0; idx2 < childNode->m_edges.size(); idx2++)
					{
						grandChildNode = childNode->m_edges[idx2];

						// Remove the idx from the idx2 list
						nodeit = find(grandChildNode->m_edges.begin(), grandChildNode->m_edges.end(), childNode);
						if (nodeit != grandChildNode->m_edges.end())
						{
							grandChildNode->m_edges.erase(nodeit);
						}

						// Reconnect the graph
						if (grandChildNode != node)
						{
							grandChildNode->m_edges.push_back(node);
							node->m_edges.push_back(grandChildNode);
						}
					}

					// remove the child node from the list of nodes
					nodeit = find(m_nodes.begin(), m_nodes.end(), childNode);
					if (nodeit != m_nodes.end())
					{
						m_nodes.erase(nodeit);
					}

					// remove the child node
					delete childNode;

					// Back up the idx for the child
					idx--;
				}
				else
				{
					// Add child to the stack if the child has not been visited
					if (!childNode->m_visited)
						nstack.push(childNode);
				}

				// If the node now has no children remove it
				if (node->m_edges.size() == 0)
				{
					m_nodes.erase(find(m_nodes.begin(), m_nodes.end(), node));
					delete node;
				}
			}

			if (nstack.empty())
			{
				for (SkeletonNode *node : m_nodes)
				{
					if (!node->m_visited)
					{
						nstack.push(node);
						break;
					}
				}
			}
		}

		// If there is only one node left, remove it
		if (m_nodes.size() == 1)
		{
			delete m_nodes[0];
			m_nodes.clear();
		}
	}
}
#endif
