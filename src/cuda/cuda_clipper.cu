/*!
CUDA Polygon Clipper

A polygon clipping implementation is CUDA.
\note This class is nowhere near complete and should be run separately from the Slicer 2 project

It will likely need to be separated into its own library. See doc/GPU Acceleration 
*/

#include <thrust/device_vector.h>

struct Point
{
	Point()
	{
		x = 0.0;
		y = 0.0;
	}
	Point(float c_x, float c_y)
    {
		x = c_x;
		y = c_y;
    }
	float x;
	float y;
};

struct Edge
{
	Edge(Point first, Point second)
    {
		p0 = first;
		p1 = second;
    }
	Point p0;
	Point p1;
};

struct BoundingBox
{
	BoundingBox(Point _min, Point _max)
    {
		min = _min;
		max = _max;
	}
	Point min;
	Point max;
};

//! \brief A kernel to check and see of a two lines intersect
__global__
void checkForIntersect(Edge ray, int numberOfVertices, Edge * edges, int * flags)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if(threadId < numberOfVertices)
	{
        Edge edge = edges[threadId];
        // If our test point is not a vertex
		if(((ray.p0.x != edge.p0.x) && (ray.p0.y != edge.p0.y)) || ((ray.p0.x != edge.p1.x)  && (ray.p0.y != edge.p0.y)))
		{
			bool intersecting = (((edge.p0.x - ray.p0.x) * (ray.p1.y - ray.p0.y) - (edge.p0.y - ray.p0.y) * (ray.p1.x - ray.p0.x)) 
			* ((edge.p1.x - ray.p0.x) * (ray.p1.y - ray.p0.y) - (edge.p1.y - ray.p0.y) * (ray.p1.x - ray.p0.x)) < 0) 
			&& (((ray.p0.x - edge.p0.x) * (edge.p1.y - edge.p0.y) - (ray.p0.y - edge.p0.y) * (edge.p1.x - edge.p0.x)) 
			* ((ray.p1.x - edge.p0.x) * (edge.p1.y - edge.p0.y) - (ray.p1.y - edge.p0.y)*(edge.p1.x - edge.p0.x)) < 0);
			
			if(intersecting)
				flags[threadId] = 1;
		}
	}
}

//! \brief Performs a point in polygon test in parallel
__host__
bool isPointInPolygon(Point test_point, thrust::device_vector<Edge> &edges, BoundingBox boundingBox)
{
	Edge * d_edge_array = thrust::raw_pointer_cast(&edges[0]);

	thrust::device_vector<int> flags(edges.size());
	int * d_flags_array = thrust::raw_pointer_cast(&flags[0]);

	//Draws our scan array just outside of out bounding box to prevent the end from laying on an edge.
	Point end(boundingBox.max.x + 1.0, boundingBox.max.y / 2);
	Edge ray(test_point, end);

	int blockSize, gridSize;
	blockSize = 1024;
	gridSize = static_cast<int>(ceil((float)edges.size() / blockSize));

	checkForIntersect<<<gridSize,blockSize>>>(ray, edges.size(), d_edge_array, d_flags_array);
	cudaDeviceSynchronize();

	return (thrust::reduce(flags.begin(), flags.end()) % 2);
}

// Driver function for testing point-in-polygon
// int main(void)
// {
// 	thrust::device_vector<Edge> edges;
// 	Point p0(0.0,0.0);
// 	Point p1(5.0,0.0);
// 	Point p2(5.0,5.0);
// 	Point p3(0.0,5.0);

// 	Edge e0(p0,p1);
// 	Edge e1(p1,p2);
// 	Edge e2(p2,p3);
// 	Edge e3(p3,p0);
	
// 	edges.push_back(e0);
// 	edges.push_back(e1);
// 	edges.push_back(e2);
// 	edges.push_back(e3);

// 	BoundingBox boundingBox(p0, p2);

//     //Prints out test points from -1 to 7.5 and checks if there are in our polygon
// 	for(float i = -1;i<8;i += 0.5)
// 	{
// 		Point testPoint(i,i);
// 		bool p = isPointInPolygon(testPoint, edges, boundingBox);
// 		printf("Point (%f,%f): %i \n", testPoint.x, testPoint.y ,p);
// 	}
// 	return 0;
// }