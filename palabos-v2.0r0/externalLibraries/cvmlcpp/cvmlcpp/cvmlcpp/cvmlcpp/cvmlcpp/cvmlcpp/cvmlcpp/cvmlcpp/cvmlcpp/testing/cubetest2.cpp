// g++ -Wall -ansi -o voxelizer-test voxelizer-test.cpp -lcvmlcpp
// ./voxelizer-test VOXEL_SIZE NUMBER_OF_FACETS MIN_RAND_COORDS MAX_RAND_COORDS

#include <iostream>
#include <cstdlib>

#include <boost/lexical_cast.hpp>

#include <volume/Voxelizer>  // must be included *after* Geometry


int main(const int argc, const char **argv)
{
	//// commandline parameter parsing ////
	
	// voxel size
	double vsize = 0.5;
	if (argc > 1)
		vsize = boost::lexical_cast<double>(argv[1]);

	// minimum of facet point coordinates
	double min = 0.0;
	if (argc > 2)
		min = boost::lexical_cast<double>(argv[2]);

	// maximum of facet point coordinates
	double max = 1.0;
	if (argc > 3)
		max = boost::lexical_cast<double>(argv[3]);


	//// setup cube geometry ////
	
	cvmlcpp::Geometry<float> g;

	const unsigned int p000 = g.addPoint(min, min, min);
	const unsigned int p100 = g.addPoint(max, min, min);
	const unsigned int p010 = g.addPoint(min, max, min);
	const unsigned int p110 = g.addPoint(max, max, min);
	const unsigned int p001 = g.addPoint(min, min, max);
	const unsigned int p101 = g.addPoint(max, min, max);
	const unsigned int p011 = g.addPoint(min, max, max);
	const unsigned int p111 = g.addPoint(max, max, max);
/*
//1.quadrant
	const unsigned int p000 = g.addPoint(1,1,1);
	const unsigned int p100 = g.addPoint(2,1,1);
	const unsigned int p010 = g.addPoint(1,2,1);
	const unsigned int p110 = g.addPoint(2,2,1);
	const unsigned int p001 = g.addPoint(1,1,2);
	const unsigned int p101 = g.addPoint(2,1,2);
	const unsigned int p011 = g.addPoint(1,2,2);
	const unsigned int p111 = g.addPoint(2,2,2);

//2.q
	const unsigned int p000 = g.addPoint(-2,1,1);
	const unsigned int p100 = g.addPoint(-1,1,1);
	const unsigned int p010 = g.addPoint(-2,2,1);
	const unsigned int p110 = g.addPoint(-1,2,1);
	const unsigned int p001 = g.addPoint(-2,1,2);
	const unsigned int p101 = g.addPoint(-1,1,2);
	const unsigned int p011 = g.addPoint(-2,2,2);
	const unsigned int p111 = g.addPoint(-1,2,2);

//3.q
	const unsigned int p000 = g.addPoint(-2,-2,1);
	const unsigned int p100 = g.addPoint(-1,-2,1);
	const unsigned int p010 = g.addPoint(-2,-1,1);
	const unsigned int p110 = g.addPoint(-1,-1,1);
	const unsigned int p001 = g.addPoint(-2,-2,2);
	const unsigned int p101 = g.addPoint(-1,-2,2);
	const unsigned int p011 = g.addPoint(-2,-1,2);
	const unsigned int p111 = g.addPoint(-1,-1,2);
//4.q
	const unsigned int p000 = g.addPoint(1,-2,1);
	const unsigned int p100 = g.addPoint(2,-2,1);
	const unsigned int p010 = g.addPoint(1,-1,1);
	const unsigned int p110 = g.addPoint(2,-1,1);
	const unsigned int p001 = g.addPoint(1,-2,2);
	const unsigned int p101 = g.addPoint(2,-2,2);
	const unsigned int p011 = g.addPoint(1,-1,2);
	const unsigned int p111 = g.addPoint(2,-1,2);

//5.q
	const unsigned int p000 = g.addPoint(1,1,-2);
	const unsigned int p100 = g.addPoint(2,1,-2);
	const unsigned int p010 = g.addPoint(1,2,-2);
	const unsigned int p110 = g.addPoint(2,2,-2);
	const unsigned int p001 = g.addPoint(1,1,-1);
	const unsigned int p101 = g.addPoint(2,1,-1);
	const unsigned int p011 = g.addPoint(1,2,-1);
	const unsigned int p111 = g.addPoint(2,2,-1);

//6.q
	const unsigned int p000 = g.addPoint(-2,1,-2);
	const unsigned int p100 = g.addPoint(-1,1,-2);
	const unsigned int p010 = g.addPoint(-2,2,-2);
	const unsigned int p110 = g.addPoint(-1,2,-2);
	const unsigned int p001 = g.addPoint(-2,1,-1);
	const unsigned int p101 = g.addPoint(-1,1,-1);
	const unsigned int p011 = g.addPoint(-2,2,-1);
	const unsigned int p111 = g.addPoint(-1,2,-1);
//7.q
	const unsigned int p000 = g.addPoint(-2,-2,-2);
	const unsigned int p100 = g.addPoint(-1,-2,-2);
	const unsigned int p010 = g.addPoint(-2,-1,-2);
	const unsigned int p110 = g.addPoint(-1,-1,-2);
	const unsigned int p001 = g.addPoint(-2,-2,-1);
	const unsigned int p101 = g.addPoint(-1,-2,-1);
	const unsigned int p011 = g.addPoint(-2,-1,-1);
	const unsigned int p111 = g.addPoint(-1,-1,-1);

//8.q
	const unsigned int p000 = g.addPoint(1,-2,-2);
	const unsigned int p100 = g.addPoint(2,-2,-2);
	const unsigned int p010 = g.addPoint(1,-1,-2);
	const unsigned int p110 = g.addPoint(2,-1,-2);
	const unsigned int p001 = g.addPoint(1,-2,-1);
	const unsigned int p101 = g.addPoint(2,-2,-1);
	const unsigned int p011 = g.addPoint(1,-1,-1);
	const unsigned int p111 = g.addPoint(2,-1,-1);
*/
	// bottom
	g.addFacet(p000, p010, p100);
	g.addFacet(p100, p010, p110);
	// top
	g.addFacet(p001, p101, p011);
	g.addFacet(p101, p111, p011);
	// left
	g.addFacet(p000, p100, p001);
	g.addFacet(p100, p101, p001);
	// right
	g.addFacet(p010, p011, p110);
	g.addFacet(p110, p011, p111);
	// front
	g.addFacet(p100, p110, p101);
	g.addFacet(p110, p101, p111);
	//back
	g.addFacet(p001, p011, p010);
	g.addFacet(p010, p000, p001);
/*
	g.addFacet(g.addPoint(0,0,0),g.addPoint(0,0,1),g.addPoint(0,1,0));
	g.addFacet(g.addPoint(1,1,0),g.addPoint(1,0,0),g.addPoint(0,1,0));
	g.addFacet(g.addPoint(0,1,0),g.addPoint(0,0,1),g.addPoint(0,1,1));
	g.addFacet(g.addPoint(0,0,0),g.addPoint(1,0,0),g.addPoint(0,0,1));
	g.addFacet(g.addPoint(0,0,0),g.addPoint(0,1,0),g.addPoint(1,0,0));
	g.addFacet(g.addPoint(1,0,0),g.addPoint(1,0,1),g.addPoint(0,0,1));
	g.addFacet(g.addPoint(0,1,1),g.addPoint(0,0,1),g.addPoint(1,0,1));
	g.addFacet(g.addPoint(1,1,1),g.addPoint(1,0,1),g.addPoint(1,1,0));
	g.addFacet(g.addPoint(1,1,0),g.addPoint(1,0,1),g.addPoint(1,0,0));
	g.addFacet(g.addPoint(0,1,1),g.addPoint(1,1,0),g.addPoint(0,1,0));
	g.addFacet(g.addPoint(1,1,1),g.addPoint(1,1,0),g.addPoint(0,1,1));
	g.addFacet(g.addPoint(1,1,1),g.addPoint(0,1,1),g.addPoint(1,0,1));
*/
	std::cout.precision(25);
	std::cout << "----------------------------------------" << std::endl;
	std::cout << "voxel size: " << vsize << std::endl;
	std::cout << "Z Buffer: " << std::endl;

	
	//// voxelize ////

	//cvmlcpp::Matrix<float, 3u> m;
	cvmlcpp::Matrix<int, 3u> m;
	//cvmlcpp::fractionVoxelize(g, m, vsize, 4, 0);
	cvmlcpp::voxelize(g, m, vsize, 0, 1, 0);

	const unsigned int xsize = m.extent(X);
	const unsigned int ysize = m.extent(Y);
	const unsigned int zsize = m.extent(Z);

	//std::cout << "min: " << min << " max: " << max << std::endl;

	std::cout << "matrix size: " << m.size() << std::endl;
	std::cout << "volume size: x:" << xsize
	                     << ", y:" << ysize
	                     << ", z:" << zsize << std::endl;

	//cvmlcpp::Matrix<float, 3u>::const_iterator it = m.begin();
	cvmlcpp::Matrix<int, 3u>::const_iterator it = m.begin();
	for (uint x = 0; x < xsize; ++x)
	{
		for (uint y = 0; y < ysize; ++y)
		{
			for (uint z = 0; z < zsize; ++z)
			{
				std::cout << *it++ << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	std::cout << "----------------------------------------" << std::endl;


	return 0;
}
