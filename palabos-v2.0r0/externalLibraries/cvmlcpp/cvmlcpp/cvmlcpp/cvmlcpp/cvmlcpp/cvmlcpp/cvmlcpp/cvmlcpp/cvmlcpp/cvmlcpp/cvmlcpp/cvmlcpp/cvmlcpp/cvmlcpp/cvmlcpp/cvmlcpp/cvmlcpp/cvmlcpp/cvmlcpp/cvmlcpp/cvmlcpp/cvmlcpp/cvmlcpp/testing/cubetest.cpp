// g++ -Wall -ansi -o vtest vtest.cpp -I/opt/cvmlcpp/default/include -L/opt/cvmlcpp/default/lib -lcvmlcpp
// LD_LIBRARY_PATH="/opt/cvmlcpp/default/lib" ./vtest 0.1

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <sys/time.h>

#include <volume/Voxelizer.experimental>  // must be included *after* Geometry
//#include <volume/VolumeIO>
//#include <volume/Facet>

void genRandSeed()
{
        struct timeval tv;
        struct timezone tz;
        gettimeofday(&tv, &tz);
        srand48(tv.tv_usec);
}

double rnd(const double minval, const double maxval)
{
	return drand48() * (maxval - minval) + minval;
}
void printMatrix(const cvmlcpp::Matrix<int,3u>& m, const cvmlcpp::Geometry<float>& g)
{
	const unsigned int xsize = m.extent(X);
	const unsigned int ysize = m.extent(Y);
	const unsigned int zsize = m.extent(Z);

	cvmlcpp::Matrix<int, 3u>::const_iterator it = m.begin();
	for (unsigned int x = 0; x < xsize; ++x)
	{
		for (unsigned int y = 0; y < ysize; ++y)
		{
			for (unsigned int z = 0; z < zsize; ++z)
			{
				std::cout << *it++ << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	
	std::cout << "number of facets: " << g.nrFacets() << std::endl;
	std::cout << "number of points: " << g.nrPoints() << std::endl;
	std::cout << "matrix size: " << m.size() << std::endl;
	std::cout << "volume size: x:" << xsize << ", y:" << ysize << ", z:" << zsize << std::endl;
}

int main(const int argc, const char **argv)
{
//	cvmlcpp::fTriangle3D t;
//	t[0] = cvmlcpp::fPoint3D(0, 0, 0);
//	std::vector<cvmlcpp::fPoint3D> pv(3);
//	cvmlcpp::fTriangle3D t(cvmlcpp::fPoint3D(0, 0, 0));
//	cvmlcpp::Facet<cvmlcpp::IndexTriangle, float> f;


	//// commandline parameter parsing ////
	
	double vsize = 0.2;
	if (argc > 1)
	{
		std::istringstream(argv[1]) >> vsize;  // convert to double
	}

	double min = 0.0;
	if (argc > 2)
	{
		std::istringstream(argv[2]) >> min;  // convert to double
	}

	double max = 1.0;
	if (argc > 3)
	{
		std::istringstream(argv[3]) >> max;  // convert to double
	}


	//// setup geometry ////

	cvmlcpp::Geometry<float> g;

// cube
	const unsigned int p000 = g.addPoint(min, min, min);
	const unsigned int p100 = g.addPoint(max, min, min);
	const unsigned int p010 = g.addPoint(min, max, min);
	const unsigned int p110 = g.addPoint(max, max, min);
	const unsigned int p001 = g.addPoint(min, min, max);
	const unsigned int p101 = g.addPoint(max, min, max);
	const unsigned int p011 = g.addPoint(min, max, max);
	const unsigned int p111 = g.addPoint(max, max, max);

	// front
	g.addFacet(p000, p010, p100);
	g.addFacet(p100, p010, p110);
	// back
	g.addFacet(p001, p101, p011);
	g.addFacet(p101, p111, p011);
	// bottom
	g.addFacet(p000, p100, p001);
	g.addFacet(p100, p101, p001);
	// top
	g.addFacet(p010, p011, p110);
	g.addFacet(p110, p011, p111);
	// right
	g.addFacet(p100, p110, p101);
	g.addFacet(p110, p111, p101);
	//left
	g.addFacet(p001, p011, p010);
	g.addFacet(p010, p000, p001);

	//// voxelize ////

	cvmlcpp::Matrix<int, 3u> m;
	cvmlcpp::voxelize(g, m, vsize, 0, 1, 0);
	printMatrix(m,g);

	return 0;
}
