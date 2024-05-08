/* This file is part of the Palabos library.
 *
 * The Palabos softare is developed since 2011 by FlowKit-Numeca Group Sarl
 * (Switzerland) and the University of Geneva (Switzerland), which jointly
 * own the IP rights for most of the code base. Since October 2019, the
 * Palabos project is maintained by the University of Geneva and accepts
 * source code contributions from the community.
 * 
 * Contact:
 * Jonas Latt
 * Computer Science Department
 * University of Geneva
 * 7 Route de Drize
 * 1227 Carouge, Switzerland
 * jonas.latt@unige.ch
 *
 * The most recent release of Palabos can be downloaded at 
 * <https://palabos.unige.ch/>
 *
 * The library Palabos is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * The library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/* Code 2.4 in the Palabos tutorial
 */

#include "palabos2D.h"
#include "palabos2D.hh"
#include <vector>
#include <iostream>
#include <iomanip>

using namespace plb;
using namespace std;

typedef double T;
#define DESCRIPTOR plb::descriptors::D2Q9Descriptor

/// Describe the geometry of the half-circular channel, used in tutorial 2.
template<typename T>
class BounceBackNodes : public DomainFunctional2D {
public:
    BounceBackNodes(plint N, plint radius)
        : cx(N/2),
          cy(N/2),
          innerR(radius),
          outerR(N/2)
    { }
    /// Return true for all cells outside the channel, on which bounce-back
    ///  dynamics must be instantiated.
    virtual bool operator() (plint iX, plint iY) const {
        T rSqr = util::sqr(iX-cx) + util::sqr(iY-cy);
        return rSqr <= innerR*innerR || rSqr >= outerR*outerR;
    }
    virtual BounceBackNodes<T>* clone() const {
        return new BounceBackNodes<T>(*this);
    }
private:
    plint cx;      //< X-position of the center of the half-circle.
    plint cy;      //< Y-position of the center of the half-circle.
    plint innerR;  //< Outer radius of the half-circle.
    plint outerR;  //< Inner radius of the half-circle.
};

template<typename T>
class FluidNodes {
public:
    FluidNodes(plint N_, plint radius_) : N(N_), radius(radius_)
    { }
    bool operator() (plint iX, plint iY) const {
        return ! BounceBackNodes<T>(N,radius)(iX,iY);
    }
private:
    plint N, radius;
};

void halfCircleSetup (
        MultiBlockLattice2D<T,DESCRIPTOR>& lattice, plint N, plint radius,
        OnLatticeBoundaryCondition2D<T,DESCRIPTOR>& boundaryCondition )
{
    // The channel is pressure-driven, with a difference deltaRho
    //   between inlet and outlet.
    T deltaRho = 1.e-2;
    T rhoIn  = 1. + deltaRho/2.;
    T rhoOut = 1. - deltaRho/2.;

    Box2D inlet (0,     N/2, N/2, N/2);
    Box2D outlet(N/2+1, N,   N/2, N/2);

    boundaryCondition.addPressureBoundary1P(inlet, lattice);
    boundaryCondition.addPressureBoundary1P(outlet, lattice);

    // Specify the inlet and outlet density.
    setBoundaryDensity (lattice, inlet, rhoIn);
    setBoundaryDensity (lattice, outlet, rhoOut);

    // Create the initial condition.
    Array<T,2> zeroVelocity((T)0.,(T)0.);
    T constantDensity = (T)1;
    initializeAtEquilibrium (
       lattice, lattice.getBoundingBox(), constantDensity, zeroVelocity );

    defineDynamics(lattice, lattice.getBoundingBox(),
                   new BounceBackNodes<T>(N, radius),
                   new BounceBack<T,DESCRIPTOR>);

    lattice.initialize();
}

void writeGifs(MultiBlockLattice2D<T,DESCRIPTOR>& lattice, plint iter)
{
    const plint imSize = 600;
    ImageWriter<T> imageWriter("leeloo");
    imageWriter.writeScaledGif(createFileName("u", iter, 6),
                               *computeVelocityNorm(lattice),
                               imSize, imSize );
}

int main(int argc, char* argv[]) {
    plbInit(&argc, &argv);

    global::directories().setOutputDir("./tmp/");

    // Parameters of the simulation
    plint N         = 400;    // Use a 400x200 domain.
    plint maxT      = 20001;
    plint imageIter = 1000;
    T omega        = 1.;
    plint radius    = N/3;    // Inner radius of the half-circle.

    MultiScalarField2D<int> flagMatrix(N+1, N/2+1);
    setToFunction(flagMatrix, flagMatrix.getBoundingBox(), FluidNodes<T>(N, radius));
    plint blockSize = 15;
    plint envelopeWidth = 1;
    MultiBlockManagement2D sparseBlockManagement =
        computeSparseManagement (
                *plb::reparallelize(flagMatrix, blockSize,blockSize),
                envelopeWidth );

    // Instantiate the multi-block, based on the created block distribution and
    // on default parameters.
    MultiBlockLattice2D<T, DESCRIPTOR> lattice (
        sparseBlockManagement,
        defaultMultiBlockPolicy2D().getBlockCommunicator(),
        defaultMultiBlockPolicy2D().getCombinedStatistics(),
        defaultMultiBlockPolicy2D().getMultiCellAccess<T,DESCRIPTOR>(),
        new BGKdynamics<T,DESCRIPTOR>(omega)
    );

    pcout << getMultiBlockInfo(lattice) << std::endl;

    OnLatticeBoundaryCondition2D<T,DESCRIPTOR>*
        boundaryCondition = createLocalBoundaryCondition2D<T,DESCRIPTOR>();

    halfCircleSetup(lattice, N, radius, *boundaryCondition);

    // Main loop over time iterations.
    for (plint iT=0; iT<maxT; ++iT) {
        if (iT%imageIter==0) {
            pcout << "Saving Gif at time step " << iT << endl;
            writeGifs(lattice, iT);
        }
        lattice.collideAndStream();
    }

    delete boundaryCondition;
}
