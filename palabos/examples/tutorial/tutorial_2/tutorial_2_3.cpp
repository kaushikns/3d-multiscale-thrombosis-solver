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

#include "palabos2D.h"
#include "palabos2D.hh"
#include <vector>
#include <iostream>
#include <iomanip>

/* Code 2.2 in the Palabos tutorial
 */

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

/// Manual instantiation of the bounce-back nodes for the boundary.
/** Attention: This is NOT the right way of doing, because it is slow and
 *  non-parallelizable. Instead, use a data-processor.
 */
void createHalfCircleManually (
        MultiBlockLattice2D<T,DESCRIPTOR>& lattice, plint N, plint radius )
{
    BounceBackNodes<T> domain(N,radius);
    for (plint iX=0; iX<=N; ++iX) {
        for (plint iY=0; iY<=N/2; ++iY) {
            if (domain(iX,iY)) {
                defineDynamics(lattice, iX, iY, new BounceBack<T,DESCRIPTOR>);
            }
        }
    }
}

/// This functional defines a data processor for the instantiation
///   of bounce-back nodes following the half-circle geometry.
template<typename T, template<typename U> class Descriptor>
class HalfCircleInstantiateFunctional
    : public BoxProcessingFunctional2D_L<T,Descriptor>
{
public:
    HalfCircleInstantiateFunctional(plint N_, plint radius_)
        : N(N_), radius(radius_)
    { }
    virtual void process(Box2D domain, BlockLattice2D<T,Descriptor>& lattice) {
        BounceBackNodes<T> bbDomain(N,radius);
        Dot2D relativeOffset = lattice.getLocation();
        for (plint iX=domain.x0; iX<=domain.x1; ++iX) {
            for (plint iY=domain.y0; iY<=domain.y1; ++iY) {
                if (bbDomain(iX+relativeOffset.x,iY+relativeOffset.y)) {
                    lattice.attributeDynamics (
                            iX,iY, new BounceBack<T,DESCRIPTOR> );
                }
            }
        }
    }
    virtual void getTypeOfModification(std::vector<modif::ModifT>& modified) const {
        modified[0] = modif::dataStructure;
    }
    virtual HalfCircleInstantiateFunctional<T,Descriptor>* clone() const 
    {
        return new HalfCircleInstantiateFunctional<T,Descriptor>(*this);
    }
private:
    plint N;
    plint radius;
};

/// Automatic instantiation of the bounce-back nodes for the boundary, 
///   using a data processor.
void createHalfCircleFromDataProcessor (
        MultiBlockLattice2D<T,DESCRIPTOR>& lattice, plint N, plint radius )
{
    applyProcessingFunctional (
            new HalfCircleInstantiateFunctional<T,DESCRIPTOR>(N,radius),
            lattice.getBoundingBox(), lattice );
}


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

    // Manual creation of the geometry works but is slow, especially in
    //   parallel programs.
    /* createHalfCircleManually(lattice, N, radius); */

    // Automatic creation of the geometry is the recommended way of doing.
    createHalfCircleFromDataProcessor(lattice, N, radius);

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

    // Parameters for the creation of the multi-block.
    
    // d is the width of the block which is exempted from the full domain.
    plint d = (plint) (2.*std::sqrt((T)util::sqr(radius)-(T)util::sqr(N/4.)));
    plint x0 = (N-d)/2 + 1;  // Begin of the exempted block.
    plint x1 = (N+d)/2 - 1;  // End of the exempted block.

    // Create a block distribution with the three added blocks.
    plint envelopeWidth = 1;
    SparseBlockStructure2D sparseBlock(N+1, N/2+1);
    sparseBlock.addBlock(Box2D(0, x0,      0, N/2),   sparseBlock.nextIncrementalId());
    sparseBlock.addBlock(Box2D(x0+1, x1-1, 0, N/4+1), sparseBlock.nextIncrementalId());
    sparseBlock.addBlock(Box2D(x1, N,      0, N/2),   sparseBlock.nextIncrementalId());

    // Instantiate the multi-block, based on the created block distribution and
    //   on default parameters.
    MultiBlockLattice2D<T, DESCRIPTOR> lattice (
        MultiBlockManagement2D (
            sparseBlock,
            defaultMultiBlockPolicy2D().getThreadAttribution(), envelopeWidth ),
        defaultMultiBlockPolicy2D().getBlockCommunicator(),
        defaultMultiBlockPolicy2D().getCombinedStatistics(),
        defaultMultiBlockPolicy2D().getMultiCellAccess<T,DESCRIPTOR>(),
        new BGKdynamics<T,DESCRIPTOR>(omega)
    );

    OnLatticeBoundaryCondition2D<T,DESCRIPTOR>*
        boundaryCondition = createLocalBoundaryCondition2D<T,DESCRIPTOR>();

    halfCircleSetup(lattice, N, radius, *boundaryCondition);

    // Main loop over time iterations.
    for (plint iT=0; iT<maxT; ++iT) {
        if (iT%imageIter==0) {
            pcout << "Saving Gif at time step " << iT << endl;
            writeGifs(lattice, iT);
        }
        // Execute lattice Boltzmann iteration.
        lattice.collideAndStream();
    }

    delete boundaryCondition;
}
