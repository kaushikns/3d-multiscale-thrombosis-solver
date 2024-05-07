/* 
Author: Kaushik N Shankar
*/

#include "palabos3D.h"
#include "palabos3D.hh"
#include <unordered_map>
#include <random>
#include "mui.h"

using namespace plb;
using namespace std;

typedef double T;
#define DESCRIPTOR descriptors::D3Q19Descriptor

plint extraLayer = 0;                  // Make the bounding box larger; for visualization purposes
                                       //   only. For the simulation, it is OK to have extraLayer=0.
const plint blockSize = 20;            // Zero means: no sparse representation.
const plint envelopeWidth = 1;         // For standard BGK dynamics.
const plint extendedEnvelopeWidth = 2; // Because the Guo off lattice boundary condition
                                       //   needs 2-cell neighbor access.

bool performOutput = false;
bool useIncompressible = false;
bool poiseuilleInlet = false;
bool convectiveScaling = false;
bool constantFlow = false;

plint margin = 2;      // Extra margin of allocated cells around the obstacle.
plint borderWidth = 1; // Because the Guo boundary condition acts in a one-cell layer.
                       // Requirement: margin>=borderWidth.

// Key variables used in the course of the simulation
// The variables are simply initialized here
// The values of each of these will be set and modifed as the simulation progresses

T kinematicViscosity = 0.;
T averageInletVelocity = 0.;
plint referenceResolution = 0;
plint resolution = 0;
T fluidDensity = 0.;
T userDefinedInletDiameter = 0.;
T referenceLength = 0.; // Reference length is always in the z direction in this case
T lx = 0.;
T ly = 0.;
T lz = 0.;
T inletP;
T outletP;

plint referenceDirection = 0;
plint openingSortDirection = 0;

T epsilon = 0.;

Array<T, 3> inletRealPos(0., 0., 0.);
Array<T, 3> outletRealPos(0., 0., 0.);

T diameterReal = 0.;
T radiusReal = 0.;

std::auto_ptr<MultiScalarField3D<int>> flagMatrixWall;
std::auto_ptr<MultiScalarField3D<int>> flagMatrixInside;
// This will be useful in determining whether a grid point lies inside or outside the domain
TriangleSet<T> *triangleSet = 0;
Array<T, 3> KMCLocation;
Array<T, 3> LBLocation;
std::auto_ptr<MultiScalarField3D<int>> boolMask;

T err = 1.;

T dx = 0.;                          // LB grid size
T dt = 0.;                          // LB time step
T h = 0.;                           // LKMC grid size
plint nx = 0, ny = 0, nz = 0;       // LKMC resolution
plint nxlb = 0, nylb = 0, nzlb = 0; // LB resolution
plint factor = 0;                   // Ratio of LB grid to KMC grid size
plint diaFactor = 0;                // Ratio of platelet diameter to KMC grid size
T pltDia = 3.e-6;                   // Platelet diameter

plint level = 0; // Level of LKMC grid refinement needed
// This is preferably kept at level = 0
// The grid size can be changed if needed as input from "param.xml"

plint nPlatelets = 0;      // Number of platelets in the system at any given time
plint nBoundPlatelets = 0; // Number of bound platelets in the system at any given time

// Base rates for platelet attachment and detachment
// THese need to be multiplied or divided with the activation state to get the rate for each platelet
T k_att_col = 0.;
T k_det_col = 0.;
T k_att_fib = 0.;
T k_det_fib = 0.;

// Characterisitc shear rate for platelet detachment rates
T charShear = 0.;

// Dispersion coefficient of each platelet
T diffusivity = 0.;

// Diffusive rate of motion
T diffusion = 0.;

// Maximum number of LKMC events per platelet
plint nEventsPerPlatelet = 6;

// Variable to keep track of time
T totalTime = 0.;
plint iter = 0;

// A temporary variable used to assign and store local velocity
Array<T, 3> velocity(0., 0., 0.);

// A temporary variable used to assign and store local platelet drift velocity
Array<T, 3> driftVelocity(0., 0., 0.);

// Mixing cup inlet flow rate
T inletRate = 0.;
T oldInletRate = 0.;

// Radial platelet concentration cumulative distribution function
vector<T> radPos;        // Scaled radial position
vector<T> pdf;           // Distribution as a function of radial position
vector<T> cdf;           // Cumulative distribution as a function of radial position
vector<T> driftFunction; // Drift velocity as a function of radial position
T normalization = 0.;    // Factor by which the probability dist must be normalized

vector<T> thrombinConc; // Concentration of free thrombin as a function of time

T dblmax = std::numeric_limits<T>::max();
T machineEps = std::numeric_limits<T>::epsilon();

// Random number generator
std::mt19937 engine(21);
std::mt19937 engine_inlet(14);
std::uniform_real_distribution<T> distribution(0., 1.);

// NN parameters
std::string nnInputFile;
Array<Array<T, 6>, 8> IW;
Array<Array<T, 8>, 8> A1;
Array<Array<T, 1>, 8> b1;
Array<Array<T, 8>, 4> LW1;
Array<Array<T, 8>, 4> A2;
Array<Array<T, 1>, 4> b2;
Array<Array<T, 4>, 1> LW2;
Array<Array<T, 1>, 1> b3;
Array<T, 4> ec50;

T dtNN = 0.1;

// Parameters to determine the activation state of a platelet
// from the calcium integral function xi(t)
T alpha = 0.;
T nHill = 0.;
T xi50 = 0.;
T relPotencyTXA2 = 0.;     // Relative potency of TXA2/U46619
bool iloprostFlag = false; // Flag for iloprost treatment
bool gsnoFlag = false;     // Flag for GSNO treatment

// Structure which defines an ``opening''. The surface geometry of the vessel,
//   as given by the user in the form of an STL file, contains holes, which in
//   the specific simulation represent inlets and outlets.
template <typename T>
struct Opening
{
    bool inlet;
    Array<T, 3> center;
    T innerRadius;
};

std::vector<Opening<T>> openings;

// A class which holds the LB Lattice: the Palabos MultiBlock
class LBClass
{
public:
    std::auto_ptr<MultiBlockLattice3D<T, DESCRIPTOR>> lattice;
    std::auto_ptr<MultiScalarField3D<T>> shearRate;
    std::auto_ptr<VoxelizedDomain3D<T>> voxelizedDomain;

    LBClass()
    {
        lattice.reset();
    }

    void getVoxelizedDomain(plint level = 0);

    void assign(std::auto_ptr<MultiBlockLattice3D<T, DESCRIPTOR>> lattice_)
    {
        lattice.reset();
        lattice = lattice_;
        updateShear();
    }
    void updateShear()
    {
        //auto velField = *computeVelocity(*lattice);
        auto strainRate = *computeStrainRateFromStress(*lattice);
        auto shearRate_ = *computeSymmetricTensorNorm(strainRate, lattice->getBoundingBox());
        shearRate.reset(new MultiScalarField3D<T>(shearRate_));
        //shearRate = computeSymmetricTensorNorm(*computeStrainRate(velField, lattice -> getBoundingBox()),
        //    lattice -> getBoundingBox());
    }
};

void LBClass::getVoxelizedDomain(plint level)
{
    referenceResolution = (plint)(referenceLength / dx);
    resolution = referenceResolution * util::twoToThePower(level);
    // The resolution is doubled at each coordinate direction with the increase of the
    //   resolution level by one. The parameter ``referenceResolution'' is by definition
    //   the resolution at grid refinement level 0.

    // The next few lines of code are typical. They transform the surface geometry of the
    //   aneurysm given by the user to more efficient data structures that are internally
    //   used by palabos. The TriangleBoundary3D structure will be later used to assign
    //   proper boundary conditions.
    DEFscaledMesh<T> *defMesh =
        new DEFscaledMesh<T>(*triangleSet, resolution, referenceDirection, margin, extraLayer);
    TriangleBoundary3D<T> boundary(*defMesh);
    delete defMesh;
    boundary.getMesh().inflate();

    dx = boundary.getDx();
    T nuLB_ = dt * kinematicViscosity / (dx * dx);
    T uAveLB = averageInletVelocity * dt / dx;
    T omega = 1. / (3. * nuLB_ + 0.5);
    Array<T, 3> location(boundary.getPhysicalLocation());
    LBLocation = location;

    /*
    pcout << "uLB=" << uAveLB << std::endl;
    pcout << "nuLB=" << nuLB_ << std::endl;
    pcout << "tau=" << 1./omega << std::endl;
    if (performOutput) {
        pcout << "dx=" << dx << std::endl;
        pcout << "dt=" << dt << std::endl;
    }
    */

    // The aneurysm simulation is an interior (as opposed to exterior) flow problem. For
    //   this reason, the lattice nodes that lay inside the computational domain must
    //   be identified and distinguished from the ones that lay outside of it. This is
    //   handled by the following voxelization process.
    const int flowType = voxelFlag::inside;

    VoxelizedDomain3D<T> voxelizedDomain_(
        boundary, flowType, extraLayer, borderWidth, extendedEnvelopeWidth, blockSize);

    voxelizedDomain.reset(new VoxelizedDomain3D<T>(voxelizedDomain_));
    /*
    if (performOutput) {
        pcout << getMultiBlockInfo(voxelizedDomain.getVoxelMatrix()) << std::endl;
    }
    */

    // Super important. Identifies the walls of the domain based on the flag

    flagMatrixInside.reset(new MultiScalarField3D<int>((MultiBlock3D &)voxelizedDomain->getVoxelMatrix()));
    flagMatrixWall.reset(new MultiScalarField3D<int>((MultiBlock3D &)voxelizedDomain->getVoxelMatrix()));

    setToConstant(*flagMatrixInside, voxelizedDomain->getVoxelMatrix(),
                  voxelFlag::inside, flagMatrixInside->getBoundingBox(), 1);
    setToConstant(*flagMatrixInside, voxelizedDomain->getVoxelMatrix(),
                  voxelFlag::innerBorder, flagMatrixInside->getBoundingBox(), 1);

    setToConstant(*flagMatrixWall, voxelizedDomain->getVoxelMatrix(),
                  voxelFlag::innerBorder, flagMatrixWall->getBoundingBox(), 1);

    //pcout << "Number of fluid cells: " << computeSum(*flagMatrixInside) << std::endl;

    nxlb = ((MultiBlock3D &)voxelizedDomain->getVoxelMatrix()).getNx();
    nylb = ((MultiBlock3D &)voxelizedDomain->getVoxelMatrix()).getNy();
    nzlb = ((MultiBlock3D &)voxelizedDomain->getVoxelMatrix()).getNz();
}

LBClass LB;

// A structure that defines information about a platelet
struct platelet
{
    Array<plint, 3> center;         // Position of the platelet
    T dp;                           // Platelet diameter
    Array<T, 3> inPos;              // Inlet position
    T inletTime;                    // Time at which the platelet is inlet into the domain
    unsigned long long int inCount; // Counter for the number of platelets entering the domain

    // KMC event specific parameters
    Array<T, 6> convection;                           // Convection rates in each direction
    Array<T, 6> motion;                               //  Motion rates in each direction // 0,2,4 +ve x,y,z; 1,3,5 -ve x,y,z
    Array<plint, 6> blockCount;                       // No. of platelets blocking motion in each direction
    T binding;                                        // Net binding rate with collagen or any other neighboring platelet
    T unbinding;                                      // Effective unbinding rate of platelet from clot mass
    std::list<list<platelet>::iterator> possibleBond; // Link to platelets available for binding
    bool bound;                                       // Binding state of platelet (whether it is bound and there is connectivity collagen / TF)
    Array<T, 8> tau;                                  // Putative time after which each event occurs (6 motion, 1 binding and 1 unbinding)
    T localShear;                                     // Local shear around a platelet
    T boundTime;                                      // Time at which platelet was first bound
    T nbonds;

    // Parameters particular to reactive agonist species
    Array<T, 4> concentrations;
    // 0:ADP, 1:Coll, 2:Thrombin, 3:TXA2
    T releaseTime; // Time after which platelet starts releasing agonist species

    // Parameters estimated using NN (activation state)
    T calcInteg;                                   // Integral of intraplatelet calcium (xi)
    T activation;                                  // Activation state of platelet [F(xi)]
    T activationx;                                 // Recent history activation
    std::unordered_map<plint, T> nnHistory;        // NN history of the platelet, an input into NN which determines the activation
    std::unordered_map<plint, T> calcIntegHistory; // Calcium integral history of the platelet
};

// List of platelets
list<platelet> plateletList;

list<platelet> dummyPlateletList;
T inletTau = dblmax; // Putative time for inlet of platelet into domain
list<platelet>::iterator inletIter;
// Dummy platelet list to which the inlet of new platelet into domain is associated
// This list will contain only one platelet, which is the dummyplatelet

// List to keep track of platelets that were bound in within
// succesive iterations of the LB/NN/FVM solvers
list<list<platelet>::iterator> boundList;

// Flag for whether an unbinding event occurred between successive iterations of LB/FVM/NN
plint nUnbindingOccurred = 0;

// A variable used to keep track of platelet occupation as function of lattice spacing
std::unordered_map<long long int, std::list<platelet>::iterator> occupation;

// Inlet count of platelets
long long int inCount = 0;

// A structure which is used to store the events in the sorted event list
struct event
{
    list<platelet>::iterator address;
    plint type;
    T tau;

    event(list<platelet>::iterator address_, plint type_, T tau_) : address(address_), type(type_), tau(tau_)
    {
    }
};

// This structure is used to dictate how the events are ordered in the list
// In this case, they are sorted according to increasing tau values (next reaction method)
struct orderEvent
{
    bool operator()(const event &e1, const event &e2) const
    {
        /*
        if (e1.tau == e2.tau)
        {
            if (e1.type > e2.type)
            {
                return e1.address -> inCount < e2.address -> inCount;
            }

            return e1.type < e2.type;
        }*/
        return e1.tau < e2.tau;
    }
};

std::set<event, orderEvent> eventList;

// This struct keeps track of inlet events into the domain
struct inletEvent
{
    T x;
    T y;
    T tau;

    inletEvent(T x_, T y_, T tau_) : x(x_), y(y_), tau(tau_)
    {
    }
};

// This is used to dictate how the inlet events are ordered (according to next
// reaction method in this case)
struct orderInletEvent
{
    bool operator()(const inletEvent &e1, const inletEvent &e2) const
    {
        if (e1.tau == e2.tau)
        {
            if (e1.x == e2.x)
            {
                return e1.y < e2.y;
            }
            return e1.x < e2.x;
        }
        return e1.tau < e2.tau;
    }
};

std::set<inletEvent, orderInletEvent> inletEventList;

// Read the NN input parameters from XML file
template <pluint m, pluint n>
void readNNParameters(std::string paramName, Array<Array<T, n>, m> &arr);

// Read the user input XML file provided.
void readParameters();

// Note: This is just a dummy. Used by an internal Palabos class to determine whether a given
// point lies inside or outside the domain
// This function assigns proper boundary conditions to the openings of the surface geometry
// Which opening is inlet and which is outlet is defined by the user in the input XML file.
void setOpenings(
    std::vector<BoundaryProfile3D<T, Array<T, 3>> *> &inletOutlets,
    TriangleBoundary3D<T> &boundary, T uLB);

/************************** Neural Network NN routines *********************************/

// Matrix manipulation routines

// Matrix - matrix multiplication
template <pluint m, pluint n, pluint p, pluint q>
Array<Array<T, q>, m> matMultiply(Array<Array<T, n>, m> mat1, Array<Array<T, q>, p> mat2);

// Matrix - matrix addition
template <pluint m, pluint n>
Array<Array<T, n>, m> matAdd(Array<Array<T, n>, m> mat1, Array<Array<T, n>, m> mat2);

// This function computes the tanh of all the elements in a matrix
// tanh is the transformation function used in the NN
template <pluint m, pluint n>
Array<Array<T, n>, m> transform(Array<Array<T, n>, m> mat);

// This function is used to map concentrations of agonists between -1 & 1
T mapConc(T conc, plint id);

// This function is used to map calcium concentration from NN output
T mapCalc(T output);

// Computing NN output
T nnOutput(platelet &plt, Array<T, 8> nnHistory);

// This function is used to obtain the NN history of a platelet at time t0
T obtainHistory(platelet &plt, plint t0);

// This function is used to obtain the recent history calcium integral of a platelet
T obtainRecentCalcInteg(platelet &plt, plint t0);

// Integrate calcium concentration and determine activation state
void integrateNN();

/************************** Lattice Kinetic MC routines ********************************/

// This function generates the 3D lattice and grid points for performing the simulation
void generateKMCLattice(plint level = 0);

// Initialize platelet with parameters so that garbage values are not carried over by struct variables
void initializePlatelet(platelet &plt);

// Generate random number between 0 and 1
T randn();

// Generate random number between 0 and 1
T randn_inlet();

// Floating point number comparison
bool areEqual(T a, T b);

bool lessThan(T a, T b);

// Updates the putative time of the event occurrence given old and new rates
T updateEventTime(T tau, T rateOld, T rateNew);

// Returns the square of the distance between the centers of two platelets in lattice units
plint distanceBetween(platelet a, platelet b);

// Tells if a given lattice point lies inside the platelet or not
bool liesInside(Array<plint, 3> position, platelet plt);

// This function checks if there are overlaps between any two given platelets
bool checkOverlap(platelet a, platelet b);

// This function checks if there are overlaps between two platelets
// The above implementation is used for the convective PFA
// The implementation below is used for checking available platelets for binding
// This difference is due to the difference in the PFA algorithm and the binding algorithm
bool checkOverlapBinding(platelet a, platelet b);

// This function takes as argument a struct platelet
// If there is any overlap with another platelet, then it returns true
bool checkOverlap(platelet plt);

// This function checks if there are overlaps between a platelet and any other platelet
// If yes, it returns the head of a linked list containing their addresses
list<list<platelet>::iterator> checkOverlap(list<platelet>::iterator current);

// This function checks if there are overlaps between a platelet and any other platelet
// If yes, it returns the head of a linked list containing their addresses
// The above implementation is used for the convective PFA
// The implementation below is used for checking available platelets for binding
// This difference is due to the difference in the PFA algorithm and the binding algorithm
list<list<platelet>::iterator> checkOverlapBinding(list<platelet>::iterator current);

// This function is used to assign the velocity given the position of the platelet center
inline void obtainVelocity(plint cx, plint cy, plint cz);

// This function is used to assign velocity given a position in space
inline void obtainVelocity(T x, T y, T z);

// Checks whether a platelet has left the domain
bool checkWithinDomain(plint cx, plint cy, plint cz);

// Checks whether a platelet is at any domain wall
bool checkAtWall(plint cx, plint cy, plint cz);

// Checks whether a platelet can interact with the reactive patch
// This function is subject to change with a change in the location of the patch
bool checkCollagen(plint cx, plint cy, plint cz);

// This function ensures that all the rates are defined such that
// a move where a platelet leaves the domain is not allowed
void checkWithinDomain(list<platelet>::iterator current);

// This function implements the pass forward algorithm (Flamm et al. J Chem Phys 2009, 2011)
// Resets convective rates for a given direction if motion is blocked by another platelet
void passForward(list<platelet>::iterator current, plint dir);

// This function finds if any platelet was blocked from moving by the platelet which just moved.
// If there exists such a platelet, then it resets the motion of the platelet from 0 to the apt value
// Furthermore, due to the motion of platelets, there may be the possibility that this platelet gets blocked
// It implements the pass forward algorithm in these cases.
void passForwardMotion(list<platelet>::iterator current, plint dir);

// This function implements the simple blocking algorithm SBA(Flamm et al. J Chem Phys 2009)
// Sets convective rate to zero if motion is blocked by another platelet in a givne direction
// More computationally efficient than the pass forward algorithm
// Inaccurate physical representation due to blocking is insignificant because platelet density is low
void simpleBlocking(list<platelet>::iterator &current, plint dir);

// The concentration probability density function must be normalized
// This function returns the integral of the PDF so it can be used as
// denominator for normalization
void normalizeConcentration();

// Computes the effective net flow rate of platelets into the domain
T computeFlowRate();

// Computes the inlet flow rate at given coordinates x,y
void computeFlowRate(T x, T y);

// Binary search function to estimate the index of radPos
plint binSearch(plint first, plint last, T val);

// This function adds a platelet randomly in the domain
// Used to initialize KMC
// To add platelets at inlet, refer to function: addPlateletInlet instead
void addPlatelet();

// This function is used to add a new platelet to the domain
// Add a biasing function so platelets enter the domain near the wall later on
void addPlateletInlet();

// Computes the enhancement in binding rates due to vwf stretching
T vwfFactor(list<platelet>::iterator current);

// Computes the enhancement in detachment rate due to shear
T detachFactor(list<platelet>::iterator current);

// This functional is used to set the binding rates associated with
// the given platelet as the argument
void setBindingRates(list<platelet>::iterator current);

// This functional is used to set the unbinding rates associated with
// the given platelet as the argument
void setUnbindingRates(list<platelet>::iterator current);

// This function sets the rate of each event for a given platelet that moves
// More efficient to use this when the velocity field does not change
void setRates(list<platelet>::iterator current, plint type);

// This function sets the rate of each event for each platelet
// Alters the rate parameters where the move is not allowed
// by calling the convective pass-forward algorithm routine
void setRates();

// Used to initalize the LKMC lattice with given initial platelet distribution
void iniKMCLattice();

// This function runs one iteration of the LKMC algorithm
void runKMC();

/******************************* LB routines ***********************************/

/// A functional used to assign Poiseuille velocity profile at inlet
T poiseuilleVelocity(plint iX, plint iY, T uLB);

class PoiseuilleVelocity3D
{
public:
    PoiseuilleVelocity3D(T uLB_) : uLB(uLB_)
    {
    }

    void operator()(plint iX, plint iY, plint iZ, Array<T, 3> &u) const
    {
        u[0] = T();
        u[1] = T();
        u[2] = poiseuilleVelocity(iX, iY, uLB);
    }

private:
    T uLB;
};

/// A functional, used to create an initial condition for velocity based on Poiseuille flow
template <typename T>
class initializeDensityAndVelocity
{
public:
    initializeDensityAndVelocity(T uLB_) : uLB(uLB_)
    {
    }

    void operator()(plint iX, plint iY, plint iZ, T &rho, Array<T, 3> &u) const
    {

        u[0] = T();
        u[1] = T();
        u[2] = poiseuilleVelocity(iX, iY, uLB);
        rho = 1.;
    }

private:
    T uLB;
};

// A function that returns whether a LB lattice point lies inside a given platelet or not
// This is used to instantiate bounceback nodes in the interior nodes of the platelet by the
// domain processing functional below
bool liesInside(Array<plint, 3> pos, Array<plint, 3> center, T dia);

// A functional used to instantiate bounceback nodes at fluid cells
// that lie inside the bound platelets
template <typename T>
class plateletShapeDomain : public DomainFunctional3D
{
public:
    plateletShapeDomain(Array<plint, 3> center_, T dia_) : center(center_), dia(dia_)
    {
    }

    // The function-call operator is overridden to specify the location
    // of bounce-back nodes.

    virtual bool operator()(plint iX, plint iY, plint iZ) const;

    virtual plateletShapeDomain<T> *clone() const;

private:
    Array<plint, 3> center;
    T dia;
};

// This function obtains the largest cuboidal bounding box that surrounds a platelet
Box3D boundingBox(platelet &plt);

void updateBoolMask(platelet &plt, int val);

// This function assigns no slip boundary conditions at platelet surfaces
// The platelet positipons will be obtained from LKMC
// void addBounceBack (MultiBlockLattice3D<T,DESCRIPTOR>& lattice);

/// Write the full velocity and the velocity-norm into a VTK file.
void writeVTK(MultiBlockLattice3D<T, DESCRIPTOR> &lattice,
              T dx, T dt, plint iter);

// This function is used to compute the local shear around a platelet
T computePltShear(list<platelet>::iterator current);

/// This is the function that prepares the actual LB simulation.
void updateLB(MultiBlockLattice3D<T, DESCRIPTOR> &lattice);

// This is the function that prepares and performs the actual simulation.
void runLB(plint level = 0);

// Write out the velocity field and platelet positions for visualization
void writeOutputFiles();

// This function is used to pass information to OpenFOAM concentration
// solver using MUI as the message passing toolkit
void interface_solution(std::vector<std::unique_ptr<mui::uniface1d>> &ifs,
                        mui::chrono_sampler_exact1d chrono_sampler);

T pfatime = 0.;
T bindingratesettime = 0.;
T bouncebacksettime = 0.;
T lbconvergencetime = 0.;

int main(int argc, char *argv[])
{
    // Initialization of MUI interface

    MPI_Comm world = mui::mpi_split_by_app();
    std::string dom = "kmcnnlb";
    std::vector<std::string> interfaces;
    interfaces.emplace_back("mpi://kmc_fvm/ifs");
    interfaces.emplace_back("mpi://fvm_nn/ifs");
    interfaces.emplace_back("mpi://fvm_lb/ifs");
    interfaces.emplace_back("mpi://lb_fvm/ifs");
    auto ifs = mui::create_uniface<mui::config_1d>(dom, interfaces);

    mui::chrono_sampler_exact1d chrono_sampler;
    mui::sampler_exact1d<plint> spatial_sampler;

    //plbInit(&argc, &argv);
    global::mpi().init(world);

    global::directories().setOutputDir("tmp");

    global::IOpolicy().activateParallelIO(true);

    // Read the simulation parameters from XML file
    readParameters();

    global::timer("runtime").start();

    LB.getVoxelizedDomain();

    runLB();

    writeVTK(*LB.lattice, dx, dt, iter);

    pcout << "Factor = " << factor << endl;

    interface_solution(ifs, chrono_sampler);

    generateKMCLattice(level);

    pcout << "KMC lattice generated" << endl;

    // Diffusive rate of motion
    //(This remains a constant throughout the simulation because both
    // the dispersion coeff. and grid size do not change)
    diffusion = diffusivity / (h * h);

    normalizeConcentration();

    inletRate = computeFlowRate();

    pcout << "Inlet rate computed " << inletRate << endl;

    iniKMCLattice();

    pcout << "KMC lattice initialized with platelets" << endl;

    setRates();

    T dtLB = 0.1;

    T lb_time = 0.;
    T nn_time = 0.;
    T ifs_time = 0.;
    T ratesettime = 0.;
    T rateupdatetime = 0.;
    T flowratetime = 0.;

    plint iterNN = 0;

    pcout << "Beginning KMC simulation" << endl;

    while (totalTime < 500.)
    {
        global::timer("rateset").restart();
        runKMC();
        ratesettime = ratesettime + global::timer("rateset").getTime();

        if (totalTime >= iterNN * dtNN)
        {
            pcout << "Updating NN history" << endl;
            global::timer("NN").restart();
            integrateNN();
            ++iterNN;
            nn_time = nn_time + global::timer("NN").getTime();
        }

        if (totalTime >= iter * dtLB && iter != floor(500. / dtLB))
        {
            pcout << "Running LB solver" << endl;
            global::timer("LB").restart();
            runLB();
            pcout << "LB solver converged successfully" << endl;
            lb_time = lb_time + global::timer("LB").getTime();

            pcout << "Computed fluid flow rate" << endl;
            global::timer("flowrate").restart();
            oldInletRate = inletRate;
            inletRate = computeFlowRate();
            flowratetime = flowratetime + global::timer("flowrate").getTime();

            pcout << "Setting up interaction between LB, KMC modules with FVM module" << endl;
            global::timer("interface").restart();
            interface_solution(ifs, chrono_sampler);
            ifs_time = ifs_time + global::timer("interface").getTime();
            pcout << "Interaction achieved successfully, starting next iteration" << endl;

            pcout << "Setting KMC activation rates based on updated NN history" << endl;
            global::timer("rateupdate").restart();
            setRates();
            rateupdatetime = rateupdatetime + global::timer("rateupdate").getTime();

            pcout << "Number of platelets in domain: " << nPlatelets << endl;
            pcout << "Number of bound platelets: " << nBoundPlatelets << endl;
            pcout << "Number of unbindings: " << nUnbindingOccurred << endl;

            nUnbindingOccurred = 0;

            if (iter % 10 == 0)
                writeOutputFiles();

            pcout << "Simulation data successfully dumped into output files" << endl;

            pcout << "Interface time = " << ifs_time << endl;

            pcout << "LB time = " << lb_time << endl;

            pcout << "Bounceback node set time = " << bouncebacksettime << endl;

            pcout << "LB Convergence time = " << lbconvergencetime << endl;

            pcout << "NN time = " << nn_time << endl;

            pcout << "Rate update time = " << rateupdatetime << endl;

            pcout << "Rate set time = " << ratesettime << endl;

            pcout << "PFA time = " << pfatime << endl;

            pcout << "Binding rate set time = " << bindingratesettime << endl;

            pcout << "Flow rate set time = " << flowratetime << endl;

            pcout << "Program run time = " << global::timer("runtime").getTime() << endl;
        }
    }

    delete triangleSet;

    return 0;
}

// All function definitions can be found here

// Read the NN input parameters from XML file
template <pluint m, pluint n>
void readNNParameters(std::string paramName, Array<Array<T, n>, m> &arr)
{
    std::vector<T> temp;
    pluint ind = 0;
    pluint i, j;
    XMLreader document(nnInputFile);

    document[paramName].read(temp);
    for (i = 0; i < m; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            arr[i][j] = temp[ind];
            ++ind;
        }
    }
}

// Read the user input XML file provided.
void readParameters()
{
    std::string paramXmlFileName = "/home/kaushik/stenosis/param.xml";

    // Read the parameter XML input file.
    XMLreader document(paramXmlFileName);

    std::string meshFileName;
    std::vector<std::string> openingType;
    document["geometry"]["mesh"].read(meshFileName);
    document["geometry"]["averageInletVelocity"].read(averageInletVelocity);
    document["geometry"]["openings"]["sortDirection"].read(openingSortDirection);
    document["geometry"]["openings"]["type"].read(openingType);
    document["geometry"]["inlet"]["x"].read(inletRealPos[0]);
    document["geometry"]["inlet"]["y"].read(inletRealPos[1]);
    document["geometry"]["inlet"]["z"].read(inletRealPos[2]);
    document["geometry"]["outlet"]["x"].read(outletRealPos[0]);
    document["geometry"]["outlet"]["y"].read(outletRealPos[1]);
    document["geometry"]["outlet"]["z"].read(outletRealPos[2]);
    document["geometry"]["inletDia"].read(diameterReal);

    document["fluid"]["kinematicViscosity"].read(kinematicViscosity);
    document["fluid"]["density"].read(fluidDensity);

    document["kmc"]["numPlatelets"].read(nPlatelets); // Initial number of platlets in the domain
    document["kmc"]["collagenAttach"].read(k_att_col);
    document["kmc"]["collagenDetach"].read(k_det_col);
    document["kmc"]["fibAttach"].read(k_att_fib);
    document["kmc"]["fibDetach"].read(k_det_fib);
    document["kmc"]["gammac"].read(charShear);
    document["kmc"]["diffusivity"].read(diffusivity);

    document["nn"]["net"].read(nnInputFile);
    document["nn"]["xi50"].read(xi50);
    document["nn"]["alpha"].read(alpha);
    document["nn"]["nHill"].read(nHill);
    document["nn"]["ec50"]["adp"].read(ec50[0]);
    document["nn"]["ec50"]["cvx"].read(ec50[1]);
    document["nn"]["ec50"]["thrombin"].read(ec50[2]);
    document["nn"]["ec50"]["u46619"].read(ec50[3]);
    document["nn"]["theta"].read(relPotencyTXA2);
    document["nn"]["iloprostflag"].read(iloprostFlag);
    document["nn"]["gsnoflag"].read(gsnoFlag);

    document["numerics"]["referenceDirection"].read(referenceDirection);
    document["numerics"]["referenceLength"].read(referenceLength);
    document["numerics"]["hLKMC"].read(h);
    document["numerics"]["hLB"].read(dx);
    //document["numerics"]["referenceResolution"].read(referenceResolution);
    document["numerics"]["dt"].read(dt);

    document["simulation"]["epsilon"].read(epsilon);
    document["simulation"]["performOutput"].read(performOutput);
    document["simulation"]["useIncompressible"].read(useIncompressible);
    document["simulation"]["poiseuilleInlet"].read(poiseuilleInlet);
    document["simulation"]["convectiveScaling"].read(convectiveScaling);
    document["simulation"]["constantFlow"].read(constantFlow);

    radiusReal = 0.5 * diameterReal;
    lz = outletRealPos[2] - inletRealPos[2];

    // Read the platelet inlet distribution function from the input xml file
    std::string inletConcFile = "/home/kaushik/stenosis/inlet_conc.xml";
    XMLreader document2(inletConcFile);
    document2["radPos"].read(radPos);
    document2["cdf"].read(cdf);
    document2["pdf"].read(pdf);
    document2["drift"].read(driftFunction);
    for (plint i = 0; i < driftFunction.size(); ++i)
    {
        driftFunction[i] = driftFunction[i] * diffusivity;
    }

    // Read the NN matrices from the input xml file
    readNNParameters("IW", IW);
    readNNParameters("A1", A1);
    readNNParameters("b1", b1);
    readNNParameters("LW1", LW1);
    readNNParameters("A2", A2);
    readNNParameters("b2", b2);
    readNNParameters("LW2", LW2);
    readNNParameters("b3", b3);

    // Read free thrombin concentration data from Chen et al, Plos Comput Biol (2019)

    std::string thrombinConcFile;
    document["thrombin"].read(thrombinConcFile);
    XMLreader document3(thrombinConcFile);
    document3["IIa"].read(thrombinConc);

    // At this part, the surface geometry of the aneurysm (as given by the user in
    //   the form of an ASCII or binary STL file) is read into a data structure
    //   comprised by a set of triangles. The DBL constant means that double
    //   precision accuracy will be used (generally the recommended choice).
    triangleSet = new TriangleSet<T>(meshFileName, DBL);

    plbIOError(openingSortDirection < 0 || openingSortDirection > 2,
               "Sort-direction of opening must be 0 (x), 1 (y), or 2 (z).");
    // The surface geometry, as provided by the STL file, must contain openings,
    //   namely inlets and outlets. On these openings, appropriate boundary conditions
    //   will be imposed by palabos. Which opening is inlet and which is outlet, is
    //   identified by the user in the input XML file.

    openings.resize(openingType.size());
    for (pluint i = 0; i < openingType.size(); ++i)
    {
        std::string next_opening = util::tolower(openingType[i]);
        if (next_opening == "inlet")
        {
            openings[i].inlet = true;
        }
        else if (next_opening == "outlet")
        {
            openings[i].inlet = false;
        }
        else
        {
            plbIOError("Unknown opening type.");
        }
    }
}

// Note: This is just a dummy. Used by an internal Palabos class to determine whether a given
// point lies inside or outside the domain
// This function assigns proper boundary conditions to the openings of the surface geometry
// Which opening is inlet and which is outlet is defined by the user in the input XML file.
void setOpenings(
    std::vector<BoundaryProfile3D<T, Array<T, 3>> *> &inletOutlets,
    TriangleBoundary3D<T> &boundary, T uLB)
{
    for (pluint i = 0; i < openings.size(); ++i)
    {
        Opening<T> &opening = openings[i];
        opening.center = computeBaryCenter(
            boundary.getMesh(),
            boundary.getInletOutlet(openingSortDirection)[i]);
        opening.innerRadius = computeInnerRadius(
            boundary.getMesh(),
            boundary.getInletOutlet(openingSortDirection)[i]);

        inletOutlets.push_back(new VelocityPlugProfile3D<T>(uLB));
    }
}

/************************** Neural Network NN routines *********************************/

/************************** Neural Network NN routines *********************************/

// Matrix manipulation routines

// Matrix - matrix multiplication
template <pluint m, pluint n, pluint p, pluint q>
Array<Array<T, q>, m> matMultiply(Array<Array<T, n>, m> mat1, Array<Array<T, q>, p> mat2)
{
    Array<Array<T, q>, m> mat3;
    for (pluint i = 0; i < m; ++i)
    {
        for (pluint j = 0; j < q; ++j)
        {
            mat3[i][j] = 0;
            for (pluint k = 0; k < n; ++k)
            {
                mat3[i][j] = mat3[i][j] + mat1[i][k] * mat2[k][j];
            }
        }
    }

    return mat3;
}

// Matrix - matrix addition
template <pluint m, pluint n>
Array<Array<T, n>, m> matAdd(Array<Array<T, n>, m> mat1, Array<Array<T, n>, m> mat2)
{
    Array<Array<T, n>, m> mat3;
    for (pluint i = 0; i < m; ++i)
    {
        for (pluint j = 0; j < n; ++j)
        {
            mat3[i][j] = mat1[i][j] + mat2[i][j];
        }
    }

    return mat3;
}

// This function computes the tanh of all the elements in a matrix
// tanh is the transformation function used in the NN
template <pluint m, pluint n>
Array<Array<T, n>, m> transform(Array<Array<T, n>, m> mat)
{
    for (pluint i = 0; i < m; ++i)
    {
        for (pluint j = 0; j < n; ++j)
        {
            mat[i][j] = tanh(mat[i][j]);
        }
    }

    return mat;
}

// This function is used to map concentrations of agonists between -1 & 1
T mapConc(T conc, plint id)
{
    T cmax = 10. * ec50[id];
    T cmin = 0.01 * ec50[id];
    if (conc > cmax)
        conc = cmax;
    else if (conc < cmin)
        conc = cmin;

    conc = log(conc);
    cmax = log(cmax);
    cmin = log(cmin);

    T mapping = 2. * (conc - cmin) / (cmax - cmin);
    mapping = mapping - 1.;

    return mapping;
}

// This function is used to map calcium concentration from NN output
T mapCalc(T output)
{
    T cmax = 1.e-3;
    T cmin = 1.e-4;

    T mapping;

    if (output <= -1.)
    {
        mapping = cmin;
    }
    else if (output >= 1.)
    {
        mapping = cmax;
    }
    else
    {
        mapping = 0.5 * (cmin + cmax + output * (cmax - cmin));
    }

    return mapping;
}

// Computing NN output
T nnOutput(platelet &plt, Array<T, 8> nnHistory)
{
    Array<Array<T, 1>, 6> c;
    for (plint i = 0; i < 4; ++i)
    {
        c[i][0] = mapConc(plt.concentrations[i], i);
    }

    if (iloprostFlag)
    {
        c[4][0] = 1.;
    }
    else
    {
        c[4][0] = -1.;
    }

    if (gsnoFlag)
    {
        c[5][0] = 1.;
    }
    else
    {
        c[5][0] = -1.;
    }

    if (plt.calcInteg < 1.e-6 && c[0][0] + c[1][0] + c[2][0] + c[3][0] + c[4][0] + c[5][0] < -5.9999)
    {
        return -1.;
    }

    if (plt.calcInteg < 1.e-6 && (iloprostFlag || gsnoFlag) && c[0][0] + c[1][0] + c[2][0] + c[3][0] < -3.9999)
    {
        return -1.;
    }

    Array<Array<T, 1>, 8> z;
    for (plint i = 0; i < 8; ++i)
    {
        z[i][0] = nnHistory[i];
    }

    auto layer1 = transform<8, 1>(matAdd<8, 1>(matAdd<8, 1>(matMultiply<8, 6, 6, 1>(IW, c), matMultiply<8, 8, 8, 1>(A1, z)), b1));
    auto layer2 = transform<4, 1>(matAdd<4, 1>(matAdd<4, 1>(matMultiply<4, 8, 8, 1>(LW1, layer1), matMultiply<4, 8, 8, 1>(A2, z)), b2));

    auto y = matAdd<1, 1>(matMultiply<1, 4, 4, 1>(LW2, layer2), b3);

    T output = y[0][0];

    return output;
}

// This function is used to obtain the NN history of a platelet at time t0
T obtainHistory(platelet &plt, plint t0)
{
    auto historyMap = plt.nnHistory;
    auto iter = historyMap.find(t0);

    if (iter != historyMap.end())
    {
        return iter->second;
    }
    return -1.;
}

// This function is used to obtain the recent history calcium integral of a platelet
T obtainRecentCalcInteg(platelet &plt, plint t0)
{
    auto calcIntegMap = plt.calcIntegHistory;
    auto iter = calcIntegMap.find(t0 - 30);

    if (iter != calcIntegMap.end())
    {
        return iter->second;
    }
    return 0.;
}

// Integrate calcium concentration and determine activation state
void integrateNN()
{
    platelet plt;
    Array<T, 8> nnHistory;
    T output;
    plint currentTime = (plint)totalTime;
    currentTime += 1;
    T calciumConc;
    T hillFunc;

    for (auto current = plateletList.begin(); current != plateletList.end(); ++current)
    {
        plt = *current;

        if (plt.bound && checkCollagen(plt.center[0], plt.center[1], plt.center[2]))
        {
            plt.concentrations[1] = 5. * ec50[1];
        }
        else
        {
            plt.concentrations[1] = 0.;
        }

        // If the platelet lies within the region for the thin film approximation
        // Then, set the local thrombin concentration to the free thrombin conc.
        // From Chen at al, Plos Comput Biol, 2019
        T posX = h * plt.center[0] + LBLocation[0];
        T posY = h * plt.center[1] + LBLocation[1];
        T posZ = h * plt.center[2] + LBLocation[2];
        if (plt.center[2] > nz / 4 && plt.center[2] < 3 * nz / 4 && plt.center[1] < ny / 2)
        {
            T distaceFromCenter = sqrt(posX * posX + posY * posY);
            posZ = posZ * 1.e6;
            T r = 0.0006438 * posZ * posZ - 0.366 * posZ + 66.06;
            r = r * 1.e-6;
            if (r - distaceFromCenter < 1.5e-5)
            {
                plt.concentrations[2] = thrombinConc[10 * currentTime - 1] * 0.001;
            }
            else
            {
                plt.concentrations[2] = 0.;
            }
        }
        else
        {
            plt.concentrations[2] = 0.;
        }

        nnHistory[0] = obtainHistory(plt, currentTime - 1);
        nnHistory[1] = obtainHistory(plt, currentTime - 2);
        nnHistory[2] = obtainHistory(plt, currentTime - 4);
        nnHistory[3] = obtainHistory(plt, currentTime - 8);
        nnHistory[4] = obtainHistory(plt, currentTime - 16);
        nnHistory[5] = obtainHistory(plt, currentTime - 32);
        nnHistory[6] = obtainHistory(plt, currentTime - 64);
        nnHistory[7] = obtainHistory(plt, currentTime - 128);

        output = nnOutput(plt, nnHistory);
        if (output < -1.)
            output = -1.;
        else if (output > 1.)
            output = 1.;

        //pcout << output << endl;

        plt.nnHistory[currentTime] = output;

        calciumConc = mapCalc(output);

        // Riemann sum for integral of calc conc
        plt.calcInteg = plt.calcInteg + (calciumConc - 1.e-4) * dtNN;

        plt.calcIntegHistory[currentTime] = plt.calcInteg;

        T recentCalcInteg = obtainRecentCalcInteg(plt, currentTime);
        recentCalcInteg = plt.calcInteg - recentCalcInteg;

        // Determination of activation state based on calc. integ.
        hillFunc = pow(plt.calcInteg, nHill);
        hillFunc = hillFunc / (pow(xi50, nHill) + hillFunc);
        plt.activation = alpha + (1 - alpha) * hillFunc;

        // Determination of recent history activation state of the platelet
        hillFunc = pow(recentCalcInteg, 2.);
        hillFunc = hillFunc / (pow(0.005, 2.) + hillFunc);
        plt.activationx = 1. + 49. * hillFunc;
        //plt.activationx = plt.activationx * plt.activationx;

        // Set release times if platelet is more than half-activated
        if (plt.calcInteg >= xi50 && plt.releaseTime < 0.)
        {
            plt.releaseTime = totalTime;
        }

        *current = plt;
    }
}

/************************** Lattice Kinetic MC routines ********************************/

// This function generates the 3D lattice and grid points for performing the simulation
void generateKMCLattice(plint level)
{
    factor = util::roundToInt(dx / h);
    //h = dx / (T) factor;
    diaFactor = util::roundToInt(pltDia / h);

    referenceResolution = (plint)(referenceLength / h);
    plint resolution = referenceResolution * util::twoToThePower(level);

    // The next few lines of code are typical. They transform the surface geometry of the
    //   vessel given by the user to more efficient data structures that are internally
    //   used by palabos. The TriangleBoundary3D structure will be later used to assign
    //   proper boundary conditions.
    DEFscaledMesh<T> *defMesh =
        new DEFscaledMesh<T>(*triangleSet, resolution, referenceDirection, factor * margin, extraLayer);
    TriangleBoundary3D<T> boundary(*defMesh);
    delete defMesh;
    boundary.getMesh().inflate();

    //T hKMC = boundary.getDx();
    //h = hKMC;

    Array<T, 3> location(boundary.getPhysicalLocation());
    KMCLocation = location;

    nx = nxlb * factor;
    ny = nylb * factor;
    nz = nzlb * factor;
}

// Initialize platelet with parameters so that garbage values are not carried over by struct variables
void initializePlatelet(platelet &plt)
{
    for (plint i = 0; i < 6; ++i)
    {
        plt.convection[i] = 0.;
        plt.motion[i] = 0.;
        plt.blockCount[i] = 0;
    }

    plt.binding = 0.;
    plt.unbinding = 0.;

    for (plint i = 0; i < 8; ++i)
    {
        plt.tau[i] = dblmax;
    }
    for (plint i = 0; i < 4; ++i)
    {
        plt.concentrations[i] = 0.;
    }
    plt.bound = false;
    plt.inCount = inCount++;
    plt.inletTime = totalTime;
    plt.releaseTime = -1.;
    plt.calcInteg = 0.;
    plt.activation = alpha;
    plt.activationx = 2.;
    plt.localShear = 0.;
    plt.boundTime = dblmax;
    plt.nbonds = 0;

    plateletList.push_front(plt);
    std::pair<long long int, std::list<platelet>::iterator> pltPair(
        (long long int)ny * nz * plt.center[0] + (long long int)nz * plt.center[1] + (long long int)plt.center[2],
        plateletList.begin());
    occupation.insert(pltPair);
}

// Generate random number between 0 and 1
T randn()
{
    return distribution(engine);
}

// Generate random number between 0 and 1
T randn_inlet()
{
    return distribution(engine_inlet);
}

// Floating point number comparison
bool areEqual(T a, T b)
{
    return fabs(a - b) <= ((fabs(a) > fabs(b) ? fabs(b) : fabs(a)) * machineEps);
}

bool lessThan(T a, T b)
{
    return (b - a) > ((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * machineEps);
}

// Updates the putative time of the event occurrence given old and new rates
T updateEventTime(T tau, T rateOld, T rateNew)
{
    if (rateNew < 1.e-10)
    {
        tau = dblmax;
    }
    else if (rateOld < 1.e-10 && rateNew >= 1.e-10)
    {
        T r = randn();
        tau = totalTime - log(r) / rateNew;
    }
    else if (rateOld >= 1.e-10 && rateNew >= 1.e-10)
    {
        tau = (rateOld / rateNew) * (tau - totalTime) + totalTime;
    }

    return tau;
}

// Returns the square of the distance between the centers of two platelets in lattice units
plint distanceBetween(platelet a, platelet b)
{
    plint px = a.center[0] - b.center[0];
    plint py = a.center[1] - b.center[1];
    plint pz = a.center[2] - b.center[2];
    return (px * px + py * py + pz * pz);
}

// Tells if a given lattice point lies inside the platelet or not
bool liesInside(Array<plint, 3> position, platelet plt)
{
    plint px, py, pz;
    px = position[0];
    py = position[1];
    pz = position[2];

    plint cx, cy, cz;
    cx = plt.center[0];
    cy = plt.center[1];
    cz = plt.center[2];

    T d = plt.dp;

    px = px - cx;
    py = py - cy;
    pz = pz - cz;

    return px * px + py * py + pz * pz <= diaFactor * diaFactor / 4;
}

// This function checks if there are overlaps between any two given platelets
bool checkOverlap(platelet a, platelet b)
{
    return distanceBetween(a, b) < diaFactor * diaFactor;
}

// This function checks if there are overlaps between two platelets
// The above implementation is used for the convective PFA
// The implementation below is used for checking available platelets for binding
// This difference is due to the difference in the PFA algorithm and the binding algorithm
bool checkOverlapBinding(platelet a, platelet b)
{
    return distanceBetween(a, b) <= diaFactor * diaFactor;
}

// This function takes as argument a struct platelet
// If there is any overlap with another platelet, then it returns true
bool checkOverlap(platelet plt)
{
    plint cx, cy, cz;
    cx = plt.center[0];
    cy = plt.center[1];
    cz = plt.center[2];

    T dia = plt.dp;
    plint n;
    n = ceil(dia / h);

    plint i, j, k;
    Array<plint, 3> pos;

    bool flag = false;

    for (i = cx - n; i <= cx + n; ++i)
    {
        for (j = cy - n; j <= cy + n; ++j)
        {
            for (k = cz - n; k <= cz + n; ++k)
            {
                if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz)
                {
                    pos[0] = i;
                    pos[1] = j;
                    pos[2] = k;

                    if (liesInside(pos, plt))
                    {
                        auto iter = occupation.find((long long int)ny * nz * i + (long long int)nz * j + (long long int)k);
                        if (iter != occupation.end())
                        {
                            if (checkOverlap(*(iter->second), plt))
                            {
                                flag = true;
                                break;
                            }
                        }
                    }
                }
            }
            if (flag)
                break;
        }
        if (flag)
            break;
    }

    return flag;
}

// This function checks if there are overlaps between a platelet and any other platelet
// If yes, it returns the head of a linked list containing their addresses
list<list<platelet>::iterator> checkOverlap(list<platelet>::iterator current)
{
    platelet plt = *current;
    plint cx, cy, cz;
    cx = plt.center[0];
    cy = plt.center[1];
    cz = plt.center[2];

    T dia = plt.dp;
    plint n;
    n = ceil(dia / h);

    plint i, j, k;
    Array<plint, 3> pos;

    list<list<platelet>::iterator> overlapLinks;

    for (i = cx - n; i <= cx + n; ++i)
    {
        for (j = cy - n; j <= cy + n; ++j)
        {
            for (k = cz - n; k <= cz + n; ++k)
            {
                if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz)
                {
                    pos[0] = i;
                    pos[1] = j;
                    pos[2] = k;

                    if (liesInside(pos, plt))
                    {
                        auto iter = occupation.find((long long int)ny * nz * i + (long long int)nz * j + (long long int)k);
                        if (iter != occupation.end())
                        {
                            if (iter->second != current)
                            {
                                if (checkOverlap(*(iter->second), plt))
                                {
                                    overlapLinks.push_back(iter->second);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return overlapLinks;
}

// This function checks if there are overlaps between a platelet and any other platelet
// If yes, it returns the head of a linked list containing their addresses
// The above implementation is used for the convective PFA
// The implementation below is used for checking available platelets for binding
// This difference is due to the difference in the PFA algorithm and the binding algorithm
list<list<platelet>::iterator> checkOverlapBinding(list<platelet>::iterator current)
{
    platelet plt = *current;
    plint cx, cy, cz;
    cx = plt.center[0];
    cy = plt.center[1];
    cz = plt.center[2];

    T dia = plt.dp;
    plint n;
    n = ceil(dia / h);

    plint i, j, k;
    Array<plint, 3> pos;

    list<list<platelet>::iterator> overlapLinks;

    for (i = cx - n; i <= cx + n; ++i)
    {
        for (j = cy - n; j <= cy + n; ++j)
        {
            for (k = cz - n; k <= cz + n; ++k)
            {
                if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz)
                {
                    pos[0] = i;
                    pos[1] = j;
                    pos[2] = k;

                    auto iter = occupation.find((long long int)ny * nz * i + (long long int)nz * j + (long long int)k);
                    if (iter != occupation.end())
                    {
                        if (iter->second != current)
                        {
                            if (checkOverlapBinding(*(iter->second), plt))
                            {
                                overlapLinks.push_back(iter->second);
                            }
                        }
                    }
                }
            }
        }
    }

    return overlapLinks;
}

// This function is used to assign the velocity given the position of the platelet center
inline void obtainVelocity(plint cx, plint cy, plint cz)
{
    T px, py, pz;
    T f = (T)factor;
    px = (T)cx / f;
    py = (T)cy / f;
    pz = (T)cz / f;

    //Array <T,3> pos (h*px, h*py, h*pz);
    Array<T, 3> pos(px, py, pz);

    //Array <T,3> realPos (pos + KMCLocation);

    std::vector<Array<T, 3>> LBPos;
    //LBPos.push_back (Array <T,3>((realPos - LBLocation)/dx));
    LBPos.push_back(pos);

    std::vector<Array<T, 3>> velocityList = velocitySingleProbes(*(LB.lattice), LBPos);

    velocity = velocityList[0] * dx / dt;
}

// This function is used to assign velocity given a position in space
inline void obtainVelocity(T x, T y, T z)
{
    Array<T, 3> pos(x, y, z);
    pos = (pos - LBLocation) / dx;
    std::vector<Array<T, 3>> LBPos;
    LBPos.push_back(pos);

    std::vector<Array<T, 3>> velocityList = velocitySingleProbes(*(LB.lattice), LBPos);

    velocity = velocityList[0] * dx / dt;
}

// Checks whether a platelet has left the domain
bool checkWithinDomain(plint cx, plint cy, plint cz)
{
    cx = cx / factor;
    cy = cy / factor;
    cz = cz / factor;

    if (flagMatrixInside->get(cx, cy, cz) == 0)
    {
        return false;
    }
    return true;
}

// Checks whether a platelet is at any domain wall
bool checkAtWall(plint cx, plint cy, plint cz)
{
    cx = cx / factor;
    cy = cy / factor;
    cz = cz / factor;

    if (flagMatrixWall->get(cx, cy, cz) == 1)
    {
        return true;
    }
    return false;
}

// Checks whether a platelet can interact with the reactive patch
// This function is subject to change with a change in the location of the patch
bool checkCollagen(plint cx, plint cy, plint cz)
{
    plint i, j, k;
    i = cx;
    j = cy;
    k = cz;

    if (cz < nz / 4 || cz > 3 * nz / 4 || cy > ny / 2)
    {
        return false;
    }

    if (!checkWithinDomain(i, j - 1, k) || !checkWithinDomain(i - 1, j, k) || !checkWithinDomain(i + 1, j, k))
    {
        return true;
    }

    return false;
}

// This function ensures that all the rates are defined such that
// a move where a platelet leaves the domain is not allowed
void checkWithinDomain(list<platelet>::iterator current)
{
    platelet plt = *current;

    plint i, j, k;
    i = plt.center[0];
    j = plt.center[1];
    k = plt.center[2];

    // Presence of walls
    // This is subject to a change in domain
    if (!checkWithinDomain(i + 1, j, k))
    {
        auto iT = eventList.find(event(current, 0, plt.tau[0]));
        if (iT != eventList.end())
        {
            eventList.erase(iT);
        }
        plt.motion[0] = 0.;
        plt.tau[0] = dblmax;
    }
    if (!checkWithinDomain(i - 1, j, k))
    {
        auto iT = eventList.find(event(current, 1, plt.tau[1]));
        if (iT != eventList.end())
        {
            eventList.erase(iT);
        }
        plt.motion[1] = 0.;
        plt.tau[1] = dblmax;
    }
    if (!checkWithinDomain(i, j + 1, k))
    {
        auto iT = eventList.find(event(current, 2, plt.tau[2]));
        if (iT != eventList.end())
        {
            eventList.erase(iT);
        }
        plt.motion[2] = 0.;
        plt.tau[2] = dblmax;
    }
    if (!checkWithinDomain(i, j - 1, k))
    {
        auto iT = eventList.find(event(current, 3, plt.tau[3]));
        if (iT != eventList.end())
        {
            eventList.erase(iT);
        }
        plt.motion[3] = 0.;
        plt.tau[3] = dblmax;
    }
    if (!checkWithinDomain(i, j, k + 1) && k < nz - 1 - factor * margin)
    {
        auto iT = eventList.find(event(current, 4, plt.tau[4]));
        if (iT != eventList.end())
        {
            eventList.erase(iT);
        }
        plt.motion[4] = 0.;
        plt.tau[4] = dblmax;
    }
    if (!checkWithinDomain(i, j, k - 1))
    {
        auto iT = eventList.find(event(current, 5, plt.tau[5]));
        if (iT != eventList.end())
        {
            eventList.erase(iT);
        }
        plt.motion[5] = 0.;
        plt.tau[5] = dblmax;
    }

    *current = plt;
}

// This function implements the pass forward algorithm (Flamm et al. J Chem Phys 2009, 2011)
// Resets convective rates for a given direction if motion is blocked by another platelet
void passForward(list<platelet>::iterator current, plint dir)
{
    list<list<platelet>::iterator> overlapLinks;

    platelet currentPlt;
    currentPlt = *current;

    // Move the platelet in the given direction to see if there's overlap
    currentPlt.center[dir / 2] = currentPlt.center[dir / 2] + (plint)pow(-1, dir);

    *current = currentPlt;

    overlapLinks = checkOverlap(current);

    currentPlt.center[dir / 2] = currentPlt.center[dir / 2] - (plint)pow(-1, dir);

    *current = currentPlt;

    if (overlapLinks.empty())
    {
        currentPlt.blockCount[dir] = 0;

        if (!currentPlt.bound)
        {
            bool flag = false;
            T oldRate = currentPlt.motion[dir];

            if (oldRate > 1.e-10)
            {
                flag = true;
            }

            currentPlt.motion[dir] = diffusion + currentPlt.convection[dir];

            if (oldRate != currentPlt.motion[dir])
            {
                if (!flag)
                {
                    eventList.erase(event(current, dir, currentPlt.tau[dir]));
                    currentPlt.tau[dir] = dblmax;
                }
                if (currentPlt.motion[dir] > 1.e-10)
                {
                    T r = randn();
                    T dtau = totalTime - log(r) / currentPlt.motion[dir];
                    eventList.insert(event(current, dir, dtau));
                    currentPlt.tau[dir] = dtau;
                }
            }

            *current = currentPlt;
            checkWithinDomain(current);
        }
        else
        {
            *current = currentPlt;
        }

        return;
    }

    list<list<platelet>::iterator>::iterator tempHead;
    list<platelet>::iterator temp;

    platelet tempPlt;

    plint count = 0;
    for (tempHead = overlapLinks.begin(); tempHead != overlapLinks.end(); ++tempHead)
    {
        count++;
    }

    T motionTransfer = currentPlt.convection[dir] / (T)count;

    plint cx, cy, cz;

    for (tempHead = overlapLinks.begin(); tempHead != overlapLinks.end(); ++tempHead)
    {
        temp = *tempHead;
        tempPlt = *temp;

        cx = tempPlt.center[0];
        cy = tempPlt.center[1];
        cz = tempPlt.center[2];

        obtainVelocity(cx, cy, cz); // Get velocity from LB

        tempPlt.convection[dir] = max(0., pow(-1, dir) * velocity[dir / 2] / h) + motionTransfer;
        *temp = tempPlt;

        if (!tempPlt.bound)
        {
            passForward(temp, dir);
        }
    }

    currentPlt.motion[dir] = 0.;
    currentPlt.blockCount[dir] = count;
    eventList.erase(event(current, dir, currentPlt.tau[dir]));
    currentPlt.tau[dir] = dblmax;

    *current = currentPlt;

    return;
}

// This function finds if any platelet was blocked from moving by the platelet which just moved.
// If there exists such a platelet, then it resets the motion of the platelet from 0 to the apt value
// Furthermore, due to the motion of platelets, there may be the possibility that this platelet gets blocked
// It implements the pass forward algorithm in these cases.
void passForwardMotion(list<platelet>::iterator current, plint dir)
{
    list<platelet>::iterator temp;
    platelet currentPlt, tempPlt;
    currentPlt = *current;
    plint i, j;

    list<list<platelet>::iterator> overlapLinks;
    list<list<platelet>::iterator>::iterator tempHead;

    *current = currentPlt;

    // Check if the platelet was blocking any platelet
    for (i = 0; i < 6; ++i)
    {
        if (i == dir)
            continue;

        // Move the platelet backwards away from the direction of motion
        currentPlt.center[dir / 2] = currentPlt.center[dir / 2] - (plint)pow(-1, dir);

        currentPlt.center[i / 2] = currentPlt.center[i / 2] + (plint)pow(-1, i);
        *current = currentPlt;

        overlapLinks = checkOverlap(current);

        currentPlt.center[i / 2] = currentPlt.center[i / 2] - (plint)pow(-1, i);
        currentPlt.center[dir / 2] = currentPlt.center[dir / 2] + (plint)pow(-1, dir);

        *current = currentPlt;

        if (!overlapLinks.empty())
        {
            if (i % 2 == 1)
            {
                j = i - 1;
            }
            else
            {
                j = i + 1;
            }

            for (tempHead = overlapLinks.begin(); tempHead != overlapLinks.end(); ++tempHead)
            {
                temp = *tempHead;
                tempPlt = *temp;

                tempPlt.blockCount[j] -= 1;

                *temp = tempPlt;

                passForward(temp, j);

                tempPlt = *temp;

                if (!tempPlt.bound)
                {
                    bool flag = false;
                    T oldRate = tempPlt.motion[i];

                    if (oldRate == 0.)
                    {
                        flag = true;
                    }

                    obtainVelocity(tempPlt.center[0], tempPlt.center[1], tempPlt.center[2]);
                    tempPlt.convection[i] = max(0., pow(-1, i) * velocity[i / 2] / h);
                    tempPlt.motion[i] = diffusion + tempPlt.convection[i];

                    if (oldRate != tempPlt.motion[i])
                    {
                        if (!flag)
                        {
                            eventList.erase(event(temp, i, tempPlt.tau[i]));
                            tempPlt.tau[i] = dblmax;
                        }

                        if (tempPlt.motion[i] != 0.)
                        {
                            T r = randn();
                            T dtau = totalTime - log(r) / tempPlt.motion[i];
                            eventList.insert(event(temp, i, dtau));
                            tempPlt.tau[i] = dtau;
                        }
                    }

                    *temp = tempPlt;
                }

                passForward(temp, i);
            }

            overlapLinks.clear();
        }
    }

    // Check if the platelet that moved is now blocked in any direction.
    // If it is, implement the pass forward algorithm at these places.

    plint k;
    if (dir % 2 == 1)
    {
        k = dir - 1;
    }
    else
    {
        k = dir + 1;
    }

    for (i = 0; i < 6; ++i)
    {
        if (i == k)
            continue;

        currentPlt.center[i / 2] = currentPlt.center[i / 2] + (plint)pow(-1, i);
        *current = currentPlt;

        overlapLinks = checkOverlap(current);

        currentPlt.center[i / 2] = currentPlt.center[i / 2] - (plint)pow(-1, i);
        *current = currentPlt;

        if (!overlapLinks.empty())
        {
            if (i % 2 == 1)
            {
                j = i - 1;
            }
            else
            {
                j = i + 1;
            }

            for (tempHead = overlapLinks.begin(); tempHead != overlapLinks.end(); tempHead++)
            {
                temp = *tempHead;
                passForward(temp, j);

                passForward(current, i);
            }

            overlapLinks.clear();
        }
    }

    return;
}

// This function implements the simple blocking algorithm SBA(Flamm et al. J Chem Phys 2009)
// Sets convective rate to zero if motion is blocked by another platelet in a givne direction
// More computationally efficient than the pass forward algorithm
// Inaccurate physical representation due to blocking is insignificant because platelet density is low
void simpleBlocking(list<platelet>::iterator &current, plint dir)
{
    global::timer("sba").restart();

    platelet plt = *current;

    plt.center[dir / 2] = plt.center[dir / 2] + (plint)pow(-1, dir);
    *current = plt;

    auto overlaplinks = checkOverlap(current);

    plt.center[dir / 2] = plt.center[dir / 2] - (plint)pow(-1, dir);

    if (!overlaplinks.empty())
    {
        plt.motion[dir] = 0.;
    }

    *current = plt;

    pfatime = pfatime + global::timer("sba").getTime();
}

// The concentration probability density function must be normalized
// This function returns the integral of the PDF so it can be used as
// denominator for normalization
void normalizeConcentration()
{
    T delta = 1. / (T)(pdf.size() - 1);
    T x, y, r;
    plint i;
    T sum = 0.;
    for (x = 0.; x <= 1.; x = x + delta)
    {
        for (y = 0.; y <= 1.; y = y + delta)
        {
            if (x * x + y * y >= 1.)
            {
                continue;
            }
            i = (plint)(sqrt(x * x + y * y) / delta);
            sum = sum + pdf[i] * delta * delta;
        }
    }

    normalization = sum;
}

// Computes the effective net flow rate of platelets into the domain
T computeFlowRate()
{
    // Based on inlet flow rate
    // Compute vol. flow rate using Palabos, multiply by conc. of platelets in blood

    plint n = pdf.size();
    T delta = 1. / (T)(n - 1);
    T r, x, y;
    T flowRate = 0.;
    plint i;
    T concPlt = 1.5e14;

    i = 0;
    T ds = concPlt * radiusReal * radiusReal * delta * delta / normalization;
    // Area element * normalized concentration

    inletEventList.clear();

    // Temporary code to compute volumetrix flow rate
    T volFlowRate = 0.;

    for (x = -1.; x <= 1.; x = x + delta)
    {
        for (y = -1.; y <= 1.; y = y + delta)
        {
            if (x * x + y * y >= 1.)
            {
                continue;
            }
            i = (plint)(sqrt(x * x + y * y) / delta);
            Array<T, 3> realPos(x * radiusReal, y * radiusReal, 0.);
            std::vector<Array<T, 3>> LBPos;
            LBPos.push_back(Array<T, 3>((realPos - LBLocation) / dx));

            std::vector<Array<T, 3>> velocityList = velocitySingleProbes(*(LB.lattice), LBPos);

            Array<T, 3> vel = velocityList[0] * dx / dt;
            T rate = vel[2] * pdf[i] * ds; // Imposing a platelet conc. profile

            T dtau = -log(randn_inlet()) / rate;

            inletEventList.insert(inletEvent(x, y, dtau));

            flowRate = flowRate + rate;

            // Temporary code to compute volumetrix flow rate
            volFlowRate = volFlowRate + vel[2] * radiusReal * radiusReal * delta * delta;
        }
    }

    // Write volumetric flow rate versus time onto file
    plb_ofstream file;
    std::string fname = "flowrate.csv";
    char filename[fname.size() + 1];
    fname.copy(filename, fname.size() + 1);
    filename[fname.size()] = '\0';
    file.open(filename, ios::app);
    file << totalTime << "," << volFlowRate << endl;
    file.close();

    return flowRate;
}

// Computes the inlet flow rate at given radial coordinate r
void computeFlowRate(T x, T y)
{
    plint n = pdf.size();
    T delta = 1. / (T)(n - 1);
    T concPlt = 1.5e14;

    T ds = concPlt * radiusReal * radiusReal * delta * delta / normalization;
    // Area element * normalized concentration

    T r = sqrt(x * x + y * y);

    plint i = (plint)(r / delta);

    Array<T, 3> realPos(x * radiusReal, y * radiusReal, 0.);
    std::vector<Array<T, 3>> LBPos;
    LBPos.push_back(Array<T, 3>((realPos - LBLocation) / dx));

    std::vector<Array<T, 3>> velocityList = velocitySingleProbes(*(LB.lattice), LBPos);

    Array<T, 3> vel = velocityList[0] * dx / dt;
    T rate = vel[2] * pdf[i] * ds; // Imposing a platelet conc. profile

    T dtau = -log(randn_inlet()) / rate;

    inletEventList.erase(inletEventList.begin());

    inletEventList.insert(inletEvent(x, y, dtau));
}

// Binary search function to estimate the index of radPos
plint binSearch(plint first, plint last, T val)
{
    if (last >= first)
    {
        plint mid = (first + last) / 2;

        if (mid > 0)
        {
            if (cdf[mid - 1] <= val && cdf[mid] >= val)
                return mid;
            else if (cdf[mid - 1] > val && cdf[mid] > val)
                return binSearch(first, mid - 1, val);
            else
                return binSearch(mid + 1, last, val);
        }
        else
        {
            if (cdf[mid] >= val)
                return mid;
            else if (cdf[mid] < val && cdf[mid + 1] >= val)
                return mid + 1;
        }
    }
    pcout << "Binary search for inlet position failed" << endl;
    exit(-1);
    return -1;
}

// This function adds a platelet randomly in the domain
// Used to initialize KMC
// To add platelets at inlet, refer to function: addPlateletInlet instead
void addPlatelet()
{
    platelet plt;

    // Impose inlet platelet conc profile with skew toward the walls
    T val = randn_inlet();
    plint ind = binSearch(0, cdf.size() - 1, val);
    T r = radPos[ind] * radiusReal;

    T x = randn_inlet() - 0.5;
    T y = randn_inlet() - 0.5;
    T dist = sqrt(x * x + y * y);
    x = x * r / dist;
    y = y * r / dist;

    plt.center[0] = (plint)((x + inletRealPos[0] - KMCLocation[0]) / h);
    plt.center[1] = (plint)((y + inletRealPos[1] - KMCLocation[1]) / h);
    plt.center[2] = margin + (plint)(randn() * (nz - margin - 1));

    plt.dp = pltDia;

    if (plt.center[0] <= 0 || plt.center[0] >= nx - 1 || plt.center[1] <= 0 || plt.center[1] >= ny - 1 || plt.center[2] <= 0 || plt.center[2] >= nz - 1)
    {
        addPlatelet();
        return;
    }

    if (checkOverlap(plt) || !checkWithinDomain(plt.center[0], plt.center[1], plt.center[2]))
    {
        addPlatelet();
        return;
    }

    initializePlatelet(plt);

    return;
}

// This function is used to add a new platelet to the domain
// Add a biasing function so platelets enter the domain near the wall later on
void addPlateletInlet()
{
    platelet plt;

    inletEvent inEvent = *(inletEventList.begin());
    T x = inEvent.x;
    T y = inEvent.y;

    computeFlowRate(x, y);

    x = x * radiusReal;
    y = y * radiusReal;

    plt.center[0] = (plint)((x + inletRealPos[0] - LBLocation[0]) / h);
    plt.center[1] = (plint)((y + inletRealPos[1] - LBLocation[1]) / h);
    plt.center[2] = nz / 8; // For now, it is assumed that the inlet plane is always normal to the z-axis

    plt.dp = pltDia;

    if (plt.center[0] <= 0 || plt.center[0] >= nx - 1 || plt.center[1] <= 0 || plt.center[1] >= ny - 1 || plt.center[2] <= 0 || plt.center[2] >= nz - 1)
    {
        addPlateletInlet();
        return;
    }

    if (checkOverlap(plt) || !checkWithinDomain(plt.center[0], plt.center[1], plt.center[2]))
    {
        addPlateletInlet();
    }
    else
    {

        nPlatelets++;
        initializePlatelet(plt);
        return;
    }
}

// Computes the enhancement in binding rates due to vwf stretching
T vwfFactor(list<platelet>::iterator current)
{
    T enhancement;

    platelet plt = *current;
    plt.localShear = computePltShear(current);
    if (plt.localShear < 3000.)
    {
        enhancement = 1.;
    }
    else if (plt.localShear >= 3000. && plt.localShear < 8000.)
    {
        enhancement = 1. + 19. * (plt.localShear - 3000.) / (5000.);
    }
    else
    {
        enhancement = 20.;
    }
    *current = plt;
    return enhancement;
}

// Computes the enhancement in detachment rate due to shear
T detachFactor(list<platelet>::iterator current)
{
    T enhancement;
    platelet plt = *current;
    plt.localShear = computePltShear(current);
    //pcout << "Local shear rate = " << plt.localShear << endl;

    if (plt.localShear <= 1000.)
    {
        enhancement = exp(plt.localShear / charShear);
    }
    else
    {
        enhancement = exp(5.) * exp((plt.localShear - 1000.) / 50000.);
    }

    *current = plt;
    return enhancement;
}

// This functional is used to set the binding rates associated with
// the given platelet as the argument
void setBindingRates(list<platelet>::iterator current)
{
    global::timer("binding").restart();

    platelet plt = *current;
    list<list<platelet>::iterator> overlapLinks;
    list<list<platelet>::iterator>::iterator tempHead;

    list<platelet>::iterator temp;
    platelet tempPlt;

    T rate = 0.;

    T enhancement = vwfFactor(current);

    // Presence of the reactive patch
    // This is subject to change with a change of domain

    if (checkCollagen(plt.center[0], plt.center[1], plt.center[2]))
    {
        rate = rate + k_att_col * plt.activation * plt.activationx;
    }

    // Move the platelet in any given direction to see if there's overlap with another platelet
    // If there is an overlap and the overlapping plt is connected to coll., then a binding event is possible

    overlapLinks = checkOverlapBinding(current);

    if (!overlapLinks.empty())
    {
        for (tempHead = overlapLinks.begin(); tempHead != overlapLinks.end(); tempHead++)
        {
            temp = *tempHead;
            tempPlt = *temp;

            T tempEnhancement = vwfFactor(temp);

            if (tempPlt.bound)
            {
                rate = rate + 0.5 * k_att_fib * sqrt(plt.activation * tempPlt.activation * plt.activationx * tempPlt.activationx) * sqrt(enhancement) * sqrt(tempEnhancement);
                // Factor of 0.5 to account for double counting
            }
        }
    }

    T dtau = dblmax;
    if (rate > 1.e-10)
    {
        T r = randn();
        dtau = totalTime - log(r) / rate;
        eventList.insert(event(current, 6, dtau));
    }

    overlapLinks.clear();

    plt.binding = rate;

    auto iT = eventList.find(event(current, 6, plt.tau[6]));
    if (iT != eventList.end())
    {
        eventList.erase(iT);
    }
    plt.tau[6] = dtau;
    *current = plt;

    bindingratesettime = bindingratesettime + global::timer("binding").getTime();
}

// This functional is used to set the unbinding rates associated with
// the given platelet as the argument
void setUnbindingRates(list<platelet>::iterator current)
{
    global::timer("binding").restart();

    platelet plt = *current;

    auto iT = eventList.find(event(current, 7, plt.tau[7]));
    if (iT != eventList.end())
    {
        eventList.erase(iT);
    }

    list<list<platelet>::iterator> overlapLinks;
    list<list<platelet>::iterator>::iterator tempHead;

    list<platelet>::iterator temp;
    platelet tempPlt;

    T rate = dblmax;
    //T rate = 0.;
    T tempRate = 0.;

    plint ind, dir, j, type;

    // Presence of the reactive patch
    // This is subject to change with a change of domain

    T enhancement = detachFactor(current) / vwfFactor(current);

    if (checkCollagen(plt.center[0], plt.center[1], plt.center[2]))
    {
        tempRate = k_det_col * enhancement / (plt.activation * plt.activationx);

        if (rate > tempRate)
        {
            rate = tempRate;
        }

        //rate += tempRate;
    }

    overlapLinks = checkOverlapBinding(current);

    if (!overlapLinks.empty())
    {
        // Check for neighboring bound platelets
        for (tempHead = overlapLinks.begin(); tempHead != overlapLinks.end(); tempHead++)
        {
            temp = *tempHead;
            if (temp == plateletList.end())
            {
                continue;
            }
            tempPlt = *temp;

            T tempEnhancement = detachFactor(temp) / vwfFactor(temp);

            tempRate = 0.5 * k_det_fib / (sqrt(plt.activation * tempPlt.activation * plt.activationx * tempPlt.activationx));

            // Factor of 0.5 to account for double counting

            tempRate = tempRate * sqrt(enhancement) * sqrt(tempEnhancement);

            if (rate > tempRate)
            {
                rate = tempRate;
            }

            //rate += tempRate;

            *temp = tempPlt;
        }
    }

    T oldRate = plt.unbinding;
    T dtau = updateEventTime(plt.tau[7], oldRate, rate);

    eventList.insert(event(current, 7, dtau));

    plt.unbinding = rate;
    plt.tau[7] = dtau;
    *current = plt;

    bindingratesettime = bindingratesettime + global::timer("binding").getTime();
}

plint nout = 0;
// This function sets the rate of each event for a given platelet that moves
// More efficient to use this when the velocity field does not change
void setRates(list<platelet>::iterator current, plint type)
{
    platelet plt = *current;

    list<platelet>::iterator temp;
    platelet tempPlt;
    plint ind, j;

    // If a motion event is executed, then apply the pass-forward algorithm accordingly
    // Check for potential binding events and assign their rates

    if (type < 6)
    {
        obtainVelocity(plt.center[0], plt.center[1], plt.center[2]); // Get velocity from LB

        for (j = 0; j < 6; ++j)
        {
            plt.convection[j] = max(0., pow(-1, j) * velocity[j / 2]) / h;
            plt.motion[j] = diffusion + plt.convection[j];

            auto iT = eventList.find(event(current, j, plt.tau[j]));
            if (iT != eventList.end())
            {
                eventList.erase(iT);
            }
            plt.tau[j] = dblmax;
            *current = plt;
            //simpleBlocking(current, j);
            plt = *current;

            if (plt.motion[j] > 1.e-10)
            {
                T r = randn();
                T dtau = totalTime - log(r) / plt.motion[j];
                eventList.insert(event(current, j, dtau));
                plt.tau[j] = dtau;
            }
        }

        *current = plt;

        checkWithinDomain(current);

        setBindingRates(current);

        passForwardMotion(current, type);
    }

    // If a binding event is executed, then set the binding rate to zero, because it cannot happen again
    // Set the unbinding rate accordingly
    // motion needs to be set to zero

    if (type == 6)
    {
        plt.binding = 0.;
        auto iT = eventList.find(event(current, type, plt.tau[type]));
        if (iT != eventList.end())
        {
            eventList.erase(iT);
        }
        plt.tau[type] = dblmax;

        for (j = 0; j < 6; ++j)
        {
            plt.motion[j] = 0.;
            eventList.erase(event(current, j, plt.tau[j]));
            plt.tau[j] = dblmax;
        }

        plt.bound = true;
        plt.boundTime = totalTime;

        //setToConstant(*boolMask, boundingBox(plt), 1);
        updateBoolMask(plt, 1);
        boundList.push_back(current);

        *current = plt;
        setUnbindingRates(current);
    }

    // If an unbinding event is executed, set the unbinding rate to zero
    // because the bond cannot be broken again
    // Set binding rates accordingly
    // If the platelet becomes free to move, set motion rates.
    if (type == 7)
    {
        ++nUnbindingOccurred;

        plt.unbinding = 0.;
        auto iT = eventList.find(event(current, type, plt.tau[type]));
        if (iT != eventList.end())
        {
            eventList.erase(iT);
        }
        plt.tau[type] = dblmax;
        plt.bound = false;
        plt.boundTime = dblmax;

        //setToConstant(*boolMask, boundingBox(plt), 0);
        updateBoolMask(plt, 0);

        obtainVelocity(plt.center[0], plt.center[1], plt.center[2]); // Get velocity from LB

        for (j = 0; j < 6; ++j)
        {
            /*
            plt.convection[j] = max(0., pow(-1,j)*velocity[j/2])/h;
            plt.motion[j] = diffusion + plt.convection[j];

            plt.tau[j] = dblmax;
            *current = plt;
            simpleBlocking(current, j);
            plt = *current;

            if (plt.motion[j] > 1.e-10)
            {
                T r = randn();
                T dtau = totalTime - log(r)/plt.motion[j];
                eventList.insert(event(current, j, dtau));
                plt.tau[j] = dtau;
            }
            */
            passForward(current, j);
        }

        *current = plt;

        checkWithinDomain(current);

        setBindingRates(current);
    }

    // Platelet being added to the domain
    if (type >= 8)
    {

        eventList.erase(event(inletIter, type, inletTau));

        addPlateletInlet();

        current = plateletList.begin();
        plt = *current;
        if (!plt.bound)
        {
            obtainVelocity(plt.center[0], plt.center[1], plt.center[2]); // Get velocity from LB

            for (j = 0; j < 6; ++j)
            {
                plt.convection[j] = max(0., pow(-1, j) * velocity[j / 2]) / h;
                plt.motion[j] = diffusion + plt.convection[j];
                plt.tau[j] = dblmax;
                *current = plt;
                //simpleBlocking(current, j);
                plt = *current;

                if (plt.motion[j] > 1.e-10)
                {
                    T r = randn();
                    T dtau = totalTime - log(r) / plt.motion[j];
                    eventList.insert(event(current, j, dtau));
                    plt.tau[j] = dtau;
                }
            }

            *current = plt;

            checkWithinDomain(current);

            passForwardMotion(current, 4);
            // SInce platelet enters domain through motion in the z-direction
            // passforwardmotion needs to be applied only in the +z direction

            setBindingRates(current);
        }
        // Inlet rate putative time reset
        T r = randn();
        T dtau = totalTime - log(r) / inletRate;
        eventList.insert(event(inletIter, 8, dtau));
        inletTau = dtau;
    }

    // Remove platelet from list if it has left the domain
    plt = *current;
    if (!checkWithinDomain(plt.center[0], plt.center[1], plt.center[2]) || plt.center[2] >= 7 * nz / 8)
    {
        plint count = 0;
        for (j = 0; j < 6; ++j)
        {
            auto iT = eventList.find(event(current, j, plt.tau[j]));
            //eventList.erase(event(current, j, plt.tau[j]));
            if (iT != eventList.end())
            {
                eventList.erase(iT);
                count++;
            }
        }

        occupation.erase((long long int)ny * nz * plt.center[0] +
                         (long long int)nz * plt.center[1] + (long long int)plt.center[2]);

        if (current != plateletList.end())
        {
            plateletList.erase(current);
        }
    }
}

// This function sets the rate of each event for each platelet
// Alters the rate parameters where the move is not allowed
// by calling the convective pass-forward algorithm routine
void setRates()
{
    list<platelet>::iterator current;

    platelet plt;

    plint ind, j;

    nBoundPlatelets = 0;
    nPlatelets = 0;

    for (current = plateletList.begin(); current != plateletList.end(); ++current)
    {
        plt = *current;

        if (!plt.bound)
        {
            obtainVelocity(plt.center[0], plt.center[1], plt.center[2]); // Get velocity from LB

            for (j = 0; j < 6; ++j)
            {
                plt.convection[j] = max(0., pow(-1, j) * velocity[j / 2]) / h;
                T oldRate = plt.motion[j];
                plt.motion[j] = diffusion + plt.convection[j];

                *current = plt;
                //simpleBlocking(current, j);
                plt = *current;

                T dtau = updateEventTime(plt.tau[j], oldRate, plt.motion[j]);

                auto iT = eventList.find(event(current, j, plt.tau[j]));
                if (iT != eventList.end())
                {
                    eventList.erase(iT);
                }
                plt.tau[j] = dtau;
                eventList.insert(event(current, j, dtau));
            }
        }
        else
        {
            //plt.localShear = computePltShear(current);
            nBoundPlatelets = nBoundPlatelets + 1;
        }
        nPlatelets = nPlatelets + 1;
        *current = plt;
    }

    // Pass forward algorithm
    for (current = plateletList.begin(); current != plateletList.end(); ++current)
    {
        for (j = 0; j < 6; ++j)
        {
            passForward(current, j);
        }
    }

    // Set binding and unbinding rates based on updated platelet activations and shear rates
    for (current = plateletList.begin(); current != plateletList.end(); ++current)
    {
        plt = *current;

        if (!plt.bound)
        {
            setBindingRates(current);
        }
        else
        {
            setUnbindingRates(current);
        }
    }

    for (current = plateletList.begin(); current != plateletList.end(); ++current)
    {
        checkWithinDomain(current);
    }

    current = plateletList.begin();

    // Inlet rate
    eventList.erase(event(inletIter, 8, inletTau));
    T dtau = updateEventTime(inletTau, oldInletRate, inletRate);
    inletTau = dtau;
    eventList.insert(event(inletIter, 8, dtau));
}

void iniKMCLattice()
{
    for (plint i = 0; i < nPlatelets; ++i)
    {
        addPlatelet();
    }

    platelet plt;
    plt.center[0] = 0;
    plt.center[1] = 0;
    plt.center[2] = 0;

    plt.dp = pltDia;

    for (plint i = 0; i < 6; ++i)
    {
        plt.convection[i] = 0.;
        plt.motion[i] = 0.;
    }
    plt.binding = 0.;
    plt.unbinding = 0.;

    for (plint i = 0; i < 8; ++i)
    {
        plt.tau[i] = dblmax;
    }
    plt.bound = false;
    plt.inCount = -1;

    dummyPlateletList.push_front(plt);

    inletIter = dummyPlateletList.begin();
}

void runKMC()
{
    list<platelet>::iterator current;
    plint j;

    auto executedEvent = eventList.begin();
    current = executedEvent->address;
    j = executedEvent->type;
    totalTime = executedEvent->tau;

    if (current == plateletList.end())
    {
        eventList.erase(eventList.begin());
        return;
    }

    if (j < 8 && current->tau[j] != totalTime)
    {
        eventList.erase(executedEvent);
        return;
    }

    platelet plt = *current;

    if (j < 6) // Motion event
    {
        occupation.erase((long long int)ny * nz * plt.center[0] + (long long int)nz * plt.center[1] + (long long int)plt.center[2]);

        plt.center[j / 2] = plt.center[j / 2] + (plint)pow(-1, j);

        std::pair<long long int, std::list<platelet>::iterator> pltPair(
            (long long int)ny * nz * plt.center[0] + (long long int)nz * plt.center[1] + (long long int)plt.center[2],
            current);
        occupation.insert(pltPair);
    }

    *current = plt;

    eventList.erase(executedEvent);

    setRates(current, j); // Update rate database
}

/******************************* LB routines ***********************************/

/// A functional used to assign Poiseuille velocity profile at inlet
T poiseuilleVelocity(plint iX, plint iY, T uLB)
{
    T x = ((T)iX * dx + LBLocation[0]);
    T y = ((T)iY * dx + LBLocation[1]);
    T r2 = (x * x + y * y) / (radiusReal * radiusReal);
    T u = 2. * uLB * (1 - r2);
    return u;
}

// A function that returns whether a LB lattice point lies inside a given platelet or not
// This is used to instantiate bounceback nodes in the interior nodes of the platelet by the
// domain processing functional below
bool liesInside(Array<plint, 3> pos, Array<plint, 3> center, T dia)
{
    Array<T, 3> centerT((T)center[0], (T)center[1], (T)center[2]);
    Array<T, 3> realCenter(dx * centerT + LBLocation);
    Array<T, 3> posT((T)pos[0], (T)pos[1], (T)pos[2]);
    Array<T, 3> realPos(dx * posT + LBLocation);
    return pow(realPos[0] - realCenter[0], 2.) + pow(realPos[1] - realCenter[1], 2.) +
               pow(realPos[2] - realCenter[2], 2.) <=
           0.25 * dia * dia;
}

// Member function definitions of plateletShapeDomain:

// The function-call operator is overridden to specify the location
// of bounce-back nodes.
template <typename T>
bool plateletShapeDomain<T>::operator()(plint iX, plint iY, plint iZ) const
{
    Array<plint, 3> pos(iX, iY, iZ);
    return liesInside(pos, center, dia);
}

template <typename T>
plateletShapeDomain<T> *plateletShapeDomain<T>::clone() const
{
    return new plateletShapeDomain<T>(*this);
}

// This function obtains the largest cuboidal bounding box that surrounds a platelet
Box3D boundingBox(platelet &plt)
{
    plint px, p_x, py, p_y, pz, p_z;
    Array<plint, 3> center(plt.center[0] / factor,
                           plt.center[1] / factor, plt.center[2] / factor);

    px = center[0];
    py = center[1];
    pz = center[2];
    p_x = center[0];
    p_y = center[1];
    p_z = center[2];

    Box3D bBox(p_x, px, p_y, py, p_z, pz);

    return bBox;
}

void updateBoolMask(platelet &plt, int val)
{
    Array<plint, 3> center(plt.center[0] / factor,
                           plt.center[1] / factor, plt.center[2] / factor);
    plint pltRad = util::roundToInt(0.5 * plt.dp / dx);
    plint px, py, pz;
    for (plint i = -pltRad; i <= pltRad; ++i)
    {
        for (plint j = -pltRad; j <= pltRad; ++j)
        {
            for (plint k = -pltRad; k <= pltRad; ++k)
            {
                px = center[0] + i;
                py = center[1] + j;
                pz = center[2] + k;
                if (px < 0 || px >= nxlb || py < 0 || py >= nylb || pz < 0 || pz >= nzlb)
                {
                    continue;
                }

                if (i * i + j * j + k * k <= pltRad * pltRad)
                {
                    Box3D boundingBox(px, px, py, py, pz, pz);
                    setToConstant(*boolMask, boundingBox, val);
                }
            }
        }
    }
}

// This function is used to compute the local shear around a platelet
T computePltShear(list<platelet>::iterator current)
{
    platelet plt = *current;

    T netShear = 0.;
    plint px, py, pz;
    px = plt.center[0] / factor;
    py = plt.center[1] / factor;
    pz = plt.center[2] / factor;
    plint count = 0;
    T maxShear = 0.;
    for (plint i = px - 1; i <= px + 1; ++i)
    {
        for (plint j = py - 1; j <= py + 1; ++j)
        {
            for (plint k = pz - 1; k <= pz + 1; ++k)
            {
                if (i < 0 || i >= nxlb || j < 0 || j >= nylb || k < 0 || k >= nzlb)
                {
                    continue;
                }
                T tempShear = LB.shearRate->get(i, j, k);
                if (isnan(tempShear))
                {
                    continue;
                }
                if (tempShear > maxShear)
                {
                    maxShear = tempShear;
                }
                netShear = netShear + tempShear;
                count = count + 1;
            }
        }
    }

    return (sqrt(2.) * maxShear / dt);
}

/// Write the full velocity and the velocity-norm into a VTK file.
void writeVTK(MultiBlockLattice3D<T, DESCRIPTOR> &lattice,
              T dx, T dt, plint iter)
{
    auto domain = lattice.getBoundingBox();
    auto vel = *computeVelocity(lattice, domain);
    ParallelVtkImageOutput3D<T> vtkOut(createFileName("vtk", iter, 6), 3, dx);
    vtkOut.writeData<T>(*computeVelocityNorm(lattice, domain), "velocityNorm", dx / dt);
    vtkOut.writeData<3, T>(vel, "velocity", dx / dt);
    vtkOut.writeData<T>(*computeSymmetricTensorNorm(*computeStrainRateFromStress(lattice, domain)), "shear", 1 / dt);
}

/// This is the function that prepares the actual LB simulation.
void updateLB(MultiBlockLattice3D<T, DESCRIPTOR> &lattice)
{
    global::timer("bounceback").restart();

    // Add bounceback nodes at places where there is platlet mass, this is obtained from LKMC
    // No slip BCs at platelet boundaries
    // addBounceBack (lattice);
    defineDynamics(lattice, *boolMask, lattice.getBoundingBox(), new BounceBack<T, DESCRIPTOR>, 1);
    boundList.clear();

    bouncebacksettime = bouncebacksettime + global::timer("bounceback").getTime();

    lattice.initialize();

    // Initialize density and velocity based on values before platelet deposition/detachment

    if (iter > 0)
        loadBinaryBlock(lattice, "checkpoint.dat");

    global::timer("lbconvergence").restart();
    // Convergence bookkeeping
    util::ValueTracer<T> velocityTracer(1., resolution, epsilon);

    // Collision and streaming iterations.
    plint j = 0;
    while (!velocityTracer.hasConverged())
    {

        lattice.collideAndStream();
        if (j % 10 == 0)
            velocityTracer.takeValue(computeAverageEnergy(lattice));
        ++j;
    }

    lbconvergencetime = lbconvergencetime + global::timer("lbconvergence").getTime();
}

/// This is the function that prepares the actual LB simulation.
void runLB(plint level)
{
    T nuLB_ = dt * kinematicViscosity / (dx * dx);
    T uAveLB = averageInletVelocity * dt / dx;
    T omega = 1. / (3. * nuLB_ + 0.5);

    if (iter == 0)
    {
        boolMask.reset(new MultiScalarField3D<int>((MultiBlock3D &)LB.voxelizedDomain->getVoxelMatrix()));
        setToConstant(*boolMask, boolMask->getBoundingBox(), 0);
    }

    Dynamics<T, DESCRIPTOR> *dynamics = 0;
    if (useIncompressible)
    {
        dynamics = new IncBGKdynamics<T, DESCRIPTOR>(omega); // In this model velocity equals momentum.
    }
    else
    {
        dynamics = new BGKdynamics<T, DESCRIPTOR>(omega); // In this model velocity equals momentum
                                                          //   divided by density.
    }

    std::auto_ptr<MultiBlockLattice3D<T, DESCRIPTOR>> lattice =
        generateMultiBlockLattice<T, DESCRIPTOR>(
            LB.voxelizedDomain->getVoxelMatrix(), envelopeWidth, dynamics);
    lattice->toggleInternalStatistics(false);

    OnLatticeBoundaryCondition3D<T, DESCRIPTOR> *boundaryCondition = createLocalBoundaryCondition3D<T, DESCRIPTOR>();

    Array<T, 3> inletPos((inletRealPos - LBLocation) / dx);
    Array<T, 3> outletPos((outletRealPos - LBLocation) / dx);

    plint diameter = util::roundToInt(diameterReal / dx);

    Box3D inletDomain(util::roundToInt(inletPos[0] - diameter), util::roundToInt(inletPos[0] + diameter),
                      util::roundToInt(inletPos[1] - diameter), util::roundToInt(inletPos[1] + diameter),
                      util::roundToInt(inletPos[2]), util::roundToInt(inletPos[2]));
    Box3D behindInlet(inletDomain.x0, inletDomain.x1,
                      inletDomain.y0, inletDomain.y1,
                      inletDomain.z0 - diameter, inletDomain.z0 - 1);

    Box3D outletDomain(util::roundToInt(outletPos[0] - diameter), util::roundToInt(outletPos[0] + diameter),
                       util::roundToInt(outletPos[1] - diameter), util::roundToInt(outletPos[1] + diameter),
                       util::roundToInt(outletPos[2]), util::roundToInt(outletPos[2]));
    Box3D behindOutlet(outletDomain.x0, outletDomain.x1,
                       outletDomain.y0, outletDomain.y1,
                       outletDomain.z0 + diameter, outletDomain.z0 + 1);

    if (constantFlow)
    {
        boundaryCondition->addVelocityBoundary2N(inletDomain, *lattice);
        setBoundaryVelocity(*lattice, inletDomain, PoiseuilleVelocity3D(uAveLB));
        //setBoundaryVelocity(*lattice, inletDomain, Array <T,3> (0., 0., uAveLB));
        boundaryCondition->addPressureBoundary2P(outletDomain, *lattice);
        setBoundaryDensity(*lattice, outletDomain, 1.);
    }
    else
    {
        if (iter == 0)
        {
            boundaryCondition->addVelocityBoundary2N(inletDomain, *lattice);
            setBoundaryVelocity(*lattice, inletDomain, PoiseuilleVelocity3D(uAveLB));
            //setBoundaryVelocity(*lattice, inletDomain, Array <T,3> (0., 0., uAveLB));
            boundaryCondition->addPressureBoundary2P(outletDomain, *lattice);
            setBoundaryDensity(*lattice, outletDomain, 1.);
        }
        else
        {
            boundaryCondition->addPressureBoundary2N(inletDomain, *lattice);
            setBoundaryDensity(*lattice, inletDomain, inletP);

            boundaryCondition->addPressureBoundary2P(outletDomain, *lattice);
            setBoundaryDensity(*lattice, outletDomain, outletP);
        }
    }

    dynamics = new BounceBack<T, DESCRIPTOR>(1.);
    defineDynamics(*lattice, *flagMatrixInside, lattice->getBoundingBox(), dynamics, 0);

    dynamics = new BounceBack<T, DESCRIPTOR>(1.);
    defineDynamics(*lattice, behindInlet, dynamics);

    dynamics = new BounceBack<T, DESCRIPTOR>(1.);
    defineDynamics(*lattice, behindOutlet, dynamics);

    // Switch all remaining outer cells to no-dynamics, except the outer
    //   boundary layer, and keep the rest as BGKdynamics.
    dynamics = new NoDynamics<T, DESCRIPTOR>;
    defineDynamics(*lattice, LB.voxelizedDomain->getVoxelMatrix(), lattice->getBoundingBox(),
                   dynamics, voxelFlag::outside);

    // Initialize density and velocity fields
    if (iter == 0)
    {
        initializeAtEquilibrium(*lattice, lattice->getBoundingBox(), initializeDensityAndVelocity<T>(uAveLB));
        //initializeAtEquilibrium (*lattice, lattice -> getBoundingBox(), 1., Array<T,3>(0., 0., uAveLB) );
        lattice->initialize();
    }
    else
    {
        saveBinaryBlock(*(LB.lattice), "checkpoint.dat");
    }

    updateLB(*lattice);

    LB.assign(lattice);

    if (iter == 0)
    {
        inletP = computeMax(*computeDensity(*LB.lattice, inletDomain), inletDomain);
        outletP = 1.;
    }
}

// Write out the velocity field and platelet positions for visualization
void writeOutputFiles()
{
    plb_ofstream file;
    std::string fname = "trajectory";
    fname = fname + ".csv.";
    fname = fname + std::to_string(iter);
    char filename[fname.size() + 1];
    fname.copy(filename, fname.size() + 1);
    filename[fname.size()] = '\0';
    file.open(filename, ios::app);

    file << "xcoord, ycoord, zcoord, activation" << endl;

    for (auto i = plateletList.begin(); i != plateletList.end(); ++i)
    {
        platelet plt = *i;

        if (!plt.bound)
        {
            continue;
        }

        Array<T, 3> realPos;

        realPos[0] = plt.center[0] * h + KMCLocation[0];
        realPos[1] = plt.center[1] * h + KMCLocation[1];
        realPos[2] = plt.center[2] * h + KMCLocation[2];

        file << realPos[0] << ",";
        file << realPos[1] << ",";
        file << realPos[2] << ",";
        file << plt.calcInteg << endl;
    }

    file.close();
    if (iter % 200 == 0)
        writeVTK(*LB.lattice, dx, dt, iter);
}

// This function is used to pass information to OpenFOAM concentration
// solver using MUI as the message passing toolkit
void interface_solution(std::vector<std::unique_ptr<mui::uniface1d>> &ifs,
                        mui::chrono_sampler_exact1d chrono_sampler)
{
    mui::point1d push_point;

    plint count = 0;
    // Push platelet centers and release times to OpenFOAM
    for (auto current = plateletList.begin(); current != plateletList.end(); ++current)
    {
        platelet plt = *current;
        Array<T, 3> realPos((T)plt.center[0], (T)plt.center[1], (T)plt.center[2]);
        realPos = h * realPos + LBLocation;
        push_point[0] = count;
        ifs[0]->push("px", push_point, realPos[0]);
        ifs[0]->push("py", push_point, realPos[1]);
        ifs[0]->push("pz", push_point, realPos[2]);
        ifs[0]->push("release", push_point, plt.releaseTime);
        ifs[0]->push("calcinteg", push_point, plt.calcInteg);
        ++count;
    }

    ifs[0]->commit(iter);
    ifs[0]->forget(iter - 1);

    // Obtain local concentration of agonist species at each platelet position
    auto adpC = ifs[1]->fetch_values<T>("adp", iter, chrono_sampler);
    auto txa2C = ifs[1]->fetch_values<T>("txa2", iter, chrono_sampler);
    auto thrombinC = ifs[1]->fetch_values<T>("thrombin", iter, chrono_sampler);

    ifs[1]->forget(iter);

    count = 0;
    for (auto current = plateletList.begin(); current != plateletList.end(); ++current)
    {
        platelet plt = *current;
        plt.concentrations[0] = adpC[count];
        plt.concentrations[2] = thrombinC[count];
        plt.concentrations[3] = txa2C[count];

        *current = plt;
        ++count;
    }

    // Get list of grid points from OpenFOAM where velocity needs to be obtained
    auto ofGridId = ifs[2]->fetch_points<T>("ofgridx",
                                            iter, chrono_sampler);
    auto ofGridx = ifs[2]->fetch_values<T>("ofgridx",
                                           iter, chrono_sampler);
    auto ofGridy = ifs[2]->fetch_values<T>("ofgridy",
                                           iter, chrono_sampler);
    auto ofGridz = ifs[2]->fetch_values<T>("ofgridz",
                                           iter, chrono_sampler);

    ifs[2]->forget(iter);

    pcout << "Palabos obtained grid points from OpenFOAM" << endl;

    // Interpolate velocity at each of the OpenFOAM grid points
    // Then, push these values back to the OpenFOAM interface
    for (plint i = 0; i < ofGridId.size(); ++i)
    {
        obtainVelocity(ofGridx[i], ofGridy[i], ofGridz[i]);
        push_point[0] = ofGridId[i][0];
        ifs[3]->push("ux", push_point, velocity[0]);
        ifs[3]->push("uy", push_point, velocity[1]);
        ifs[3]->push("uz", push_point, velocity[2]);
    }

    ifs[3]->commit(iter);
    ifs[3]->forget(iter - 1);

    pcout << "Pushed values to MUI interface" << endl;
    ++iter;
}
