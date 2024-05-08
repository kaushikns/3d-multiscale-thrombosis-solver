/*
Author: Kaushik N Shankar
*/

#include "palabos3D.h"
#include "palabos3D.hh"
#include <unordered_map>
#include <map>
#include <vector>
#include <random>
#include <algorithm>
#include "mui.h"

using namespace plb;
using namespace std;

typedef double T;
#define DESCRIPTOR descriptors::D3Q19Descriptor

long long inCount = 0;
long long removeCount = 0;
int countoverlapbound = 0;

MPI_File fileflag;
MPI_File filevel;
MPI_File fileshear;

std::string fileflagName;
std::string filevelName;
std::string fileshearName;

int numprocs = 0;    // Number of parallel MPI processes
int procid = 0;      // Rank of the MPI process
T totalRate = 0.;    // Sum of rates of all events in a processor
T nullRate = 0.;     // Rate of null event in a processor
T maxTotalRate = 0.; // Maximum of the sum of rates in a processor
T tempMaxTotalRate = 0.;
MPI_Request req;
MPI_Comm world; // MPI World communicator
MPI_Datatype mpi_platelet, mpi_commplatelet;

long long int movecount = 0;
long long int nullcount = 0;
long long int exceedcount = 0;
plint update_freq = 3600;

bool loadFromCheckpoint = false;
T checkPointTime = 0.;
plint numprocs_checkpoint = 0;

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

T inletP = 1.;
T outletP = 1.;
T deltaPTotal = 0.001;
T deltaPSys0 = 0.;
bool presFlag = false;
T averageInletVelocityComputed = 0.;
vector<T> inletPs(10);

plint margin = 1;      // Extra margin of allocated cells around the obstacle.
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

plint referenceDirection = 0;
plint openingSortDirection = 0;

T epsilon = 0.;

std::auto_ptr<MultiScalarField3D<int>> flagMatrixInside;
// This will be useful in determining whether a grid point lies inside or outside the domain
TriangleSet<T> *triangleSet = 0;
Array<T, 3> inletRealPos(0., 0., 0.);
Array<T, 3> outletRealPos(0., 0., 0.);

T diameterReal = 0.;
T radiusReal = 0.;
T radiusEffectivex = 0.;
T radiusEffectivey = 0.;
plint minx, maxx, miny, maxy;

Array<T, 3> KMCLocation;
Array<T, 3> LBLocation;
std::auto_ptr<MultiScalarField3D<int>> boolMask;

T dtLB = 0.1;                       // LB interface coupling time step
T dx = 0.;                          // LB grid size
T dt = 0.;                          // LB time step
T h = 0.;                           // LKMC grid size
plint nx = 0, ny = 0, nz = 0;       // LKMC resolution
plint nxlb = 0, nylb = 0, nzlb = 0; // LB resolution
plint factor = 0;                   // Ratio of LB grid to KMC grid size
plint diaFactor = 0;                // Ratio of platelet diameter to KMC grid size
T pltDia = 3.e-6;                   // Platelet diameter

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

// Variable to keep track of time
T totalTime = 0.;
plint tsteps = 0;
T totalTimeNRM = 0.;
plint iter = 0;
plint iter_checkpoint = 0;

// A temporary variable used to assign and store local velocity
Array<T, 3> velocity(0., 0., 0.);

// A temporary variable used to assign and store local platelet drift velocity
Array<T, 3> driftVelocity(0., 0., 0.);

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
    std::unique_ptr<MultiBlockLattice3D<T, DESCRIPTOR>> lattice;
    std::auto_ptr<MultiScalarField3D<T>> shearRate;
    std::auto_ptr<VoxelizedDomain3D<T>> voxelizedDomain;

    LBClass()
    {
        lattice.reset();
    }

    void getVoxelizedDomain(plint level = 0);

    void assign(std::unique_ptr<MultiBlockLattice3D<T, DESCRIPTOR>> &lattice_)
    {
        lattice.swap(lattice_);
        lattice_.reset();
        // updateShear();
    }
    void updateShear()
    {
        auto strainRate = *computeStrainRateFromStress(*lattice);
        auto shearRate_ = *computeSymmetricTensorNorm(strainRate, lattice->getBoundingBox());
        shearRate.reset(new MultiScalarField3D<T>(shearRate_));
    }
};

LBClass LB;

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
long intmax = std::numeric_limits<long>::max();
T machineEps = std::numeric_limits<T>::epsilon();

// Random number generator
std::mt19937 engine(0);
std::mt19937 engine_inlet(0);
std::mt19937 engine_parallel(0);
std::uniform_real_distribution<T> distribution(0., 1.);

// NN parameters
std::string nnInputFile;
vector<Array<Array<T, 6>, 8>> IW;
vector<Array<Array<T, 8>, 8>> A1;
vector<Array<Array<T, 1>, 8>> b1;
vector<Array<Array<T, 8>, 4>> LW1;
vector<Array<Array<T, 8>, 4>> A2;
vector<Array<Array<T, 1>, 4>> b2;
vector<Array<Array<T, 4>, 1>> LW2;
vector<Array<Array<T, 1>, 1>> b3;
Array<T, 4> ec50;

std::string donorid;

T dtNN = 0.1;

// Parameters to determine the activation state of a platelet
// from the calcium integral function xi(t)
T alpha = 0.;
T nHill = 0.;
T xi50 = 0.;
bool iloprostFlag = false; // Flag for iloprost treatment
bool gsnoFlag = false;     // Flag for GSNO treatment

struct flagMatrix
{
    int val;

    flagMatrix()
    {
        val = 0;
    }

    int get(plint px, plint py, plint pz)
    {
        MPI_Status stat;
        MPI_Offset offset;
        plint loc = pz + py * nzlb + px * nzlb * nylb;
        offset = loc * sizeof(int);
        MPI_File_read_at(fileflag, offset, &val, 1, MPI_INT, &stat);
        return val;
    }
};

flagMatrix flagMatrixLKMC;

// A structure that defines information about a platelet
struct platelet
{
    Array<plint, 3> center; // Position of the platelet
    T dp;                   // Platelet diameter
    long layer;             // Layer of the clot platelet belongs to.
    bool unbound;           // Flag for whether the platelet belongs to a layer of the clot which came off

    // KMC event specific parameters
    Array<T, 6> convection;     // Convection rates in each direction
    Array<T, 6> motion;         //  Motion rates in each direction // 0,2,4 +ve x,y,z; 1,3,5 -ve x,y,z
    Array<plint, 6> blockCount; // No. of platelets blocking motion in each direction
    T binding;                  // Net binding rate with collagen or any other neighboring platelet
    T unbinding;                // Effective unbinding rate of platelet from clot mass
    bool bound;                 // Binding state of platelet (whether it is bound and there is connectivity collagen / TF)
    Array<T, 8> tau;            // Putative time after which each event occurs (6 motion, 1 binding and 1 unbinding)
    T localShear;               // Local shear around a platelet
    T lastMotionTime;

    // Parameters particular to reactive agonist species
    Array<T, 4> concentrations;
    // 0:ADP, 1:Coll, 2:Thrombin, 3:TXA2
    T releaseTime; // Time after which platelet starts releasing agonist species

    // Parameters estimated using NN (activation state)
    T calcInteg;                   // Integral of intraplatelet calcium (xi)
    T activation;                  // Activation state of platelet [F(xi)]
    T activationx;                 // Recent history activation
    Array<T, 129> nnHistory;       // NN history of the platelet, an input into NN which determines the activation
    Array<T, 31> calcIntegHistory; // Calcium integral history of the platelet

    T sumRates; // Required for platelet redistribution across procs
    plint lastMotionType;

    platelet()
    {
        dp = pltDia;
        center[0] = center[1] = center[2] = 0;
        layer = intmax / 2;
        unbound = false;

        for (plint i = 0; i < 6; ++i)
        {
            convection[i] = 0.;
            motion[i] = 0.;
            blockCount[i] = 0;
        }
        lastMotionTime = totalTime;

        binding = 0.;
        unbinding = 0.;

        for (plint i = 0; i < 8; ++i)
        {
            tau[i] = dblmax;
        }
        for (plint i = 0; i < 4; ++i)
        {
            concentrations[i] = 0.;
        }
        for (plint i = 0; i < 129; ++i)
        {
            nnHistory[i] = -1.;
        }
        for (plint i = 0; i < 31; ++i)
        {
            calcIntegHistory[i] = 0.;
        }

        bound = false;
        releaseTime = -1.;
        calcInteg = 0.;
        activation = alpha;
        activationx = 2.;
        localShear = 0.;

        lastMotionType = 4;
    }
};

// List of platelets
vector<platelet> plateletList;

T inletTau = dblmax; // Putative time for inlet of platelet into domain

platelet dummyPlatelet;
// Dummy platelet to which the inlet of new platelet into domain & null event is associated
// This list will contain only one platelet, which is the dummyplatelet

T nullTau = dblmax; // Putative time for null event  (parallel KMC)

// Flag for whether a binding or an unbinding event occurred between successive iterations of LB/FVM/NN
plint nUnbindingOccurred = 0;
vector<long long int> unboundLocationsLocal;
vector<long int> unboundLevelsLocal;
vector<plint> unboundListLocal;
struct commPlatelet
{
    int idx;
    int proc;
    long long int loc;
    T activation;
    T bondTime;
    int type;
};
vector<commPlatelet> commPlateletList;

// A variable used to keep track of platelet occupation as function of lattice spacing
// If queried, it returns the activation state of the platelet
// std::unordered_map<long long int, T> occupation;
// std::map<long long int, plint> occupation;
std::unordered_multimap<long long int, plint> occupation;
std::map<long long int, T> occupationGlobal;
std::unordered_set<plint> boolMaskLKMC;

// A structure which is used to store the events in the sorted event list
struct event
{
    plint idx;
    plint type;
    T tau;

    event(plint idx_, plint type_, T tau_) : idx(idx_), type(type_), tau(tau_)
    {
    }
};

// This structure is used to dictate how the events are ordered in the list
// In this case, they are sorted according to increasing tau values (next reaction method)
struct orderEvent
{
    bool operator()(const event &e1, const event &e2) const
    {
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
        return e1.tau < e2.tau;
    }
};

std::set<inletEvent, orderInletEvent> inletEventList;

// Read the NN input parameters from XML file
template <pluint m, pluint n>
void readNNParameters(std::string paramName, Array<Array<T, n>, m> &arr);

// Read the user input XML file provided.
void readParameters();

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

// Generate random number between 0 and 1
// Parallel synchronized random number generation
T randn_parallel();

// Updates the putative time of the event occurrence given old and new rates
T updateEventTime(T tau, T rateOld, T rateNew);

// Returns the square of the distance between the centers of two platelets in lattice units
plint distanceBetween(platelet a, platelet b);

// Tells if a given lattice point lies inside the platelet or not
bool liesInside(Array<plint, 3> position, platelet plt);

// This function obtains the largest cuboidal bounding box that surrounds a platelet
Box3D boundingBox(platelet &plt);

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
list<plint> checkOverlap(plint current, plint dir);

// This function checks if there are overlaps between a platelet and any other platelet
// If yes, it returns the head of a linked list containing their addresses
// The above implementation is used for the convective PFA
// The implementation below is used for checking available platelets for binding
// This difference is due to the difference in the PFA algorithm and the binding algorithm
list<T> checkOverlapBinding(plint current);

// This function is used to assign the platelet drift velocity given the position of the platelet center
inline void obtainDriftVelocity(Array<T, 3> pos);

// Reads velocity from the file output by Palabos
Array<T, 3> readVelocity(plint px, plint py, plint pz);

// Peforms trilinear interpolation of velocity at a given location in lattice units
void interpolateVelocity(T px, T py, T pz);

// This function is used to assign the velocity given the position of the platelet center
inline void obtainVelocity(plint cx, plint cy, plint cz);

// Checks whether a platelet has left the domain
bool checkWithinDomain(plint cx, plint cy, plint cz);

// Checks whether a platelet can interact with the reactive patch
// This function is subject to change with a change in the location of the patch
bool checkCollagen(plint cx, plint cy, plint cz);

// This function ensures that all the rates are defined such that
// a move where a platelet leaves the domain is not allowed
void checkWithinDomain(plint current);

// This function implements the simple blocking algorithm SBA(Flamm et al. J Chem Phys 2009)
// Sets convective rate to zero if motion is blocked by another platelet in a givne direction
// More computationally efficient than the pass forward algorithm
// Inaccurate physical representation due to blocking is insignificant because platelet density is low
bool simpleBlocking(plint current, plint dir);

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
bool addPlateletInlet();

// Computes the enhancement in binding rates due to vwf stretching
T vwfFactor(plint current);

// Computes the enhancement in detachment rate due to shear
T detachFactor(plint current);

// This functional is used to set the binding rates associated with
// the given platelet as the argument
void setBindingRates(plint current);

// This functional is used to set the unbinding rates associated with
// the given platelet as the argument
void setUnbindingRates(plint current);

// This function sets the rate of each event for a given platelet that moves
// More efficient to use this when the velocity field does not change
void setRates(plint current, plint type);

// This function sets the rate of each event for each platelet
// Alters the rate parameters where the move is not allowed
// by calling the convective pass-forward algorithm routine
void setRates();

// Creating custom MPI type to exchange info about platelets
void defineMPIStruct(MPI_Datatype &type);

// Creating custom MPI type to exchange info about platelets
void defineMPIStructComm(MPI_Datatype &type);

bool sortByLocation(const platelet &a, const platelet &b);

void distributePlatelets();

void updateEventList();

// Removes platelets that are no longer in the domain from the platelet list
// Also frees up heap memory associated with STL allocation
void removeExitPlatelets();

// Used to initalize the LKMC lattice with given initial platelet distribution
void iniKMCLattice();

// Update the unordered map that contains the platelet positions as a function of
// lattice spacing across all processors
void updatePlateletMap(plint type, platelet &plt);

// Update the unordered map that contains the platelet positions as a function of
// lattice spacing across all processors
void updatePlateletMap();

// Obtain the keys and values in a given unordered_map as two vectors
template <typename T1, typename T2>
// void mapToVec(unordered_map<T1,T2>& umap, vector<T1> & vec_keys, vector<T2> & vec_vals);
void mapToVec(map<T1, T2> &umap, vector<T1> &vec_keys, vector<T2> &vec_vals);

// Obtain the keys and values in a given unordered_map as two vectors
template <typename T1, typename T2>
// void vecToMap(unordered_map<T1,T2>& umap, vector<T1> & vec_keys, vector<T2> & vec_vals);
void vecToMap(map<T1, T2> &umap, vector<T1> &vec_keys, vector<T2> &vec_vals);

// This function runs one iteration of the LKMC algorithm
void runKMC();

// Reads shear rate from file output by Palabos
T readShear(plint px, plint py, plint pz);

// This function is used to compute the local shear around a platelet
T computePltShear(plint current);

// Write out the velocity field and platelet positions for visualization
void writeOutputFiles();

// Checkpoint the simulation data to resume
void checkpointSimulation();

// Load previously checkpointed data
void loadSimulation(plint iter, plint numprocs_checkpoint);

/************************** Lattice Boltzmann routines ********************************/

// Note: This is just a dummy. Used by an internal Palabos class to determine whether a given
// point lies inside or outside the domain
// This function assigns proper boundary conditions to the openings of the surface geometry
// Which opening is inlet and which is outlet is defined by the user in the input XML file.
void setOpenings(
    std::vector<BoundaryProfile3D<T, Array<T, 3>> *> &inletOutlets,
    TriangleBoundary3D<T> &boundary, T uLB);

void determineEffectiveRadius();

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

/// A functional used to update the boolmask
// The boolmask is used to assign bounceback nodes
// Bounceback nodes will be insantiated at locations where the boolmask is 1
class updateBoolMask : public BoxProcessingFunctional3D_S<int>
{
public:
    updateBoolMask(plint iter_)
    {
        iter = iter_;
    }

    virtual void process(Box3D domain, ScalarField3D<int> &scalarField)
    {
        Dot3D offset = scalarField.getLocation();
        plint iX, iY, iZ;

        plint pltRad = util::roundToInt(0.5 * pltDia / dx);

        for (auto it = occupationGlobal.begin(); it != occupationGlobal.end(); ++it)
        {
            if (it->second < 100.)
            {
                continue;
            }

            long long int loc = it->first;

            long long int cx, cy, cz, cx_, cy_, cz_;

            cx = loc / (ny * nz);
            cy = (loc / nz) % ny;
            cz = loc % nz;

            cx_ = cx % factor;
            cy_ = cy % factor;
            cz_ = cz % factor;

            cx = cx / factor;
            cy = cy / factor;
            cz = cz / factor;

            for (plint i = -pltRad; i <= pltRad; ++i)
            {
                for (plint j = -pltRad; j <= pltRad; ++j)
                {
                    for (plint k = -pltRad; k <= pltRad; ++k)
                    {
                        iX = (plint)cx + i - offset.x;
                        iY = (plint)cy + j - offset.y;
                        iZ = (plint)cz + k - offset.z;

                        if (iX < domain.x0 || iX > domain.x1)
                            continue;
                        if (iY < domain.y0 || iY > domain.y1)
                            continue;
                        if (iZ < domain.z0 || iZ > domain.z1)
                            continue;

                        if (i * i + j * j + k * k <= pltRad * pltRad)
                        {
                            scalarField.get(iX, iY, iZ) = 1;
                        }
                    }
                }
            }
        }
    }

    virtual updateBoolMask *clone() const
    {
        return new updateBoolMask(*this);
    }

    void getTypeOfModification(std::vector<modif::ModifT> &modified) const
    {
        modified[0] = modif::staticVariables;
    }

    virtual BlockDomain::DomainT appliesTo() const
    {
        return BlockDomain::bulkAndEnvelope;
    }

private:
    plint iter;
};

/// Write the velocity field into a VTK file.
void writeVTK(MultiBlockLattice3D<T, DESCRIPTOR> &lattice, T dx, T dt, plint iter);

/// This is the function that prepares the actual LB simulation.
void updateLB(MultiBlockLattice3D<T, DESCRIPTOR> &lattice);

// This is the function that prepares and performs the actual simulation.
void runLB(plint level = 0, MultiBlockLattice3D<T, DESCRIPTOR> *iniVal = 0);

// Save boundary conditions for checkpointed simulation.
void saveCheckpointBC();

// Load boundary conditions for checkpointed simulation.
void loadCheckpointBC(plint iter);

/********* Routine that uses MUI to exchange solutions between different solvers ***********/

// This function is used to pass information to OpenFOAM concentration
// solver using MUI as the message passing toolkit
void interface_solution(std::vector<std::unique_ptr<mui::uniface<mui::one_dim>>> &ifs,
                        mui::chrono_sampler_exact<mui::one_dim> chrono_sampler);

T pfatime = 0.;
T getvelocitytime = 0.;
T getsheartime = 0.;
T checkdomaintime = 0.;
T bindingratesettime = 0.;
T commtime = 0.;
T mapupdatetime = 0.;
T bouncebacksettime = 0.;
T lbconvergencetime = 0.;
T lbtime = 0.;
T totalRunTime = 0.;
T ifstime = 0.;

int main(int argc, char *argv[])
{
    // Initialization of MUI interface
    world = mui::mpi_split_by_app();
    std::string dom = "kmcnnlb";
    std::vector<std::string> interfaces;
    interfaces.push_back("comma/ifs");
    interfaces.push_back("commb/ifs");
    interfaces.push_back("commc/ifs");
    auto ifs = mui::create_uniface<mui::one_dim>(dom, interfaces);

    mui::chrono_sampler_exact<mui::one_dim> chrono_sampler;

    // plbInit(&argc, &argv);
    global::mpi().init(world);
    global::directories().setOutputDir("tmp");
    global::IOpolicy().activateParallelIO(true);
    global::IOpolicy().setIndexOrderingForStreams(IndexOrdering::forward);

    MPI_Comm_size(world, &numprocs);
    MPI_Comm_rank(world, &procid);

    defineMPIStruct(mpi_platelet);
    defineMPIStructComm(mpi_commplatelet);

    int seed1 = 823543, seed2 = 823543;
    seed1 += procid;
    seed2 += numprocs + procid;
    engine.seed(seed1);
    engine_inlet.seed(seed2);
    engine_parallel.seed(7);

    global::timer("runtime").restart();

    // Read the simulation parameters from XML file
    readParameters();

    T dtLB = 0.1;

    if (loadFromCheckpoint)
    {
        totalTime = checkPointTime;
        iter = util::roundToInt(totalTime / dtLB);
        iter_checkpoint = iter;

        referenceResolution = (plint)(referenceLength / dx);

        LB.getVoxelizedDomain();

        pcout << "LB lattice spacing:" << dx << endl;

        pcout << "LB time step:" << dt << endl;

        loadCheckpointBC(iter);
        generateKMCLattice();
        loadSimulation(iter, numprocs_checkpoint);
    }
    else
    {
        plint iniLevel = 0;
        plint maxLevel = 0;
        std::unique_ptr<MultiBlockLattice3D<T, DESCRIPTOR>> iniConditionLattice(nullptr);

        // This code incorporates the concept of smooth grid refinement until convergence is
        //   achieved. The word ``smooth'' indicates that as the refinement level increases
        //   by one, the whole grid doubles in each direction. When the grid is refined, both
        //   dx and dt have to change. Whether dt is changed as dx^2 (diffusive behavior)
        //   or as dx (convective behavior), is controlled by the input variable
        //   ``convectiveScaling'' (the recommended choice is not to use convective scaling).

        plint level;
        referenceResolution = (plint)(referenceLength / dx);
        for (level = iniLevel; level <= maxLevel; ++level)
        {
            pcout << std::endl
                  << "Running LB simulation at level " << level << endl;

            LB.getVoxelizedDomain(level);

            pcout << "LB lattice spacing:" << dx << endl;

            pcout << "LB time step:" << dt << endl;

            global::timer("LB").restart();
            runLB(level, iniConditionLattice.get());
            lbtime = lbtime + global::timer("LB").getTime();

            if (level != maxLevel)
            {
                plint dxScale = -1;
                plint dtScale = -2;
                if (convectiveScaling)
                {
                    dtScale = -1;
                }

                // The converged simulation of the previous grid level is used as the initial
                // condition
                //   for the simulation at the next grid level (after appropriate interpolation has
                //   taken place).

                iniConditionLattice = std::unique_ptr<MultiBlockLattice3D<T, DESCRIPTOR>>(refine(
                    *LB.lattice, dxScale, dtScale, new BGKdynamics<T, DESCRIPTOR>(1.)));
            }
        }
    }

    global::timer("ifs").restart();
    interface_solution(ifs, chrono_sampler);
    ifstime = ifstime + global::timer("ifs").getTime();

    if (!loadFromCheckpoint)
    {
        generateKMCLattice();
    }

    determineEffectiveRadius();

    normalizeConcentration();

    oldInletRate = computeFlowRate();
    pcout << "Computed fluid flow rate:" << oldInletRate << endl;

    // Divide platelets among nprocs MPI processes
    nPlatelets = nPlatelets / numprocs;
    update_freq = update_freq / numprocs;

    pcout << "KMC lattice spacing:" << h << endl;

    pcout << nx << endl
          << ny << endl
          << nz << endl;

    // Diffusive rate of motion
    //(This remains a constant throughout the simulation because both
    // the dispersion coeff. and grid size do not change)
    diffusion = diffusivity / (h * h);

    if (!loadFromCheckpoint)
    {
        pcout << "Initializing KMC lattice" << endl;
        iniKMCLattice();
        pcout << "KMC lattice initialized with platelets" << endl;
    }

    setRates();
    updatePlateletMap();
    updateEventList();

    T nntime = 0.;
    T ratesettime = 0.;
    T rateupdatetime = 0.;
    T flowratetime = 0.;
    T kmctime = 0.;

    plint iterNN = iter;

    commtime = 0.;
    mapupdatetime = 0.;

    while (totalTime < 360.)
    {
        global::timer("kmc").restart();
        runKMC();
        kmctime += global::timer("kmc").getTime();
        ;

        if (totalTime >= iterNN * dtNN)
        {
            pcout << "Updating NN history" << endl;
            global::timer("nn").restart();
            integrateNN();
            ++iterNN;
            nntime += global::timer("nn").getTime();
        }

        if (totalTime >= iter * dtLB && iter != floor(360. / dtLB))
        {
            removeExitPlatelets();

            pcout << "Occupation size: " << occupation.size() << endl;

            updatePlateletMap();

            pcout << "Setting up interaction of KMC/NN with LB/FVM module" << endl;
            MPI_Barrier(world);
            global::timer("ifs").restart();
            interface_solution(ifs, chrono_sampler);
            MPI_Barrier(world);
            ifstime = ifstime + global::timer("ifs").getTime();
            pcout << "Interaction of KMC/NN achieved with other modules successfully, starting next iteration" << endl;

            oldInletRate = computeFlowRate();

            cout << "Number of platelets in domain: " << nPlatelets << endl;
            cout << "Number of bound platelets: " << nBoundPlatelets << endl;
            cout << "Number of unbindings: " << nUnbindingOccurred << endl;

            nUnbindingOccurred = 0;

            pcout << "Setting KMC rates based on updated NN history and LB velocity fields" << endl;
            setRates();
            updatePlateletMap();
            updateEventList();

            if (iter % 50 == 0)
            {
                writeOutputFiles();
            }
            if (iter % 100 == 0)
            {
                checkpointSimulation();
                saveCheckpointBC();
                saveBinaryBlock(*LB.lattice, "checkpoint" + std::to_string(iter) + ".dat");
            }
            pcout << "Velocity interpolation time = " << getvelocitytime << endl;
            pcout << "Shear interpolation time = " << getsheartime << endl;
            pcout << "Check within domain time = " << checkdomaintime << endl;

            pcout << "KMC time = " << kmctime << endl;
            pcout << "Comm time = " << commtime << endl;
            pcout << "Map update time = " << mapupdatetime << endl;

            pcout << "NN time = " << nntime << endl;

            pcout << "Total time = " << totalTime << endl;
            pcout << "NRM time = " << totalTimeNRM << endl;

            pcout << "LB time = " << lbtime << endl;
            pcout << "Bounceback node set time = " << bouncebacksettime << endl;
            pcout << "LB Convergence time = " << lbconvergencetime << endl;

            pcout << "Interface time = " << ifstime << endl;

            totalRunTime = totalRunTime + global::timer("runtime").getTime();
            global::timer("runtime").restart();
            pcout << "Total run time = " << totalRunTime << endl;

            int totalBound = 0;
            MPI_Allreduce(&nBoundPlatelets, &totalBound, 1, MPI_INT, MPI_SUM, world);
            pcout << "Total bound platelets:" << totalBound << endl;

            // Write platelet count bound versus time onto file
            plb_ofstream file;
            std::string fname = "plateletcount.csv";
            char filename[fname.size() + 1];
            fname.copy(filename, fname.size() + 1);
            filename[fname.size()] = '\0';
            file.open(filename, ios::app);
            file << totalTime << "," << totalBound << endl;
            file.close();

            MPI_Barrier(world);
        }
    }

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
    std::string paramXmlFileName = "param.xml";

    // Read the parameter XML input file.
    XMLreader document(paramXmlFileName);

    std::string line;
    ifstream file("directoryinfo.txt");
    getline(file, line);
    filevelName = line;
    getline(file, line);
    fileflagName = line;
    getline(file, line);
    fileshearName = line;
    file.close();
    fileflagName = fileflagName + ".dat";
    filevelName = filevelName + ".dat";
    fileshearName = fileshearName + ".dat";

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
    document["geometry"]["dimensions"]["diameter"].read(diameterReal);
    document["geometry"]["dimensions"]["lz"].read(lz);
    radiusReal = 0.5 * diameterReal;

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
    document["nn"]["iloprostflag"].read(iloprostFlag);
    document["nn"]["gsnoflag"].read(gsnoFlag);
    document["nn"]["donorid"].read(donorid);

    document["fluid"]["kinematicViscosity"].read(kinematicViscosity);
    document["fluid"]["density"].read(fluidDensity);

    document["numerics"]["referenceDirection"].read(referenceDirection);
    document["numerics"]["referenceLength"].read(referenceLength);
    document["numerics"]["hLB"].read(dx);
    // document["numerics"]["referenceResolution"].read(referenceResolution);
    document["numerics"]["dt"].read(dt);
    document["numerics"]["hLKMC"].read(h);

    document["simulation"]["epsilon"].read(epsilon);
    document["simulation"]["performOutput"].read(performOutput);
    document["simulation"]["useIncompressible"].read(useIncompressible);
    document["simulation"]["poiseuilleInlet"].read(poiseuilleInlet);
    document["simulation"]["convectiveScaling"].read(convectiveScaling);
    document["simulation"]["constantFlow"].read(constantFlow);

    // Read the platelet inlet distribution function from the input xml file
    std::string inletConcFile = "/jet/home/kaushik7/inlet_conc.xml";
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
    IW.resize(80);
    A1.resize(80);
    b1.resize(80);
    LW1.resize(80);
    A2.resize(80);
    b2.resize(80);
    LW2.resize(80);
    b3.resize(80);

    for (plint q = 0; q < IW.size(); ++q)
    {
        nnInputFile = "/jet/home/kaushik7/nninput/nn_input_" + std::to_string(q) + ".xml";
        readNNParameters("IW", IW[q]);
        readNNParameters("A1", A1[q]);
        readNNParameters("b1", b1[q]);
        readNNParameters("LW1", LW1[q]);
        readNNParameters("A2", A2[q]);
        readNNParameters("b2", b2[q]);
        readNNParameters("LW2", LW2[q]);
        readNNParameters("b3", b3[q]);
    }

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

    // If loading and resuming previous simulation from a checkpoint
    document["checkpoint"]["flag"].read(loadFromCheckpoint);
    document["checkpoint"]["time"].read(checkPointTime);
    document["checkpoint"]["numprocs"].read(numprocs_checkpoint);
}

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

    T output = 0.;

    plint count = 0;

    for (plint q = 0; q < IW.size(); ++q)
    {
        auto layer1 = transform<8, 1>(matAdd<8, 1>(matAdd<8, 1>(matMultiply<8, 6, 6, 1>(IW[q], c), matMultiply<8, 8, 8, 1>(A1[q], z)), b1[q]));
        auto layer2 = transform<4, 1>(matAdd<4, 1>(matAdd<4, 1>(matMultiply<4, 8, 8, 1>(LW1[q], layer1), matMultiply<4, 8, 8, 1>(A2[q], z)), b2[q]));

        auto y = matAdd<1, 1>(matMultiply<1, 4, 4, 1>(LW2[q], layer2), b3[q]);

        output += y[0][0];
        count += 1;
    }

    output = output / ((T)count);

    return output;
}

// This function is used to obtain the NN history of a platelet at time t0
T obtainHistory(platelet &plt, plint t0)
{
    if (t0 < 0)
        return -1.;

    return plt.nnHistory[t0 % 129];
}

// This function is used to obtain the recent history calcium integral of a platelet
T obtainRecentCalcInteg(platelet &plt, plint t0)
{
    if (t0 < 30)
        return 0.;

    t0 = t0 - 30;

    return plt.calcIntegHistory[t0 % 31];
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

    for (auto current = 0; current < plateletList.size(); ++current)
    {
        plt = plateletList[current];

        if (plt.bound && checkCollagen(plt.center[0], plt.center[1], plt.center[2]))
        {
            plt.concentrations[1] = 4 * ec50[1];
        }
        else
        {
            plt.concentrations[1] = 0.;
        }

        // If the platelet lies within the region for the thin film approximation
        // Then, set the local thrombin concentration to the free thrombin conc.
        // From Chen at al, Plos Comput Biol, 2019
        if (plt.center[2] > nz / 4 && plt.center[2] < 3 * nz / 4 && plt.center[1] < ny / 2)
        {
            T posX = h * (plt.center[0] + 0.5) + LBLocation[0];
            T posY = h * (plt.center[1] + 0.5) + LBLocation[1];
            T posZ = h * (plt.center[2] + 0.5) + LBLocation[2];
            T distaceFromCenter = sqrt(posX * posX + posY * posY);
            if (radiusReal - distaceFromCenter < 1.5e-5)
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

        plt.nnHistory[currentTime % 129] = output;

        calciumConc = 0.5 * (mapCalc(output) + mapCalc(plt.nnHistory[(currentTime - 1) % 129]));

        // Riemann sum for integral of calc conc
        plt.calcInteg = plt.calcInteg + (calciumConc - 1.e-4) * dtNN;

        T recentCalcInteg = obtainRecentCalcInteg(plt, currentTime);
        recentCalcInteg = plt.calcInteg - recentCalcInteg;

        plt.calcIntegHistory[currentTime % 31] = plt.calcInteg;

        // Determination of activation state based on calc. integ.
        hillFunc = pow(plt.calcInteg, nHill);
        hillFunc = hillFunc / (pow(xi50, nHill) + hillFunc);
        plt.activation = alpha + (1 - alpha) * hillFunc;

        // Determination of recent history activation state of the platelet
        hillFunc = pow(recentCalcInteg, 2.);
        hillFunc = hillFunc / (pow(0.005, 2.) + hillFunc);
        plt.activationx = 2. + 98. * hillFunc;

        recentCalcInteg = 0.;
        for (plint j = 0; j < 128; ++j)
        {
            calciumConc = mapCalc(obtainHistory(plt, currentTime - j));
            recentCalcInteg = recentCalcInteg + (calciumConc - 1.e-4);
        }

        // Set release times if platelet is more than half-activated
        if (recentCalcInteg >= xi50 && plt.releaseTime < 0.)
        {
            plt.releaseTime = totalTime;
        }

        plateletList[current] = plt;
    }
}

/************************** Lattice Kinetic MC routines ********************************/

// This function generates the 3D lattice and grid points for performing the simulation
void generateKMCLattice(plint level)
{
    factor = util::roundToInt(dx / h);
    h = dx / (T)factor;
    diaFactor = util::roundToInt(pltDia / h);

    KMCLocation = LBLocation;
    KMCLocation = KMCLocation + Array<T, 3>(0.5 * h, 0.5 * h, 0.5 * h);

    nx = nxlb * factor;
    ny = nylb * factor;
    nz = nzlb * factor;
}

// Initialize platelet with parameters so that garbage values are not carried over by struct variables
void initializePlatelet(platelet &plt)
{
    plateletList.push_back(plt);
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

// Generate random number between 0 and 1
// Parallel synchronized random number generation
T randn_parallel()
{
    return distribution(engine_parallel);
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
        tau = totalTimeNRM - log(r) / rateNew;
    }
    else if (rateOld >= 1.e-10 && rateNew >= 1.e-10)
    {
        tau = totalTimeNRM + (rateOld / rateNew) * (tau - totalTimeNRM);
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
    return distanceBetween(a, b) <= diaFactor * diaFactor && distanceBetween(a, b) >= diaFactor * diaFactor / 2;
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
                    platelet tempPlt;
                    tempPlt.center[0] = i;
                    tempPlt.center[1] = j;
                    tempPlt.center[2] = k;

                    if (liesInside(tempPlt.center, plt))
                    {
                        auto count = occupation.count((long long int)ny * nz * i + (long long int)nz * j + (long long int)k);
                        if (count > 0)
                        {
                            if (checkOverlap(tempPlt, plt))
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

list<plint> checkOverlap(plint current, plint dir)
{
    platelet plt = plateletList[current];

    plint px, py, pz;
    px = plt.center[0];
    py = plt.center[1];
    pz = plt.center[2];

    plt.center[dir / 2] = plt.center[dir / 2] + (plint)pow(-1, dir);

    plint cx, cy, cz;
    cx = plt.center[0];
    cy = plt.center[1];
    cz = plt.center[2];

    T dia = plt.dp;
    plint n;
    n = ceil(dia / h);

    plint i, j, k;
    Array<plint, 3> pos;

    list<plint> overlapLinks;

    for (i = cx - n; i <= cx + n; ++i)
    {
        for (j = cy - n; j <= cy + n; ++j)
        {
            for (k = cz - n; k <= cz + n; ++k)
            {
                if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz)
                {
                    platelet tempPlt;
                    tempPlt.center[0] = i;
                    tempPlt.center[1] = j;
                    tempPlt.center[2] = k;

                    if (liesInside(tempPlt.center, plt))
                    {
                        auto count = occupation.count((long long int)ny * nz * i + (long long int)nz * j + (long long int)k);
                        if (count > 0)
                        {
                            if (i != px || j != py || k != pz)
                            {
                                if (checkOverlap(tempPlt, plt))
                                {
                                    auto range = occupation.equal_range((long long int)ny * nz * i + (long long int)nz * j + (long long int)k);
                                    for (auto iter = range.first; iter != range.second; ++iter)
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

list<T> checkOverlapSimple(plint current, plint dir)
{
    platelet plt = plateletList[current];

    plint px, py, pz;
    px = plt.center[0];
    py = plt.center[1];
    pz = plt.center[2];

    plt.center[dir / 2] = plt.center[dir / 2] + (plint)pow(-1, dir);

    plint cx, cy, cz;
    cx = plt.center[0];
    cy = plt.center[1];
    cz = plt.center[2];

    T dia = plt.dp;
    plint n;
    n = ceil(dia / h);

    plint i, j, k;
    Array<plint, 3> pos;

    list<T> overlapLinks;

    for (i = cx - n; i <= cx + n; ++i)
    {
        for (j = cy - n; j <= cy + n; ++j)
        {
            for (k = cz - n; k <= cz + n; ++k)
            {
                if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz)
                {
                    platelet tempPlt;
                    tempPlt.center[0] = i;
                    tempPlt.center[1] = j;
                    tempPlt.center[2] = k;

                    if (liesInside(tempPlt.center, plt))
                    {
                        auto iter = occupationGlobal.find((long long int)ny * nz * i + (long long int)nz * j + (long long int)k);
                        if (iter != occupationGlobal.end())
                        {
                            if (i != px || j != py || k != pz)
                            {
                                // if (checkOverlap(tempPlt, plt))
                                if (checkOverlap(tempPlt, plt) && iter->second > 100.)
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
list<T> checkOverlapBinding(plint current)
{
    platelet plt = plateletList[current];
    plint cx, cy, cz;
    cx = plt.center[0];
    cy = plt.center[1];
    cz = plt.center[2];

    T dia = plt.dp;
    plint n;
    n = ceil(dia / h);

    plint i, j, k;
    Array<plint, 3> pos;

    list<T> overlapLinks;

    for (i = cx - n; i <= cx + n; ++i)
    {
        for (j = cy - n; j <= cy + n; ++j)
        {
            for (k = cz - n; k <= cz + n; ++k)
            {
                if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz)
                {
                    platelet tempPlt;
                    tempPlt.center[0] = i;
                    tempPlt.center[1] = j;
                    tempPlt.center[2] = k;

                    auto iterlocal = occupation.find((long long int)ny * nz * i + (long long int)nz * j + (long long int)k);

                    if (iterlocal != occupation.end())
                    {
                        if (iterlocal->second >= 0 && iterlocal->second < plateletList.size())
                        {
                            if (i != plt.center[0] || j != plt.center[1] || k != plt.center[2])
                            {
                                tempPlt = plateletList[iterlocal->second];

                                if (checkOverlapBinding(tempPlt, plt))
                                {
                                    T val = tempPlt.bound ? (100. * tempPlt.layer + tempPlt.activation * tempPlt.activationx) : 0.;
                                    overlapLinks.push_back(val);
                                    if (tempPlt.bound)
                                    {
                                        continue; // Avoids double counting of local and global platelet lists
                                    }
                                }
                            }
                        }
                    }

                    auto iter = occupationGlobal.find((long long int)ny * nz * i + (long long int)nz * j + (long long int)k);

                    if (iter != occupationGlobal.end())
                    {
                        if (i != plt.center[0] || j != plt.center[1] || k != plt.center[2])
                        {
                            if (checkOverlapBinding(tempPlt, plt))
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

// This function is used to assign the platelet drift velocity given the position of the platelet center
inline void obtainDriftVelocity(Array<T, 3> pos)
{
    return;
    Array<T, 3> realPos(dx * pos + LBLocation);
    T x = realPos[0];
    T y = realPos[1];
    x = x / radiusReal;
    y = y / radiusReal;
    T r = sqrt(x * x + y * y);
    T delta = 1. / (T)(driftFunction.size() - 1);
    plint i = (plint)(r / delta);

    if (i >= driftFunction.size())
    {
        driftVelocity[0] = driftVelocity[1] = driftVelocity[2] = 0.;
        return;
    }

    driftVelocity[0] = x * driftFunction[i];
    driftVelocity[1] = y * driftFunction[i];
}

Array<T, 3> readVelocity(plint px, plint py, plint pz)
{
    MPI_Status stat;
    MPI_Offset offset;
    plint loc = pz + py * nzlb + px * nzlb * nylb;
    offset = loc * sizeof(T) * 3;
    Array<T, 3> u(0., 0., 0.);
    MPI_File_read_at(filevel, offset, &u, 3, MPI_DOUBLE, &stat);
    return u;
}

void interpolateVelocity(T px, T py, T pz)
{
    Dot3D referencePos((plint)px, (plint)py, (plint)pz);
    velocity.resetToZero();
    plint x0, y0, z0, x1, y1, z1;
    x0 = (plint)px;
    y0 = (plint)py;
    z0 = (plint)pz;
    plint loc = z0 + y0 * nzlb + x0 * nzlb * nylb;

    x1 = x0 + 1;
    y1 = y0 + 1;
    z1 = z0 + 1;
    T xd, yd, zd;
    xd = px - (T)x0;
    yd = py - (T)y0;
    zd = pz - (T)z0;

    auto v000 = readVelocity(x0, y0, z0);
    auto v001 = readVelocity(x0, y0, z1);
    auto v010 = readVelocity(x0, y1, z0);
    auto v011 = readVelocity(x0, y1, z1);
    auto v100 = readVelocity(x1, y0, z0);
    auto v101 = readVelocity(x1, y0, z1);
    auto v110 = readVelocity(x1, y1, z0);
    auto v111 = readVelocity(x1, y1, z1);

    auto v00 = (1. - xd) * v000 + xd * v100;
    auto v01 = (1. - xd) * v001 + xd * v101;
    auto v10 = (1. - xd) * v010 + xd * v110;
    auto v11 = (1. - xd) * v011 + xd * v111;

    auto v0 = (1. - yd) * v00 + yd * v10;
    auto v1 = (1. - yd) * v01 + yd * v11;

    velocity = (1. - zd) * v0 + zd * v1;
    velocity = (dx / dt) * velocity;
}

// This function is used to assign the velocity given the position of the platelet center
inline void obtainVelocity(plint cx, plint cy, plint cz)
{
    getvelocitytime -= MPI_Wtime();
    T px, py, pz;
    T f = (T)factor;
    px = ((T)cx) / f;
    py = ((T)cy) / f;
    pz = ((T)cz) / f;
    px = px - 0.25;
    py = py - 0.25;
    pz = pz - 0.25;

    interpolateVelocity(px, py, pz);

    Array<T, 3> pos((T)px, (T)py, (T)pz);

    // Superimpose platelet drift velocity on the velocity field
    velocity = velocity + driftVelocity;

    getvelocitytime += MPI_Wtime();
}

// Checks whether a platelet has left the domain
bool checkWithinDomain(plint cx, plint cy, plint cz)
{
    checkdomaintime -= MPI_Wtime();

    if (cx < 0 || cx >= nx || cy < 0 || cy >= ny || cz < 0 || cz >= nz)
    {
        checkdomaintime += MPI_Wtime();
        return false;
    }

    cx = cx / factor;
    cy = cy / factor;
    cz = cz / factor;

    if (flagMatrixLKMC.get(cx, cy, cz) == 0)
    {
        checkdomaintime += MPI_Wtime();
        return false;
    }

    checkdomaintime += MPI_Wtime();
    return true;
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

    if (!checkWithinDomain(i, j - 1, k) || !checkWithinDomain(i, j + 1, k) || !checkWithinDomain(i - 1, j, k) || !checkWithinDomain(i + 1, j, k))
    {
        return true;
    }

    return false;
}

// This function ensures that all the rates are defined such that
// a move where a platelet leaves the domain is not allowed
void checkWithinDomain(plint current)
{
    platelet plt = plateletList[current];

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
    if (!checkWithinDomain(i, j, k + 1) && k / factor < nzlb - 3 - margin)
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

    plateletList[current] = plt;
}

// This function implements the pass forward algorithm (Flamm et al. J Chem Phys 2009, 2011)
// Resets convective rates for a given direction if motion is blocked by another platelet
void passForward(plint current, plint dir)
{
    list<plint> overlapLinks;

    overlapLinks = checkOverlap(current, dir);

    platelet currentPlt = plateletList[current];

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

            T motionRate = diffusion + currentPlt.convection[dir];

            if (oldRate != motionRate)
            {
                if (!flag)
                {
                    eventList.erase(event(current, dir, currentPlt.tau[dir]));
                    currentPlt.tau[dir] = dblmax;
                }

                if (motionRate > 1.e-10)
                {
                    T r = randn();
                    T dtau = totalTimeNRM - log(r) / motionRate;
                    eventList.insert(event(current, dir, dtau));
                    currentPlt.tau[dir] = dtau;
                }
            }

            plateletList[current] = currentPlt;
            checkWithinDomain(current);
        }
        else
        {
            plateletList[current] = currentPlt;
        }

        return;
    }

    platelet tempPlt;

    plint count = 0;
    for (auto tempHead = overlapLinks.begin(); tempHead != overlapLinks.end(); ++tempHead)
    {
        count++;
    }

    T motionTransfer = currentPlt.convection[dir] / (T)count;

    plint cx, cy, cz;

    for (auto tempHead = overlapLinks.begin(); tempHead != overlapLinks.end(); ++tempHead)
    {
        plint temp = *tempHead;
        tempPlt = plateletList[temp];

        cx = tempPlt.center[0];
        cy = tempPlt.center[1];
        cz = tempPlt.center[2];

        obtainVelocity(cx, cy, cz); // Get velocity from LB

        tempPlt.convection[dir] = max(0., pow(-1, dir) * velocity[dir / 2] / h) + motionTransfer;
        plateletList[temp] = tempPlt;

        if (!tempPlt.bound)
        {
            passForward(temp, dir);
        }
    }

    currentPlt.blockCount[dir] = count;
    eventList.erase(event(current, dir, currentPlt.tau[dir]));
    currentPlt.tau[dir] = dblmax;

    plateletList[current] = currentPlt;

    return;
}

// This function finds if any platelet was blocked from moving by the platelet which just moved.
// If there exists such a platelet, then it resets the motion of the platelet from 0 to the apt value
// Furthermore, due to the motion of platelets, there may be the possibility that this platelet gets blocked
// It implements the pass forward algorithm in these cases.
void passForwardMotion(plint current, plint dir)
{
    plint temp;
    platelet currentPlt, tempPlt;
    currentPlt = plateletList[current];
    plint i, j;

    // Check if the platelet was blocking any platelet
    for (i = 0; i < 6; ++i)
    {
        if (i == dir)
            continue;

        // Move the platelet backwards away from the direction of motion
        currentPlt.center[dir / 2] = currentPlt.center[dir / 2] - (plint)pow(-1, dir);

        plateletList[current] = currentPlt;

        auto overlapLinks = checkOverlap(current, i);

        currentPlt.center[dir / 2] = currentPlt.center[dir / 2] + (plint)pow(-1, dir);

        plateletList[current] = currentPlt;

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

            for (auto tempHead = overlapLinks.begin(); tempHead != overlapLinks.end(); ++tempHead)
            {
                temp = *tempHead;
                tempPlt = plateletList[temp];

                tempPlt.blockCount[j] -= 1;

                plateletList[temp] = tempPlt;

                passForward(temp, j);

                tempPlt = plateletList[temp];

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
                    T motionRate = diffusion + tempPlt.convection[i];

                    if (oldRate != motionRate)
                    {
                        if (!flag)
                        {
                            eventList.erase(event(temp, i, tempPlt.tau[i]));
                            tempPlt.tau[i] = dblmax;
                        }

                        if (motionRate > 1.e-10)
                        {
                            T r = randn();
                            T dtau = totalTimeNRM - log(r) / motionRate;
                            eventList.insert(event(temp, i, dtau));
                            tempPlt.tau[i] = dtau;
                        }
                    }

                    plateletList[temp] = tempPlt;
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

        auto overlapLinks = checkOverlap(current, i);

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

            for (auto tempHead = overlapLinks.begin(); tempHead != overlapLinks.end(); tempHead++)
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

bool simpleBlocking(plint current, plint dir)
{
    auto overlaplinks = checkOverlapSimple(current, dir);

    if (!overlaplinks.empty())
    {
        platelet plt = plateletList[current];
        if (plt.motion[dir] > 1.e-10)
        {
            T r = randn();
            T dtau = totalTimeNRM - log(r) / plt.motion[dir];
            eventList.insert(event(current, dir, dtau));
            plt.tau[dir] = dtau;
        }
        plateletList[current] = plt;

        return true;
    }
    return false;
}

// To do: Parallelize the computation of flow rate and normalizations

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
    T x, y;
    T flowRate = 0.;
    plint i, j;
    T concPlt = 1.5e14;

    i = 0;
    T ds = concPlt * radiusReal * radiusReal * delta * delta / normalization;
    // Area element * normalized concentration

    std::set<inletEvent, orderInletEvent> tempInletEventList;
    inletEventList.swap(tempInletEventList);

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

            Array<T, 3> realPos(x * radiusEffectivex, y * radiusEffectivey, dx);

            realPos[0] = minx + (realPos[0] + radiusEffectivex - 0.5 * h) * (maxx - minx) / (2 * radiusEffectivex - h);
            realPos[1] = miny + (realPos[1] + radiusEffectivey - 0.5 * h) * (maxy - miny) / (2 * radiusEffectivey - h);

            obtainVelocity(util::roundToInt(realPos[0]), util::roundToInt(realPos[1]), nz / 8);

            // Temporary code to compute volumetrix flow rate
            volFlowRate = volFlowRate + velocity[2] * radiusReal * radiusReal * delta * delta;

            /*
            if (sqrt(realPos[0] * realPos[0] + realPos[1] * realPos[1]) < radiusReal - 1.5e-5)
            {
                continue;
            }
            */

            T rate = velocity[2] * pdf[i] * ds; // Imposing a platelet conc. profile

            T dtau = -log(randn_inlet()) / rate;

            inletEventList.insert(inletEvent(x, y, dtau));

            flowRate = flowRate + rate;
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

// Computes the inlet flow rate at given coordinates x,y
void computeFlowRate(T x, T y)
{
    plint n = pdf.size();
    T delta = 1. / (T)(n - 1);
    T concPlt = 1.5e14;

    T ds = concPlt * radiusReal * radiusReal * delta * delta / normalization;
    // Area element * normalized concentration

    T r = sqrt(x * x + y * y);

    plint i = (plint)(r / delta);

    Array<T, 3> realPos(x * radiusEffectivex, y * radiusEffectivey, dx);

    realPos[0] = minx + (realPos[0] + radiusEffectivex - 0.5 * h) * (maxx - minx) / (2 * radiusEffectivex - h);
    realPos[1] = miny + (realPos[1] + radiusEffectivey - 0.5 * h) * (maxy - miny) / (2 * radiusEffectivey - h);

    obtainVelocity(util::roundToInt(realPos[0]), util::roundToInt(realPos[1]), nz / 8);

    T rate = velocity[2] * pdf[i] * ds; // Imposing a platelet conc. profile

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
    cout << "Binary search for inlet position failed" << endl;
    MPI_Abort(MPI_COMM_WORLD, -1);
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
    plt.center[2] = margin + (plint)(randn_inlet() * (nz - margin - 1));

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
bool addPlateletInlet()
{
    platelet plt;

    inletEvent inEvent = *(inletEventList.begin());
    T x = inEvent.x;
    T y = inEvent.y;
    T r = sqrt(x * x + y * y);

    computeFlowRate(x, y);

    Array<T, 3> realPos(x * radiusEffectivex, y * radiusEffectivey, dx);

    realPos[0] = minx + (realPos[0] + radiusEffectivex - 0.5 * h) * (maxx - minx) / (2 * radiusEffectivex - h);
    realPos[1] = miny + (realPos[1] + radiusEffectivey - 0.5 * h) * (maxy - miny) / (2 * radiusEffectivey - h);

    plt.center[0] = util::roundToInt(realPos[0]);
    plt.center[1] = util::roundToInt(realPos[1]);
    plt.center[2] = nz / 8; // For now, it is assumed that the inlet plane is always normal to the z-axis

    plt.dp = pltDia;

    if (plt.center[0] <= 0 || plt.center[0] >= nx - 1 || plt.center[1] <= 0 || plt.center[1] >= ny - 1 || plt.center[2] <= 0 || plt.center[2] >= nz - 1)
    {
        return false;
    }

    if (checkOverlap(plt) || !checkWithinDomain(plt.center[0], plt.center[1], plt.center[2]))
    {
        return false;
    }
    else
    {
        nPlatelets++;
        initializePlatelet(plt);
        return true;
    }
}

// Computes the enhancement in binding rates due to vwf stretching
T vwfFactor(plint current)
{
    T enhancement;

    platelet plt = plateletList[current];
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
    plateletList[current] = plt;
    return enhancement;
}

// Computes the enhancement in detachment rate due to shear
T detachFactor(plint current)
{
    T enhancement;
    platelet plt = plateletList[current];
    plt.localShear = computePltShear(current);
    if (plt.localShear <= 1000.)
    {
        enhancement = exp(plt.localShear / charShear);
    }
    else
    {
        enhancement = exp(5.) * exp(plt.localShear / 20000.);
    }

    plateletList[current] = plt;
    return enhancement;
}

bool isBindingAllowed(platelet &plt)
{
    auto loc = (long long int)ny * nz * plt.center[0] +
               (long long int)nz * plt.center[1] + (long long int)plt.center[2];

    auto iT = occupationGlobal.find(loc - ny * nz);
    if (iT != occupationGlobal.end())
    {
        if (iT->second > 100.)
        {
            return false;
        }
    }

    iT = occupationGlobal.find(loc + ny * nz);
    if (iT != occupationGlobal.end())
    {
        if (iT->second > 100.)
        {
            return false;
        }
    }

    iT = occupationGlobal.find(loc - nz);
    if (iT != occupationGlobal.end())
    {
        if (iT->second > 100.)
        {
            return false;
        }
    }

    iT = occupationGlobal.find(loc + nz);
    if (iT != occupationGlobal.end())
    {
        if (iT->second > 100.)
        {
            return false;
        }
    }

    iT = occupationGlobal.find(loc - 1);
    if (iT != occupationGlobal.end())
    {
        if (iT->second > 100.)
        {
            return false;
        }
    }

    iT = occupationGlobal.find(loc + 1);
    if (iT != occupationGlobal.end())
    {
        if (iT->second > 100.)
        {
            return false;
        }
    }

    return true;
}

// This functional is used to set the binding rates associated with
// the given platelet as the argument
void setBindingRates(plint current)
{
    platelet plt = plateletList[current];
    platelet tempPlt;

    long layer = intmax / 2;

    T rate = 0.;

    T enhancement = vwfFactor(current);

    // Presence of the reactive patch
    // This is subject to change with a change of domain

    if (checkCollagen(plt.center[0], plt.center[1], plt.center[2]))
    {
        // rate = rate + k_att_col * plt.activation * enhancement;
        rate = rate + k_att_col * plt.activation * plt.activationx * enhancement;
        layer = 1;
    }

    // Move the platelet in any given direction to see if there's overlap with another platelet
    // If there is an overlap and the overlapping plt is connected to coll., then a binding event is possible

    auto overlapLinks = checkOverlapBinding(current);

    if (!overlapLinks.empty())
    {
        for (auto temp = overlapLinks.begin(); temp != overlapLinks.end(); ++temp)
        {
            T tempActivation = *temp;

            if (tempActivation > 100.)
            {
                long tempLayer = (long)(tempActivation / 100);

                if (layer > tempLayer + 1)
                {
                    layer = tempLayer + 1;
                }
                tempActivation = tempActivation - 100. * tempLayer;
                rate = rate + k_att_fib * sqrt(plt.activation * plt.activationx * tempActivation) * enhancement;
            }
        }
    }

    if (!isBindingAllowed(plt))
    {
        rate = 0.;
    }

    T dtau = dblmax;
    if (rate > 1.e-10)
    {
        T r = randn();
        dtau = totalTimeNRM - log(r) / rate;
        eventList.insert(event(current, 6, dtau));
    }

    plt.binding = rate;
    plt.layer = layer;

    auto iT = eventList.find(event(current, 6, plt.tau[6]));
    if (iT != eventList.end())
    {
        eventList.erase(iT);
    }
    plt.tau[6] = dtau;
    plateletList[current] = plt;
}

// This functional is used to set the unbinding rates associated with
// the given platelet as the argument
void setUnbindingRates(plint current)
{
    platelet plt = plateletList[current];

    auto iT = eventList.find(event(current, 7, plt.tau[7]));
    if (iT != eventList.end())
    {
        eventList.erase(iT);
    }

    T rate = dblmax;
    T tempRate = 0.;

    plint ind, dir, j, type;

    long minLayer = intmax / 2;
    plint lowercount = 0;

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

        minLayer = 1;
        ++lowercount;
    }

    auto overlapLinks = checkOverlapBinding(current);

    if (!overlapLinks.empty())
    {
        // Check for neighboring bound platelets
        for (auto temp = overlapLinks.begin(); temp != overlapLinks.end(); ++temp)
        {
            T tempActivation = *temp;

            if (tempActivation > 100.)
            {
                long tempLayer = (long)(tempActivation / 100);

                if (minLayer > tempLayer)
                {
                    minLayer = tempLayer;
                }
                if (plt.layer > tempLayer)
                {
                    ++lowercount;
                }
                tempActivation = tempActivation - 100. * tempLayer;

                tempRate = k_det_fib * enhancement / (sqrt(plt.activation * plt.activationx * tempActivation));

                if (rate > tempRate)
                {
                    rate = tempRate;
                }
            }
        }
    }

    if (lowercount == 0 || plt.layer > minLayer + 1)
    {
        rate = dblmax;
    }

    T r, dtau;

    dtau = plt.tau[7];
    dtau = updateEventTime(dtau, plt.unbinding, rate);

    if (rate > 0.5 * dblmax)
    {
        dtau = totalTimeNRM;
    }

    eventList.insert(event(current, 7, dtau));

    plt.tau[7] = dtau;
    plt.unbinding = rate;
    plateletList[current] = plt;
}

plint nout = 0;
// This function sets the rate of each event for a given platelet that moves
// More efficient to use this when the velocity field does not change
void setRates(plint current, plint type)
{
    platelet plt = plateletList[current];

    plint ind, j;

    // If a motion event is executed, then apply the pass-forward algorithm accordingly
    // Check for potential binding events and assign their rates

    if (type < 6)
    {
        // Parallel routine needs the sum of all rate processes
        for (j = 0; j < 6; ++j)
        {
            totalRate = totalRate - plt.motion[j];
        }
        totalRate = totalRate - plt.binding;

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
            plateletList[current] = plt;
            plt = plateletList[current];

            if (plt.motion[j] > 1.e-10)
            {
                T r = randn();
                T dtau = totalTimeNRM - log(r) / plt.motion[j];
                eventList.insert(event(current, j, dtau));
                plt.tau[j] = dtau;
            }
        }

        plt.lastMotionTime = totalTime;

        plateletList[current] = plt;

        checkWithinDomain(current);

        setBindingRates(current);

        plt = plateletList[current];

        // Parallel routine needs the sum of all rate processes
        for (j = 0; j < 6; ++j)
        {
            totalRate = totalRate + plt.motion[j];
        }
        totalRate = totalRate + plt.binding;

        // passForwardMotion(current, type);
    }

    // If a binding event is executed, then set the binding rate to zero, because it cannot happen again
    // Set the unbinding rate accordingly
    // motion needs to be set to zero

    if (type == 6)
    {
        auto iT = eventList.find(event(current, type, plt.tau[type]));
        if (iT != eventList.end())
        {
            eventList.erase(iT);
        }
        plt.tau[type] = dblmax;

        // Parallel routine needs the sum of all rate processes
        for (j = 0; j < 6; ++j)
        {
            totalRate = totalRate - plt.motion[j];
        }
        totalRate = totalRate - plt.binding;
        plt.binding = 0.;

        for (j = 0; j < 6; ++j)
        {
            plt.motion[j] = 0.;
            eventList.erase(event(current, j, plt.tau[j]));
            plt.tau[j] = dblmax;
        }

        plt.bound = true;

        plateletList[current] = plt;
        setUnbindingRates(current);

        plt = plateletList[current];
        // Parallel routine needs the sum of all rate processes
        if (plt.unbinding < 0.5 * dblmax)
        {
            totalRate = totalRate + plt.unbinding;
        }
    }

    // If an unbinding event is executed, set the unbinding rate to zero
    // because the bond cannot be broken again
    // Set binding rates accordingly
    // If the platelet becomes free to move, set motion rates.
    if (type == 7)
    {
        ++nUnbindingOccurred;

        if (plt.unbinding < 0.5 * dblmax)
        {
            totalRate = totalRate - plt.unbinding;
        }

        plt.unbinding = 0.;
        auto iT = eventList.find(event(current, type, plt.tau[type]));
        if (iT != eventList.end())
        {
            eventList.erase(iT);
        }
        plt.tau[type] = dblmax;
        plt.bound = false;

        obtainVelocity(plt.center[0], plt.center[1], plt.center[2]); // Get velocity from LB
        for (j = 0; j < 6; ++j)
        {
            plt.convection[j] = max(0., pow(-1, j) * velocity[j / 2]) / h;
            plt.motion[j] = diffusion + plt.convection[j];

            plt.tau[j] = dblmax;
            plateletList[current] = plt;
            plt = plateletList[current];

            if (plt.motion[j] > 1.e-10)
            {
                T r = randn();
                T dtau = totalTimeNRM - log(r) / plt.motion[j];
                eventList.insert(event(current, j, dtau));
                plt.tau[j] = dtau;
            }
        }

        plt.lastMotionTime = totalTime;

        plateletList[current] = plt;

        checkWithinDomain(current);

        setBindingRates(current);

        plt = plateletList[current];
        // Parallel routine needs the sum of all rate processes
        for (j = 0; j < 6; ++j)
        {
            totalRate = totalRate + plt.motion[j];
        }
        totalRate = totalRate + plt.binding;

        /*
        for (j = 0; j < 6; ++j)
        {
            passForward(current, j);
        }
        */
    }

    // Platelet being added to the domain
    if (type >= 8)
    {
        if (!plt.bound)
        {
            obtainVelocity(plt.center[0], plt.center[1], plt.center[2]); // Get velocity from LB

            for (j = 0; j < 6; ++j)
            {
                plt.convection[j] = max(0., pow(-1, j) * velocity[j / 2]) / h;
                plt.motion[j] = diffusion + plt.convection[j];
                plt.tau[j] = dblmax;
                plateletList[current] = plt;
                plt = plateletList[current];

                if (plt.motion[j] > 1.e-10)
                {
                    T r = randn();
                    T dtau = totalTimeNRM - log(r) / plt.motion[j];
                    eventList.insert(event(current, j, dtau));
                    plt.tau[j] = dtau;
                }
            }

            plateletList[current] = plt;

            checkWithinDomain(current);

            setBindingRates(current);

            plt = plateletList[current];
            // Parallel routine needs the sum of all rate processes
            for (j = 0; j < 6; ++j)
            {
                totalRate = totalRate + plt.motion[j];
            }
            totalRate = totalRate + plt.binding;

            if (plt.unbinding < 0.5 * dblmax)
            {
                totalRate = totalRate + plt.unbinding;
            }

            std::pair<long long int, plint> pltPair(
                (long long int)ny * nz * plt.center[0] + (long long int)nz * plt.center[1] + (long long int)plt.center[2], current);
            occupation.insert(pltPair);

            // passForwardMotion(current, 4);
            //  SInce platelet enters domain through motion in the z-direction
            //  passforwardmotion needs to be applied only in the +z direction
        }
    }

    // Remove platelet from list if it has left the domain
    plt = plateletList[current];
    if (!checkWithinDomain(plt.center[0], plt.center[1], plt.center[2]) || plt.center[2] >= 7 * nz / 8)
    {
        // Parallel routine needs the sum of all rate processes
        for (j = 0; j < 6; ++j)
        {
            totalRate = totalRate - plt.motion[j];
        }
        totalRate = totalRate - plt.binding;

        if (plt.unbinding < 0.5 * dblmax)
        {
            totalRate = totalRate - plt.unbinding;
        }

        plint count = 0;
        for (j = 0; j < 6; ++j)
        {
            auto iT = eventList.find(event(current, j, plt.tau[j]));
            if (iT != eventList.end())
            {
                eventList.erase(iT);
                count++;
            }
        }

        auto range = occupation.equal_range((long long int)ny * nz * plt.center[0] +
                                            (long long int)nz * plt.center[1] + (long long int)plt.center[2]);
        for (auto iT = range.first; iT != range.second; ++iT)
        {
            if (iT->second == current)
            {
                occupation.erase(iT);
                break;
            }
        }
    }
}

// This function sets the rate of each event for each platelet
// Alters the rate parameters where the move is not allowed
// by calling the convective pass-forward algorithm routine
void setRates()
{
    plint current;

    platelet plt;

    plint ind, j;

    totalTimeNRM = totalTime;

    nBoundPlatelets = 0;
    nPlatelets = 0;

    // Parallel routine needs the sum of all rate processes
    totalRate = 0.;

    for (current = 0; current < plateletList.size(); ++current)
    {
        plt = plateletList[current];

        if (!plt.bound)
        {
            obtainVelocity(plt.center[0], plt.center[1], plt.center[2]); // Get velocity from LB

            for (j = 0; j < 6; ++j)
            {
                plt.convection[j] = max(0., pow(-1, j) * velocity[j / 2]) / h;
                plt.motion[j] = diffusion + plt.convection[j];
                T dtau = totalTimeNRM - log(randn()) / plt.motion[j];
                plt.tau[j] = dtau;
            }

            plateletList[current] = plt;
            checkWithinDomain(current);
        }
        else
        {
            nBoundPlatelets = nBoundPlatelets + 1;
        }
        nPlatelets = nPlatelets + 1;
    }

    // Set binding and unbinding rates based on updated platelet activations and shear rates
    for (current = 0; current < plateletList.size(); ++current)
    {
        plt = plateletList[current];
        if (!plt.bound)
        {
            setBindingRates(current);
        }
        else
        {
            setUnbindingRates(current);
        }
    }

    // Parallel routine needs the sum of all rate processes
    for (current = 0; current < plateletList.size(); ++current)
    {
        plt = plateletList[current];
        plt.sumRates = 0.;
        for (j = 0; j < 6; ++j)
        {
            plt.sumRates = plt.sumRates + plt.motion[j];
        }
        plt.sumRates = plt.sumRates + plt.binding;

        if (plt.unbinding < 0.5 * dblmax)
        {
            plt.sumRates = plt.sumRates + plt.unbinding;
        }

        totalRate = totalRate + plt.sumRates;
        plateletList[current] = plt;
    }
}

// Creating custom MPI type to exchange info about platelets
void defineMPIStruct(MPI_Datatype &type)
{
    const int nitems = 22;
    int blocklengths[nitems] = {3, 1, 1, 1, 6, 6, 6, 1, 1, 1, 8, 1, 1, 4, 1, 1, 1, 1, 129, 31, 1, 1};
    MPI_Datatype types[nitems] = {
        MPI_LONG,
        MPI_DOUBLE,
        MPI_LONG,
        MPI_CXX_BOOL,
        MPI_DOUBLE,
        MPI_DOUBLE,
        MPI_LONG,
        MPI_DOUBLE,
        MPI_DOUBLE,
        MPI_CXX_BOOL,
        MPI_DOUBLE,
        MPI_DOUBLE,
        MPI_DOUBLE,
        MPI_DOUBLE,
        MPI_DOUBLE,
        MPI_DOUBLE,
        MPI_DOUBLE,
        MPI_DOUBLE,
        MPI_DOUBLE,
        MPI_DOUBLE,
        MPI_DOUBLE,
        MPI_LONG};
    MPI_Aint offsets[nitems];
    offsets[0] = offsetof(struct platelet, center);
    offsets[1] = offsetof(struct platelet, dp);
    offsets[2] = offsetof(struct platelet, layer);
    offsets[3] = offsetof(struct platelet, unbound);
    offsets[4] = offsetof(struct platelet, convection);
    offsets[5] = offsetof(struct platelet, motion);
    offsets[6] = offsetof(struct platelet, blockCount);
    offsets[7] = offsetof(struct platelet, binding);
    offsets[8] = offsetof(struct platelet, unbinding);
    offsets[9] = offsetof(struct platelet, bound);
    offsets[10] = offsetof(struct platelet, tau);
    offsets[11] = offsetof(struct platelet, localShear);
    offsets[12] = offsetof(struct platelet, lastMotionTime);
    offsets[13] = offsetof(struct platelet, concentrations);
    offsets[14] = offsetof(struct platelet, releaseTime);
    offsets[15] = offsetof(struct platelet, calcInteg);
    offsets[16] = offsetof(struct platelet, activation);
    offsets[17] = offsetof(struct platelet, activationx);
    offsets[18] = offsetof(struct platelet, nnHistory);
    offsets[19] = offsetof(struct platelet, calcIntegHistory);
    offsets[20] = offsetof(struct platelet, sumRates);
    offsets[21] = offsetof(struct platelet, lastMotionType);
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &type);
    MPI_Type_commit(&type);
}

bool sortByLocation(const platelet &a, const platelet &b)
{
    // return (a.center[0]*a.center[0] + a.center[1]*a.center[1] + a.center[2]*a.center[2]) < (b.center[0]*b.center[0] + b.center[1]*b.center[1] + b.center[2]*b.center[2]);

    // return a.center[2] < b.center[2];
    return a.center[1] < b.center[1];
    // return a.center[0] < b.center[0];
}

void distributePlatelets()
{
    T sumTotalRate;
    MPI_Allreduce(&totalRate, &sumTotalRate, 1, MPI_DOUBLE, MPI_SUM, world);
    T avgTotalRate = sumTotalRate / numprocs;

    vector<int> nplatelets;

    nplatelets.resize(numprocs);

    int nplatelets_local = plateletList.size();

    MPI_Allgather(&nplatelets_local, 1, MPI_INT, &nplatelets[0], 1, MPI_INT, world);

    vector<int> displs;
    displs.push_back(0);

    int nplatelets_total = nplatelets[0];
    for (int i = 1; i < numprocs; ++i)
    {
        displs.push_back(displs[i - 1] + nplatelets[i - 1]);
        nplatelets_total = nplatelets_total + nplatelets[i];
    }

    vector<platelet> netPlateletList;
    netPlateletList.resize(nplatelets_total);
    MPI_Allgatherv(&plateletList[0], nplatelets_local, mpi_platelet, &netPlateletList[0], &nplatelets[0], &displs[0], mpi_platelet, world);

    vector<platelet> tempPlateletList{};
    plateletList.swap(tempPlateletList);

    T cumulativeRate = 0.;
    int assigned_proc = 0;

    for (int i = 0; i < nplatelets_total; ++i)
    {
        if (assigned_proc == procid)
        {
            plateletList.push_back(netPlateletList[i]);
        }

        cumulativeRate = cumulativeRate + netPlateletList[i].sumRates;

        if (cumulativeRate > ((1. + assigned_proc) * avgTotalRate))
        {
            ++assigned_proc;
        }
    }

    maxTotalRate = 1.1 * (avgTotalRate + inletRate);

    T sumMotion = 0.;
    T sumBonding = 0.;
    T sumUnbounding = 0.;
    if (avgTotalRate > 1.e10)
    {
        for (int i = 0; i < netPlateletList.size(); ++i)
        {
            platelet plt = netPlateletList[i];

            for (int j = 0; j < 6; ++j)
                sumMotion += plt.motion[j];

            sumBonding += plt.binding;

            if (plt.binding > 1.e10)
            {
                cout << "Act " << plt.activation << endl;
                cout << "Act x " << plt.activationx << endl;
                cout << "Layer " << plt.layer << endl;
                MPI_Abort(MPI_COMM_WORLD, 0);
            }

            if (plt.unbinding < 0.5 * dblmax)
                sumUnbounding += plt.unbinding;
        }
    }
}

void updateEventList()
{
    std::set<event, orderEvent> tempEventList{};
    eventList.swap(tempEventList);

    // Parallel routine needs the sum of all rate processes
    totalRate = 0.;

    nPlatelets = 0;
    nBoundPlatelets = 0;

    platelet plt;

    for (int current = 0; current < plateletList.size(); ++current)
    {
        plt = plateletList[current];

        if (!plt.bound)
        {
            obtainVelocity(plt.center[0], plt.center[1], plt.center[2]); // Get velocity from LB

            for (int j = 0; j < 6; ++j)
            {
                plt.convection[j] = max(0., pow(-1, j) * velocity[j / 2]) / h;
                plt.motion[j] = diffusion + plt.convection[j];
                T dtau = totalTimeNRM - log(randn()) / plt.motion[j];
                plt.tau[j] = dtau;
                if (plt.motion[j] > 1.e-10)
                {
                    eventList.insert(event(current, j, plt.tau[j]));
                }
            }

            plateletList[current] = plt;
            checkWithinDomain(current);
        }
        else
        {
            nBoundPlatelets = nBoundPlatelets + 1;
        }
        nPlatelets = nPlatelets + 1;
    }

    // Set binding and unbinding rates based on updated platelet activations and shear rates
    for (int current = 0; current < plateletList.size(); ++current)
    {
        plt = plateletList[current];
        if (!plt.bound)
        {
            setBindingRates(current);
        }
        else
        {
            setUnbindingRates(current);
        }
    }

    // Parallel routine needs the sum of all rate processes
    for (int current = 0; current < plateletList.size(); ++current)
    {
        plt = plateletList[current];
        plt.sumRates = 0.;
        for (int j = 0; j < 6; ++j)
        {
            plt.sumRates = plt.sumRates + plt.motion[j];
        }
        plt.sumRates = plt.sumRates + plt.binding;

        if (plt.unbinding < 0.5 * dblmax)
        {
            plt.sumRates = plt.sumRates + plt.unbinding;
        }

        totalRate = totalRate + plt.sumRates;
    }

    MPI_Allreduce(&totalRate, &maxTotalRate, 1, MPI_DOUBLE, MPI_MAX, world);

    T fractionRate = exp(exp(maxTotalRate / totalRate));
    T sumFractionRate;

    MPI_Allreduce(&fractionRate, &sumFractionRate, 1, MPI_DOUBLE, MPI_SUM, world);

    inletRate = (fractionRate / sumFractionRate) * oldInletRate;

    // inletRate = oldInletRate / numprocs;
    totalRate = totalRate + inletRate;

    // Inlet rate
    T dtau = totalTimeNRM - log(randn()) / inletRate;
    inletTau = dtau;
    eventList.insert(event(-1, 8, dtau));

    // Pass forward algorithm
    /*
    for (int current = 0; current < plateletList.size(); ++current)
    {
        for (plint j = 0; j < 6; ++j)
        {
            passForward(current, j);
        }
    }
    */
}

// Removes platelets that are no longer in the domain from the platelet list
// Also frees up heap memory associated with STL allocation
void removeExitPlatelets()
{
    std::set<event, orderEvent> tempEventList{};
    eventList.swap(tempEventList);

    plint current;

    platelet plt;

    vector<platelet> tempPlateletList{};
    for (current = 0; current < plateletList.size(); ++current)
    {
        plt = plateletList[current];
        if (!checkWithinDomain(plt.center[0], plt.center[1], plt.center[2]) || plt.center[2] >= 7 * nz / 8)
        {
            continue;
        }

        tempPlateletList.push_back(plt);
    }

    plateletList.swap(tempPlateletList);
}

void iniKMCLattice()
{
    for (plint i = 0; i < nPlatelets; ++i)
    {
        addPlatelet();
    }
}

void resolveOverlaps()
{
    // std::map<long long int, plint> tempOccupation;
    std::unordered_multimap<long long int, plint> tempOccupation;
    occupation.swap(tempOccupation);

    vector<platelet> tempPlateletList{};

    for (plint current = 0; current < plateletList.size(); ++current)
    {
        platelet plt = plateletList[current];

        if (plt.unbound)
        {
            ++removeCount;
            continue;
        }

        auto loc = (long long int)ny * nz * plt.center[0] +
                   (long long int)nz * plt.center[1] + (long long int)plt.center[2];

        if (!plt.bound)
        {
            auto iT = occupationGlobal.find(loc);
            if (iT != occupationGlobal.end())
            {
                if (iT->second > 100.)
                {
                    ++removeCount;
                    continue;
                }
            }

            iT = occupationGlobal.find(loc - ny * nz);
            if (iT != occupationGlobal.end())
            {
                if (iT->second > 100.)
                {
                    ++removeCount;
                    continue;
                }
            }

            iT = occupationGlobal.find(loc + ny * nz);
            if (iT != occupationGlobal.end())
            {
                if (iT->second > 100.)
                {
                    ++removeCount;
                    continue;
                }
            }

            iT = occupationGlobal.find(loc - nz);
            if (iT != occupationGlobal.end())
            {
                if (iT->second > 100.)
                {
                    ++removeCount;
                    continue;
                }
            }

            iT = occupationGlobal.find(loc + nz);
            if (iT != occupationGlobal.end())
            {
                if (iT->second > 100.)
                {
                    ++removeCount;
                    continue;
                }
            }

            iT = occupationGlobal.find(loc - 1);
            if (iT != occupationGlobal.end())
            {
                if (iT->second > 100.)
                {
                    ++removeCount;
                    continue;
                }
            }

            iT = occupationGlobal.find(loc + 1);
            if (iT != occupationGlobal.end())
            {
                if (iT->second > 100.)
                {
                    ++removeCount;
                    continue;
                }
            }
        }

        tempPlateletList.push_back(plt);
    }

    plateletList.swap(tempPlateletList);

    for (plint current = 0; current < plateletList.size(); ++current)
    {
        platelet plt = plateletList[current];

        auto loc = (long long int)ny * nz * plt.center[0] +
                   (long long int)nz * plt.center[1] + (long long int)plt.center[2];

        std::pair<long long int, plint> pltPair(loc, current);
        occupation.insert(pltPair);
    }
}

// Update the unordered map that contains the platelet positions as a function of
// lattice spacing across all processors
void updatePlateletMap()
{
    countoverlapbound = 0;

    // std::unordered_map<long long int, T> tempOccupation;
    std::map<long long int, T> tempOccupationGlobal;
    occupationGlobal.swap(tempOccupationGlobal);

    std::vector<long long int> sendLocations;
    std::vector<T> sendActivations;
    int send_size;

    for (plint j = 0; j < numprocs; ++j)
    {
        if (procid == j)
        {
            mapToVec(occupationGlobal, sendLocations, sendActivations);
            send_size = sendLocations.size();
        }
        MPI_Bcast(&send_size, 1, MPI_INT, j, world);
        sendLocations.resize(send_size);
        sendActivations.resize(send_size);
        MPI_Bcast(&sendLocations[0], send_size, MPI_LONG_LONG_INT, j, world);
        MPI_Bcast(&sendActivations[0], send_size, MPI_DOUBLE, j, world);
        vecToMap(occupationGlobal, sendLocations, sendActivations);
    }

    resolveOverlaps();
}

// Obtain the keys and values in a given unordered_map as two vectors
template <typename T1, typename T2>
// void mapToVec(unordered_map<T1,T2>& umap, vector<T1> & vec_keys, vector<T2> & vec_vals)
void mapToVec(map<T1, T2> &umap, vector<T1> &vec_keys, vector<T2> &vec_vals)
{
    vec_keys.clear();
    vec_vals.clear();
    plint current;
    platelet plt;
    // unordered_map<long long int, T>::iterator iter;
    map<long long int, T>::iterator iter;

    for (current = 0; current < plateletList.size(); ++current)
    {
        plt = plateletList[current];

        if (plt.unbound)
        {
            continue;
        }

        auto loc = (long long int)ny * nz * plt.center[0] +
                   (long long int)nz * plt.center[1] + (long long int)plt.center[2];

        T val;

        if (plt.bound)
        {
            val = 100. * plt.layer + plt.activation * plt.activationx;
        }
        else
        {
            val = plt.activation * plt.activationx;
        }

        auto iT = umap.find(loc);

        if (plt.bound && iT != umap.end())
        {
            if (iT->second > 100.)
                ++countoverlapbound;
        }

        // Checking for bound overlapped
        auto loc_ = (long long int)ny * nz * (plt.center[0] - 1) +
                    (long long int)nz * plt.center[1] + (long long int)plt.center[2];
        auto iT_ = umap.find(loc_);

        if (plt.bound && iT_ != umap.end())
        {
            if (iT_->second > 100.)
                ++countoverlapbound;
        }

        loc_ = (long long int)ny * nz * (plt.center[0] + 1) +
               (long long int)nz * plt.center[1] + (long long int)plt.center[2];
        iT_ = umap.find(loc_);

        if (plt.bound && iT_ != umap.end())
        {
            if (iT_->second > 100.)
                ++countoverlapbound;
        }

        loc_ = (long long int)ny * nz * plt.center[0] +
               (long long int)nz * (plt.center[1] - 1) + (long long int)plt.center[2];
        iT_ = umap.find(loc_);

        if (plt.bound && iT_ != umap.end())
        {
            if (iT_->second > 100.)
                ++countoverlapbound;
        }

        loc_ = (long long int)ny * nz * plt.center[0] +
               (long long int)nz * (plt.center[1] + 1) + (long long int)plt.center[2];
        iT_ = umap.find(loc_);

        if (plt.bound && iT_ != umap.end())
        {
            if (iT_->second > 100.)
                ++countoverlapbound;
        }

        loc_ = (long long int)ny * nz * plt.center[0] +
               (long long int)nz * plt.center[1] + (long long int)(plt.center[2] - 1);
        iT_ = umap.find(loc_);

        if (plt.bound && iT_ != umap.end())
        {
            if (iT_->second > 100.)
                ++countoverlapbound;
        }

        loc_ = (long long int)ny * nz * plt.center[0] +
               (long long int)nz * plt.center[1] + (long long int)(plt.center[2] + 1);
        iT_ = umap.find(loc_);

        if (plt.bound && iT_ != umap.end())
        {
            if (iT_->second > 100.)
                ++countoverlapbound;
        }

        if (!plt.bound)
        {
            if (iT != umap.end())
            {
                plt.unbound = true;
                plateletList[current] = plt;
            }
            else
            {
                umap[loc] = val;
                vec_keys.push_back(loc);
                vec_vals.push_back(val);
            }
        }
        else
        {
            if (iT != umap.end())
            {
                if (iT->second > 100.)
                {
                    plt.unbound = true;
                    plateletList[current] = plt;
                }
                else
                {
                    umap[loc] = val;
                    vec_keys.push_back(loc);
                    vec_vals.push_back(val);
                }
            }
            else
            {
                umap[loc] = val;
                vec_keys.push_back(loc);
                vec_vals.push_back(val);
            }
        }
    }
}

// Obtain the keys and values in a given unordered_map as two vectors
template <typename T1, typename T2>
// void vecToMap(unordered_map<T1,T2>& umap, vector<T1> & vec_keys, vector<T2> & vec_vals)
void vecToMap(map<T1, T2> &umap, vector<T1> &vec_keys, vector<T2> &vec_vals)
{
    for (plint i = 0; i < vec_keys.size(); ++i)
    {
        umap[vec_keys[i]] = vec_vals[i];
    }
}

// Creating custom MPI type to exchange info about platelets
void defineMPIStructComm(MPI_Datatype &type)
{
    const int nitems = 6;
    int blocklengths[nitems] = {1, 1, 1, 1, 1, 1};
    MPI_Datatype types[nitems] = {
        MPI_INT,
        MPI_INT,
        MPI_LONG_LONG,
        MPI_DOUBLE,
        MPI_DOUBLE,
        MPI_INT};
    MPI_Aint offsets[nitems];
    offsets[0] = offsetof(struct commPlatelet, idx);
    offsets[1] = offsetof(struct commPlatelet, proc);
    offsets[2] = offsetof(struct commPlatelet, loc);
    offsets[3] = offsetof(struct commPlatelet, activation);
    offsets[4] = offsetof(struct commPlatelet, bondTime);
    offsets[5] = offsetof(struct commPlatelet, type);
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &type);
    MPI_Type_commit(&type);
}

bool sortByTime(const commPlatelet &a, const commPlatelet &b)
{
    return a.bondTime < b.bondTime;
}

void updateNeighboringRates(long long int loc)
{
    platelet plt;
    plt.center[0] = loc / (ny * nz);
    plt.center[1] = (loc / nz) % ny;
    plt.center[2] = loc % nz;
    plint cx, cy, cz;
    cx = plt.center[0];
    cy = plt.center[1];
    cz = plt.center[2];

    T dia = plt.dp;
    plint n;
    n = ceil(dia / h);

    plint i, j, k;
    Array<plint, 3> pos;

    list<T> overlapLinks;

    for (i = cx - n; i <= cx + n; ++i)
    {
        for (j = cy - n; j <= cy + n; ++j)
        {
            for (k = cz - n; k <= cz + n; ++k)
            {
                if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz)
                {
                    platelet tempPlt;
                    tempPlt.center[0] = i;
                    tempPlt.center[1] = j;
                    tempPlt.center[2] = k;

                    auto range = occupation.equal_range((long long int)ny * nz * i + (long long int)nz * j + (long long int)k);

                    for (auto iter = range.first; iter != range.second; ++iter)
                    {
                        if (iter->second >= 0 && iter->second < plateletList.size())
                        {
                            if (i != plt.center[0] || j != plt.center[1] || k != plt.center[2])
                            {
                                tempPlt = plateletList[iter->second];

                                if (checkOverlapBinding(tempPlt, plt))
                                {
                                    if (!tempPlt.bound)
                                    {
                                        setRates(iter->second, tempPlt.lastMotionType);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void updateUnbindingRates(long long int loc, long layer)
{
    platelet plt;
    plt.center[0] = loc / (ny * nz);
    plt.center[1] = (loc / nz) % ny;
    plt.center[2] = loc % nz;
    plint cx, cy, cz;
    cx = plt.center[0];
    cy = plt.center[1];
    cz = plt.center[2];

    T dia = plt.dp;
    plint n;
    n = ceil(dia / h);

    plint i, j, k;
    Array<plint, 3> pos;

    list<T> overlapLinks;

    vector<long long int> unboundLocs;
    vector<long> unboundLayers;

    for (i = cx - n; i <= cx + n; ++i)
    {
        for (j = cy - n; j <= cy + n; ++j)
        {
            for (k = cz - n; k <= cz + n; ++k)
            {
                if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz)
                {
                    platelet tempPlt;
                    tempPlt.center[0] = i;
                    tempPlt.center[1] = j;
                    tempPlt.center[2] = k;

                    auto range = occupation.equal_range((long long int)ny * nz * i + (long long int)nz * j + (long long int)k);

                    for (auto iterlocal = range.first; iterlocal != range.second; ++iterlocal)
                    {
                        if (iterlocal->second >= 0 && iterlocal->second < plateletList.size())
                        {
                            if (i != plt.center[0] || j != plt.center[1] || k != plt.center[2])
                            {
                                tempPlt = plateletList[iterlocal->second];

                                if (checkOverlapBinding(tempPlt, plt))
                                {
                                    if (tempPlt.bound && tempPlt.layer > layer)
                                    {
                                        if (tempPlt.unbinding < 0.5 * dblmax)
                                        {
                                            totalRate = totalRate - tempPlt.unbinding;
                                        }
                                        tempPlt.bound = false;
                                        tempPlt.unbound = true;
                                        for (plint q = 0; q < 8; ++q)
                                        {
                                            tempPlt.tau[q] = dblmax;
                                        }
                                        plateletList[iterlocal->second] = tempPlt;
                                        unboundListLocal.push_back(iterlocal->second);

                                        long long temploc = (long long int)ny * nz * i + (long long int)nz * j + (long long int)k;
                                        unboundLocs.push_back(temploc);
                                        unboundLayers.push_back(tempPlt.layer);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    int n_local = unboundLocs.size();
    vector<int> n_global;
    n_global.resize(numprocs);

    MPI_Allgather(&n_local, 1, MPI_INT, &n_global[0], 1, MPI_INT, world);

    vector<int> displs;
    displs.push_back(0);
    int n_total = n_global[0];

    for (int i = 1; i < numprocs; ++i)
    {
        displs.push_back(displs[i - 1] + n_global[i - 1]);
        n_total = n_total + n_global[i];
    }

    if (n_total == 0)
    {
        return;
    }

    vector<long long> netLocs;
    vector<long> netLayers;
    netLocs.resize(n_total);
    netLayers.resize(n_total);
    MPI_Allgatherv(&unboundLocs[0], n_local, MPI_LONG_LONG, &netLocs[0], &n_global[0], &displs[0], MPI_LONG_LONG, world);
    MPI_Allgatherv(&unboundLayers[0], n_local, MPI_LONG, &netLayers[0], &n_global[0], &displs[0], MPI_LONG, world);

    for (i = 0; i < netLocs.size(); ++i)
    {
        auto iT = occupationGlobal.find(netLocs[i]);

        if (iT != occupationGlobal.end())
        {
            occupationGlobal.erase(iT);
        }
    }
    for (i = 0; i < netLocs.size(); ++i)
    {
        updateUnbindingRates(netLocs[i], netLayers[i]);
    }
}

void updateBoundPlateletMap()
{
    vector<int> nplatelets;
    nplatelets.resize(numprocs);

    int nplatelets_local = commPlateletList.size();

    MPI_Allgather(&nplatelets_local, 1, MPI_INT, &nplatelets[0], 1, MPI_INT, world);

    vector<int> displs;
    displs.push_back(0);

    int nplatelets_total = nplatelets[0];
    for (int i = 1; i < numprocs; ++i)
    {
        displs.push_back(displs[i - 1] + nplatelets[i - 1]);
        nplatelets_total = nplatelets_total + nplatelets[i];
    }

    vector<commPlatelet> netPlateletList;
    netPlateletList.resize(nplatelets_total);
    MPI_Allgatherv(&commPlateletList[0], nplatelets_local, mpi_commplatelet, &netPlateletList[0], &nplatelets[0], &displs[0], mpi_commplatelet, world);

    // std::sort(netPlateletList.begin(), netPlateletList.end(), sortByTime);

    int j;
    std::unordered_set<long long int> boundLocs;
    vector<commPlatelet> netPlateletListCorrected;
    for (j = 0; j < netPlateletList.size(); ++j)
    {
        commPlatelet cplt = netPlateletList[j];

        if (cplt.type == 6)
        {
            bool flag = false;
            auto iT = boundLocs.begin();

            iT = boundLocs.find(cplt.loc);
            if (iT != boundLocs.end())
            {
                flag = true;
            }

            iT = boundLocs.find(cplt.loc - ny * nz);
            if (iT != boundLocs.end())
            {
                flag = true;
            }

            iT = boundLocs.find(cplt.loc + ny * nz);
            if (iT != boundLocs.end())
            {
                flag = true;
            }

            iT = boundLocs.find(cplt.loc - nz);
            if (iT != boundLocs.end())
            {
                flag = true;
            }

            iT = boundLocs.find(cplt.loc + nz);
            if (iT != boundLocs.end())
            {
                flag = true;
            }

            iT = boundLocs.find(cplt.loc - 1);
            if (iT != boundLocs.end())
            {
                flag = true;
            }

            iT = boundLocs.find(cplt.loc + 1);
            if (iT != boundLocs.end())
            {
                flag = true;
            }

            if (flag)
            {
                continue;
            }
            else
            {
                boundLocs.insert(cplt.loc);
                netPlateletListCorrected.push_back(cplt);
            }
        }
        if (cplt.type >= 0 && cplt.type < 6)
        {
            long long int center[3] = {cplt.loc / (ny * nz), (cplt.loc / nz) % ny, cplt.loc % nz};

            center[cplt.type / 2] = center[cplt.type / 2] - pow(-1, cplt.type / 2);

            auto iT = occupationGlobal.find((long long int)ny * nz * center[0] +
                                            (long long int)nz * center[1] + (long long int)center[2]);

            if (iT != occupationGlobal.end())
            {
                occupationGlobal.erase(iT);
            }
        }

        occupationGlobal[cplt.loc] = cplt.activation;
    }

    for (j = 0; j < netPlateletList.size(); ++j)
    {
        commPlatelet cplt = netPlateletList[j];
        if (cplt.type == 7)
        {
            long layer = (long)(cplt.activation / 100.);
            occupationGlobal[cplt.loc] = cplt.activation - 100. * layer;
        }
    }

    // Don't allow two binding events to occur simultaneously at the same location on different procs if you can help it
    for (j = 0; j < netPlateletListCorrected.size(); ++j)
    {
        commPlatelet cplt = netPlateletListCorrected[j];

        auto range = occupation.equal_range(cplt.loc);

        for (auto iT = range.first; iT != range.second; ++iT)
        {
            if (procid != cplt.proc || iT->second != cplt.idx)
            {
                if (iT->second >= 0 && iT->second < plateletList.size())
                {
                    plateletList[iT->second].bound = false;
                }
            }
        }

        range = occupation.equal_range(cplt.loc - ny * nz);

        for (auto iT = range.first; iT != range.second; ++iT)
        {
            if (procid != cplt.proc || iT->second != cplt.idx)
            {
                if (iT->second >= 0 && iT->second < plateletList.size())
                {
                    plateletList[iT->second].bound = false;
                }
            }
        }

        range = occupation.equal_range(cplt.loc + ny * nz);

        for (auto iT = range.first; iT != range.second; ++iT)
        {
            if (procid != cplt.proc || iT->second != cplt.idx)
            {
                if (iT->second >= 0 && iT->second < plateletList.size())
                {
                    plateletList[iT->second].bound = false;
                }
            }
        }

        range = occupation.equal_range(cplt.loc - nz);

        for (auto iT = range.first; iT != range.second; ++iT)
        {
            if (procid != cplt.proc || iT->second != cplt.idx)
            {
                if (iT->second >= 0 && iT->second < plateletList.size())
                {
                    plateletList[iT->second].bound = false;
                }
            }
        }

        range = occupation.equal_range(cplt.loc + nz);

        for (auto iT = range.first; iT != range.second; ++iT)
        {
            if (procid != cplt.proc || iT->second != cplt.idx)
            {
                if (iT->second >= 0 && iT->second < plateletList.size())
                {
                    plateletList[iT->second].bound = false;
                }
            }
        }

        range = occupation.equal_range(cplt.loc - 1);

        for (auto iT = range.first; iT != range.second; ++iT)
        {
            if (procid != cplt.proc || iT->second != cplt.idx)
            {
                if (iT->second >= 0 && iT->second < plateletList.size())
                {
                    plateletList[iT->second].bound = false;
                }
            }
        }

        range = occupation.equal_range(cplt.loc + 1);

        for (auto iT = range.first; iT != range.second; ++iT)
        {
            if (procid != cplt.proc || iT->second != cplt.idx)
            {
                if (iT->second >= 0 && iT->second < plateletList.size())
                {
                    plateletList[iT->second].bound = false;
                }
            }
        }
    }

    for (j = 0; j < netPlateletList.size(); ++j)
    {
        commPlatelet cplt = netPlateletList[j];

        if (cplt.type == 7)
        {
            long layer = (long)(cplt.activation / 100.);
            updateUnbindingRates(cplt.loc, layer);
        }
    }

    for (j = 0; j < unboundListLocal.size(); ++j)
    {
        plint ind = unboundListLocal[j];
        plateletList[ind].unbound = false;
        setRates(ind, 7);
    }

    vector<plint> tempUnboundListLocal{};
    unboundListLocal.swap(tempUnboundListLocal);

    for (j = 0; j < netPlateletList.size(); ++j)
    {
        commPlatelet cplt = netPlateletList[j];

        if (cplt.type == 6)
        {
            updateNeighboringRates(cplt.loc);
        }
    }

    vector<commPlatelet> tempCommPlateletList{};
    commPlateletList.swap(tempCommPlateletList);
}

void runKMC()
{
    plint current;
    plint j;

    MPI_Barrier(world);

    global::timer("commtime").restart();

    if (numprocs > 1)
    {
        if (movecount % update_freq == 0)
        {
            totalRate = totalRate - inletRate;
            MPI_Allreduce(&totalRate, &maxTotalRate, 1, MPI_DOUBLE, MPI_MAX, world);

            T fractionRate = exp(exp(maxTotalRate / totalRate));
            T sumFractionRate;

            MPI_Allreduce(&fractionRate, &sumFractionRate, 1, MPI_DOUBLE, MPI_SUM, world);

            inletRate = (fractionRate / sumFractionRate) * oldInletRate;

            // inletRate = oldInletRate / numprocs;
            totalRate = totalRate + inletRate;

            maxTotalRate = 1.1 * maxTotalRate;

            eventList.erase(event(-1, 8, inletTau));
            inletTau = totalTimeNRM - log(randn()) / inletRate;
            eventList.insert(event(-1, 8, inletTau));

            global::timer("mapupdate").restart();
            updateBoundPlateletMap();
            mapupdatetime += global::timer("mapupdate").getTime();
        }
    }

    if (numprocs == 1)
    {
        maxTotalRate = totalRate;
    }

    nullRate = maxTotalRate - totalRate;

    if (nullRate < 0.)
    {
        nullRate = 0.;
    }

    commtime = commtime + global::timer("commtime").getTime();

    if (nullRate > 1.e-10)
    {
        nullTau = totalTimeNRM - log(randn()) / nullRate;
        eventList.insert(event(-1, -1, nullTau));
    }
    else
    {
        nullTau = dblmax;
    }

    T oldTime = totalTimeNRM;

    auto executedEvent = eventList.begin();
    current = executedEvent->idx;
    j = executedEvent->type;
    totalTimeNRM = executedEvent->tau;
    platelet plt;

    if (totalTimeNRM < oldTime)
    {
        totalTimeNRM = oldTime;
    }

    if (j >= 0 && j < 8)
    {
        if (current >= 0 && current < plateletList.size())
        {
            plt = plateletList[current];
            if (plt.tau[j] != totalTimeNRM)
            {
                j = -1;
                current = -1;
            }
            else
            {
                if (totalTimeNRM < oldTime)
                {
                    cout << "Tau = " << plt.tau[j] << endl;
                    cout << "Rate of motion = " << ((j < 6) ? plt.motion[j] : 0) << endl;
                    cout << "Bonding rate = " << plt.binding << endl;
                    cout << "Unbonding rate = " << plt.unbinding << endl;
                    cout << "Activation = " << plt.activation << endl;
                    cout << "Local shear = " << plt.localShear << endl;
                    cout << "Old time = " << oldTime << endl;
                }
            }
        }
        else
        {
            j = -1;
            current = -1;
        }
    }

    // Inlet event
    if (j == 8)
    {
        if (inletTau == totalTimeNRM)
        {
            if (addPlateletInlet())
            {
                ++inCount;
                current = plateletList.size() - 1;
            }
            else
            {
                j = -1;
                current = -1;
            }

            if (totalTimeNRM < oldTime)
            {
                cout << "Tau = " << inletTau << endl;
                cout << "Rate = " << inletRate << endl;
            }
        }
        else
        {
            j = -1;
            current = -1;
        }

        // Inlet rate putative time reset
        T r = randn();
        T dtau = totalTimeNRM - log(r) / inletRate;
        eventList.insert(event(-1, 8, dtau));
        inletTau = dtau;
    }

    if (j >= 0 && j < 6 /*&& numprocs > 1*/)
    {
        if (simpleBlocking(current, j))
        {
            j = -1;
            current = -1;
        }
    }

    if (j == 6 && numprocs > 1)
    {
        auto iTBound = occupationGlobal.find((long long int)ny * nz * plt.center[0] +
                                             (long long int)nz * plt.center[1] + (long long int)plt.center[2]);
        if (iTBound != occupationGlobal.end())
        {
            if (iTBound->second > 100.)
            {
                j = -1;
                current = -1;
            }
        }
    }

    if (current >= 0)
    {
        plt = plateletList[current];
    }
    else
    {
        plt = dummyPlatelet;
    }

    if (j >= 0 && j < 6) // Motion event
    {
        /*
        auto iT = occupation.find((long long int)ny * nz * plt.center[0] +
                                  (long long int)nz * plt.center[1] + (long long int)plt.center[2]);
        if (iT != occupation.end())
        {
            occupation.erase(iT);
        }
        */

        auto range = occupation.equal_range((long long int)ny * nz * plt.center[0] +
                                            (long long int)nz * plt.center[1] + (long long int)plt.center[2]);
        for (auto iT = range.first; iT != range.second; ++iT)
        {
            if (iT->second == current)
            {
                occupation.erase(iT);
                break;
            }
        }

        plt.center[j / 2] = plt.center[j / 2] + (plint)pow(-1, j);
        plt.lastMotionType = j;

        std::pair<long long int, plint> pltPair(
            (long long int)ny * nz * plt.center[0] + (long long int)nz * plt.center[1] + (long long int)plt.center[2], current);
        occupation.insert(pltPair);
    }

    if (j == 6)
        plt.bound = true;
    if (j == 7)
        plt.bound = false;

    if (current >= 0)
    {
        plateletList[current] = plt;
    }

    // if (j == 6 || j == 7 && numprocs > 1) // Bonding/unbonding event
    if (j == 6 || j == 7)
    {
        commPlatelet cplt;
        cplt.idx = current;
        cplt.proc = procid;
        cplt.loc = (long long int)ny * nz * plt.center[0] +
                   (long long int)nz * plt.center[1] + (long long int)plt.center[2];
        // cplt.activation = (j == 6) ? (100. * plt.layer + plt.activation * plt.activationx) : (plt.activation * plt.activationx);
        cplt.activation = 100. * plt.layer + plt.activation * plt.activationx;
        cplt.bondTime = totalTime;
        cplt.type = j;
        commPlateletList.push_back(cplt);
    }

    eventList.erase(executedEvent);

    if (numprocs == 1)
    {
        global::timer("mapupdate").restart();
        updateBoundPlateletMap();
        mapupdatetime += global::timer("mapupdate").getTime();
    }

    if (j == 6 && numprocs > 1)
    {
        plt = plateletList[current];
        if (!plt.bound)
            j = plt.lastMotionType;
    }

    if (j >= 0)
    {
        setRates(current, j); // Update rate database
    }

    ++movecount;
    if (j < 0)
    {
        ++nullcount;
    }

    T dtKMC = -log(randn_parallel()) / maxTotalRate;
    totalTime = totalTime + dtKMC;

    eventList.erase(event(-1, -1, nullTau));

    if (totalRate > maxTotalRate)
    {
        ++exceedcount;
    }
}

void runKMCOld()
{
    plint current;
    plint j;

    global::timer("commtime").restart();

    if (numprocs > 1)
    {
        if (totalTime >= 0.005 * tsteps)
        {
            totalRate = totalRate - inletRate;

            inletRate = oldInletRate / numprocs;

            totalRate = totalRate + inletRate;

            maxTotalRate = totalRate;
            ++tsteps;

            eventList.erase(event(-1, 8, inletTau));
            inletTau = totalTimeNRM - log(randn()) / inletRate;
            eventList.insert(event(-1, 8, inletTau));

            global::timer("mapupdate").restart();
            updateBoundPlateletMap();
            mapupdatetime += global::timer("mapupdate").getTime();
        }
    }

    if (numprocs == 1)
    {
        maxTotalRate = totalRate;
    }

    nullRate = maxTotalRate - totalRate;

    if (nullRate < 0.)
    {
        nullRate = 0.;
    }

    commtime = commtime + global::timer("commtime").getTime();

    if (nullRate > 1.e-10)
    {
        nullTau = totalTimeNRM - log(randn()) / nullRate;
        eventList.insert(event(-1, -1, nullTau));
    }
    else
    {
        nullTau = dblmax;
    }

    T oldTime = totalTimeNRM;

    auto executedEvent = eventList.begin();
    current = executedEvent->idx;
    j = executedEvent->type;
    totalTimeNRM = executedEvent->tau;
    platelet plt;

    if (totalTimeNRM < oldTime)
    {
        totalTimeNRM = oldTime;
    }

    if (j >= 0 && j < 8)
    {
        if (current >= 0 && current < plateletList.size())
        {
            plt = plateletList[current];
            if (plt.tau[j] != totalTimeNRM)
            {
                j = -1;
                current = -1;
            }
            else
            {
                if (totalTimeNRM < oldTime)
                {
                    cout << "Tau = " << plt.tau[j] << endl;
                    cout << "Rate of motion = " << ((j < 6) ? plt.motion[j] : 0) << endl;
                    cout << "Bonding rate = " << plt.binding << endl;
                    cout << "Unbonding rate = " << plt.unbinding << endl;
                    cout << "Activation = " << plt.activation << endl;
                    cout << "Local shear = " << plt.localShear << endl;
                    cout << "Old time = " << oldTime << endl;
                }
            }
        }
        else
        {
            j = -1;
            current = -1;
        }
    }

    // Inlet event
    if (j == 8)
    {
        if (inletTau == totalTimeNRM)
        {
            if (addPlateletInlet())
            {
                ++inCount;
                current = plateletList.size() - 1;
            }
            else
            {
                j = -1;
                current = -1;
            }

            if (totalTimeNRM < oldTime)
            {
                cout << "Tau = " << inletTau << endl;
                cout << "Rate = " << inletRate << endl;
            }
        }
        else
        {
            j = -1;
            current = -1;
        }

        // Inlet rate putative time reset
        T r = randn();
        T dtau = totalTimeNRM - log(r) / inletRate;
        eventList.insert(event(-1, 8, dtau));
        inletTau = dtau;
    }

    if (j >= 0 && j < 6 /*&& numprocs > 1*/)
    {
        if (simpleBlocking(current, j))
        {
            j = -1;
            current = -1;
        }
    }

    if (j == 6 && numprocs > 1)
    {
        auto iTBound = occupationGlobal.find((long long int)ny * nz * plt.center[0] +
                                             (long long int)nz * plt.center[1] + (long long int)plt.center[2]);
        if (iTBound != occupationGlobal.end())
        {
            if (iTBound->second > 100.)
            {
                j = -1;
                current = -1;
            }
        }
    }

    if (current >= 0)
    {
        plt = plateletList[current];
    }
    else
    {
        plt = dummyPlatelet;
    }

    if (j >= 0 && j < 6) // Motion event
    {
        /*
        auto iT = occupation.find((long long int)ny * nz * plt.center[0] +
                                  (long long int)nz * plt.center[1] + (long long int)plt.center[2]);
        if (iT != occupation.end())
        {
            occupation.erase(iT);
        }
        */

        auto range = occupation.equal_range((long long int)ny * nz * plt.center[0] +
                                            (long long int)nz * plt.center[1] + (long long int)plt.center[2]);
        for (auto iT = range.first; iT != range.second; ++iT)
        {
            if (iT->second == current)
            {
                occupation.erase(iT);
                break;
            }
        }

        plt.center[j / 2] = plt.center[j / 2] + (plint)pow(-1, j);
        plt.lastMotionType = j;

        std::pair<long long int, plint> pltPair(
            (long long int)ny * nz * plt.center[0] + (long long int)nz * plt.center[1] + (long long int)plt.center[2], current);
        occupation.insert(pltPair);
    }

    if (j == 6)
        plt.bound = true;
    if (j == 7)
        plt.bound = false;

    if (current >= 0)
    {
        plateletList[current] = plt;
    }

    // if (j == 6 || j == 7 && numprocs > 1) // Bonding/unbonding event
    if (j == 6 || j == 7)
    {
        commPlatelet cplt;
        cplt.idx = current;
        cplt.proc = procid;
        cplt.loc = (long long int)ny * nz * plt.center[0] +
                   (long long int)nz * plt.center[1] + (long long int)plt.center[2];
        // cplt.activation = (j == 6) ? (100. * plt.layer + plt.activation * plt.activationx) : (plt.activation * plt.activationx);
        cplt.activation = 100. * plt.layer + plt.activation * plt.activationx;
        cplt.bondTime = totalTime;
        cplt.type = j;
        commPlateletList.push_back(cplt);
    }

    eventList.erase(executedEvent);

    if (j == 6 && numprocs > 1)
    {
        plt = plateletList[current];
        if (!plt.bound)
            j = plt.lastMotionType;
    }

    if (j >= 0)
    {
        setRates(current, j); // Update rate database
    }

    ++movecount;

    if (j < 0)
    {
        ++nullcount;
    }

    totalTime = totalTimeNRM;

    eventList.erase(event(-1, -1, nullTau));

    if (totalRate > maxTotalRate)
    {
        ++exceedcount;
    }
}

T readShear(plint px, plint py, plint pz)
{
    MPI_Status stat;
    MPI_Offset offset;
    plint loc = pz + py * nzlb + px * nzlb * nylb;
    offset = loc * sizeof(T);
    T shear;
    MPI_File_read_at(fileshear, offset, &shear, 1, MPI_DOUBLE, &stat);
    return shear;
}

// This function is used to compute the local shear around a platelet
T computePltShear(plint current)
{
    getsheartime -= MPI_Wtime();
    platelet plt = plateletList[current];

    T netShear = 0.;
    plint px, py, pz;
    px = plt.center[0] / factor;
    py = plt.center[1] / factor;
    pz = plt.center[2] / factor;
    plint count = 0;
    T maxShear = 0.;
    plint pltRad = util::roundToInt(0.5 * pltDia / dx);
    pltRad = 1;

    vector<T> localShears;

    for (plint i = px - pltRad; i <= px + pltRad; ++i)
    {
        for (plint j = py - pltRad; j <= py + pltRad; ++j)
        {
            for (plint k = pz - pltRad; k <= pz + pltRad; ++k)
            {
                if (i < 0 || i >= nxlb || j < 0 || j >= nylb || k < 0 || k >= nzlb)
                {
                    continue;
                }
                T tempShear = readShear(i, j, k);
                if (isnan(tempShear))
                {
                    continue;
                }
                if (tempShear > maxShear)
                {
                    maxShear = tempShear;
                }

                if (tempShear / dt > 10.)
                {
                    netShear = netShear + tempShear;
                    count = count + 1;
                    localShears.push_back(tempShear);
                }
            }
        }
    }

    count = 0;
    netShear = 0;
    for (int q = 0; q < localShears.size(); ++q)
    {
        if (localShears[q] > 0.75 * maxShear)
        {
            netShear = netShear + localShears[q];
            count = count + 1;
        }
    }

    getsheartime += MPI_Wtime();
    return (sqrt(2.) * maxShear / dt);

    return 0.;
}

// Write out the velocity field and platelet positions for visualization
void writeOutputFiles()
{
    std::string fname = "platelets";
    fname = fname + std::to_string(procid) + ".csv." + std::to_string(iter);
    char filename[fname.size() + 1];
    fname.copy(filename, fname.size() + 1);
    filename[fname.size()] = '\0';

    ofstream file;
    file.open(filename, ios::app);

    std::string fnamebound = "bound" + fname;
    char filenamebound[fnamebound.size() + 1];
    fnamebound.copy(filenamebound, fnamebound.size() + 1);
    filenamebound[fnamebound.size()] = '\0';

    ofstream filebound;
    filebound.open(filenamebound, ios::app);

    file << "xcoord, ycoord, zcoord, activation" << endl;
    filebound << "xcoord, ycoord, zcoord, activation" << endl;

    for (plint i = 0; i < plateletList.size(); ++i)
    {
        platelet plt = plateletList[i];

        Array<T, 3> realPos;

        realPos[0] = (plt.center[0]) * h + KMCLocation[0];
        realPos[1] = (plt.center[1]) * h + KMCLocation[1];
        realPos[2] = (plt.center[2]) * h + KMCLocation[2];

        file << realPos[0] << ",";
        file << realPos[1] << ",";
        file << realPos[2] << ",";
        file << plt.calcInteg << endl;

        if (plt.bound)
        {
            filebound << realPos[0] << ",";
            filebound << realPos[1] << ",";
            filebound << realPos[2] << ",";
            filebound << plt.calcInteg << endl;
        }
    }

    file.close();
    filebound.close();

    if ((iter - iter_checkpoint) % 200 == 0)
    {
        writeVTK(*LB.lattice, dx, dt, iter);
    }
}

// Checkpoint the simulation data to resume
void checkpointSimulation()
{
    std::string fname = "checkpointkmc_";
    fname = fname + std::to_string(iter) + "_" + std::to_string(procid) + ".dat";
    char filename[fname.size() + 1];
    fname.copy(filename, fname.size() + 1);
    filename[fname.size()] = '\0';

    ofstream file;
    file.open(filename, ios::app | ios::binary);

    if (!file)
    {
        cout << "Could not open file to checkpoint on procid "
             << procid
             << "at timestep "
             << iter << endl;

        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    for (plint i = 0; i < plateletList.size(); ++i)
    {
        platelet plt = plateletList[i];

        file.write((char *)&plt, sizeof(struct platelet));
    }

    file.close();
}

// Load previously checkpointed data
void loadSimulation(plint iter, plint numprocs_checkpoint)
{
    plateletList.clear();
    totalRate = 0.;
    if (numprocs != numprocs_checkpoint)
    {
        if (procid == 0)
        {
            for (plint j = 0; j < numprocs_checkpoint; ++j)
            {
                std::string fname = "checkpointkmc_";
                fname = fname + std::to_string(iter) + "_" + std::to_string(j) + ".dat";
                char filename[fname.size() + 1];
                fname.copy(filename, fname.size() + 1);
                filename[fname.size()] = '\0';

                ifstream file;
                file.open(filename, ios::in | ios::binary);

                if (!file)
                {
                    cout << "Could not open file " << j
                         << " to load from checkpoint" << endl;
                    MPI_Abort(MPI_COMM_WORLD, -1);
                }

                platelet plt;

                while (!file.eof())
                {
                    file.read((char *)&plt, sizeof(struct platelet));
                    plateletList.push_back(plt);
                    totalRate = totalRate + plt.sumRates;
                }

                file.close();
            }
        }
        distributePlatelets();
    }
    else
    {
        std::string fname = "checkpointkmc_";
        fname = fname + std::to_string(iter) + "_" + std::to_string(procid) + ".dat";
        char filename[fname.size() + 1];
        fname.copy(filename, fname.size() + 1);
        filename[fname.size()] = '\0';

        ifstream file;
        file.open(filename, ios::in | ios::binary);

        if (!file)
        {
            cout << "Could not open file " << procid
                 << " to load from checkpoint" << endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        platelet plt;

        while (!file.eof())
        {
            file.read((char *)&plt, sizeof(struct platelet));
            plateletList.push_back(plt);
            totalRate = totalRate + plt.sumRates;
        }

        file.close();
    }

    updatePlateletMap();
}

void LBClass::getVoxelizedDomain(plint level)
{
    T nuLB_ = dt * kinematicViscosity / (dx * dx);

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

    dt = nuLB_ * dx * dx / kinematicViscosity;
    T uAveLB = averageInletVelocity * dt / dx;
    T omega = 1. / (3. * nuLB_ + 0.5);
    Array<T, 3> location(boundary.getPhysicalLocation());
    LBLocation = location;

    // The aneurysm simulation is an interior (as opposed to exterior) flow problem. For
    //   this reason, the lattice nodes that lay inside the computational domain must
    //   be identified and distinguished from the ones that lay outside of it. This is
    //   handled by the following voxelization process.
    const int flowType = voxelFlag::inside;

    VoxelizedDomain3D<T> voxelizedDomain_(
        boundary, flowType, extraLayer, borderWidth, extendedEnvelopeWidth, blockSize);

    voxelizedDomain.reset(new VoxelizedDomain3D<T>(voxelizedDomain_));

    // Super important. Identifies the walls of the domain based on the flag

    flagMatrixInside.reset(new MultiScalarField3D<int>((MultiBlock3D &)voxelizedDomain->getVoxelMatrix()));

    setToConstant(*flagMatrixInside, voxelizedDomain->getVoxelMatrix(),
                  voxelFlag::inside, flagMatrixInside->getBoundingBox(), 1);
    setToConstant(*flagMatrixInside, voxelizedDomain->getVoxelMatrix(),
                  voxelFlag::innerBorder, flagMatrixInside->getBoundingBox(), 1);

    // pcout << "Number of fluid cells: " << computeSum(*flagMatrixInside) << std::endl;

    nxlb = ((MultiBlock3D &)voxelizedDomain->getVoxelMatrix()).getNx();
    nylb = ((MultiBlock3D &)voxelizedDomain->getVoxelMatrix()).getNy();
    nzlb = ((MultiBlock3D &)voxelizedDomain->getVoxelMatrix()).getNz();

    boolMask.reset(new MultiScalarField3D<int>((MultiBlock3D &)voxelizedDomain->getVoxelMatrix()));
    setToConstant(*boolMask, boolMask->getBoundingBox(), 0);

    parallelIO::saveFull(*flagMatrixInside, "flag", IndexOrdering::forward, false);
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

void determineEffectiveRadius()
{
    for (plint i = nx - 1; i >= 0; --i)
    {
        if (checkWithinDomain(i, ny / 2, nz / 8))
        {
            maxx = i;
            break;
        }
    }
    for (plint i = 0; i < nx; ++i)
    {
        if (checkWithinDomain(i, ny / 2, nz / 8))
        {
            minx = i;
            break;
        }
    }
    for (plint j = ny - 1; j >= 0; --j)
    {
        if (checkWithinDomain(nx / 2, j, nz / 8))
        {
            maxy = j;
            break;
        }
    }
    for (plint j = 0; j < ny; ++j)
    {
        if (checkWithinDomain(nx / 2, j, nz / 8))
        {
            miny = j;
            break;
        }
    }

    radiusEffectivex = (maxx - minx + 1) * 0.5 * h;
    radiusEffectivey = (maxy - miny + 1) * 0.5 * h;
}

/// A functional used to assign Poiseuille velocity profile at inlet
T poiseuilleVelocity(plint iX, plint iY, T uLB)
{
    T x = ((T)iX * dx + LBLocation[0]);
    T y = ((T)iY * dx + LBLocation[1]);
    T r2 = (x * x + y * y) / (radiusReal * radiusReal);
    T u = 2. * uLB * (1 - r2);
    return u;
}

/// Write the full velocity and the velocity-norm into a VTK file.
void writeVTK(MultiBlockLattice3D<T, DESCRIPTOR> &lattice,
              T dx, T dt, plint iter)
{
    ParallelVtkImageOutput3D<T> vtkOut(createFileName("vtk", iter, 6), 3, dx);
    vtkOut.writeData<T>(*computeVelocityNorm(lattice), "velocityNorm", dx / dt);
    vtkOut.writeData<3, T>(*computeVelocity(lattice), "velocity", dx / dt);
    auto strainRate = computeStrainRateFromStress(lattice);
    auto shearRate = computeSymmetricTensorNorm(*strainRate, lattice.getBoundingBox());
    vtkOut.writeData<T>(*shearRate, "shear", sqrt(2.) / dt);
}

/// This is the function that prepares the actual LB simulation.
void updateLB(MultiBlockLattice3D<T, DESCRIPTOR> &lattice)
{
    global::timer("bounceback").restart();

    // Add bounceback nodes at places where there is platlet mass, this is obtained from LKMC
    // No slip BCs at platelet boundaries

    global::timer("bounceback").restart();
    // boolMask.reset(new MultiScalarField3D<int>((MultiBlock3D &)LB.voxelizedDomain->getVoxelMatrix()));

    setToConstant(*boolMask, boolMask->getBoundingBox(), 0);

    applyProcessingFunctional(new updateBoolMask(iter), boolMask->getBoundingBox(), *boolMask);
    bouncebacksettime = bouncebacksettime + global::timer("bounceback").getTime();

    if (iter > 0)
    {
        defineDynamics(lattice, *boolMask, lattice.getBoundingBox(), new BounceBack<T, DESCRIPTOR>, 1);
    }

    bouncebacksettime = bouncebacksettime + global::timer("bounceback").getTime();

    if (loadFromCheckpoint && ((iter - iter_checkpoint) == 0))
    {
        loadBinaryBlock(lattice, "checkpoint" + std::to_string(iter) + ".dat");
    }

    // Convergence bookkeeping
    util::ValueTracer<T> velocityTracer(1., resolution, epsilon);

    pcout << "Starting LB iteration at iter:" << iter << endl;

    global::timer("lbconvergence").restart();

    pluint maxT = (pluint)(dtLB / dt);
    for (pluint j = 0; j < maxT; ++j)
    {
        lattice.collideAndStream();

        if (j % 10 == 0)
        {
            velocityTracer.takeValue(computeAverageEnergy(lattice));
        }

        if (velocityTracer.hasConverged())
        {
            pcout << "Convergence at iter = " << j << endl;
            break;
        }
    }

    lbconvergencetime = lbconvergencetime + global::timer("lbconvergence").getTime();

    pcout << "LB iteration converged at iter:" << iter << endl;

    auto strainRate = computeStrainRateFromStress(lattice);
    auto velField = computeVelocity(lattice, lattice.getBoundingBox());
    auto shearRate = computeSymmetricTensorNorm(*strainRate, lattice.getBoundingBox());

    parallelIO::saveFull(*velField, "velocity", IndexOrdering::forward, false);
    parallelIO::saveFull(*shearRate, "shear", IndexOrdering::forward, false);

    strainRate.reset();
    velField.reset();
    shearRate.reset();

    pcout << "Velocity and shear fields dumped into output files for reading by KMC" << endl;
}

/// This is the function that prepares the actual LB simulation.
void runLB(plint level, MultiBlockLattice3D<T, DESCRIPTOR> *iniVal)
{
    Array<T, 3> location(LBLocation);
    T nuLB_ = dt * kinematicViscosity / (dx * dx);
    T uAveLB = averageInletVelocity * dt / dx;
    T omega = 1. / (3. * nuLB_ + 0.5);

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

    std::unique_ptr<MultiBlockLattice3D<T, DESCRIPTOR>> lattice =
        generateMultiBlockLattice<T, DESCRIPTOR>(
            LB.voxelizedDomain->getVoxelMatrix(), envelopeWidth, dynamics);
    lattice->toggleInternalStatistics(false);

    OnLatticeBoundaryCondition3D<T, DESCRIPTOR> *boundaryCondition = createLocalBoundaryCondition3D<T, DESCRIPTOR>();

    Array<T, 3> inletPos((inletRealPos - location) / dx);
    Array<T, 3> outletPos((outletRealPos - location) / dx);

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
        boundaryCondition->addPressureBoundary2P(outletDomain, *lattice);
        setBoundaryDensity(*lattice, outletDomain, 1.);
    }
    else
    {
        if (iter < 10)
        {
            boundaryCondition->addVelocityBoundary2N(inletDomain, *lattice);
            setBoundaryVelocity(*lattice, inletDomain, PoiseuilleVelocity3D(uAveLB));
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
    // if (iter == 0)
    //{
    //    initializeAtEquilibrium (*lattice, lattice -> getBoundingBox(), initializeDensityAndVelocity<T>(uAveLB));
    // initializeAtEquilibrium (*lattice, lattice -> getBoundingBox(), 1., Array<T,3>(0., 0., uAveLB) );
    //}

    lattice->initialize();

    if (iniVal)
    {
        if (level > 0)
        {
            Box3D toDomain(lattice->getBoundingBox());
            Box3D fromDomain(toDomain.shift(
                margin, margin, margin)); // During rescaling, the margin doubled in size,
                                          //   an effect which is cancelled here through a shift.
            copy(*iniVal, fromDomain, *lattice, toDomain, modif::staticVariables);
        }
        else
        {
            Box3D toDomain(lattice->getBoundingBox());
            Box3D fromDomain(toDomain);
            copy(*iniVal, fromDomain, *lattice, toDomain, modif::staticVariables);
        }
    }

    updateLB(*lattice);

    inletP = computeMax(*computeDensity(*lattice, inletDomain), inletDomain);
    outletP = 1.;

    inletPs[iter % inletPs.size()] = inletP;
    T averageinletP = 0.;
    for (int m = 0; m < inletPs.size(); ++m)
    {
        averageinletP += inletPs[m];
    }
    averageinletP /= inletPs.size();

    averageInletVelocityComputed = uAveLB * (1 - (inletP - outletP) / deltaPTotal);

    LB.assign(lattice);

    if (iter - iter_checkpoint == 0)
    {
        writeVTK(*LB.lattice, dx, dt, iter);
    }
}

// Save boundary conditions for checkpointed simulation.
void saveCheckpointBC()
{
    if (procid == 0)
    {
        std::string fname = "checkpointbc_";
        fname = fname + std::to_string(iter) + ".dat";
        char filename[fname.size() + 1];
        fname.copy(filename, fname.size() + 1);
        filename[fname.size()] = '\0';

        ofstream file;
        file.open(filename, ios::out | ios::binary);

        if (!file)
        {
            cout << "Could not open file to save BCs for checkpoint" << endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        file.write((char *)&inletP, sizeof(T));
        file.write((char *)&outletP, sizeof(T));

        file.close();
    }
}

// Load boundary conditions for checkpointed simulation.
void loadCheckpointBC(plint iter)
{
    if (procid == 0)
    {
        std::string fname = "checkpointbc_";
        fname = fname + std::to_string(iter) + ".dat";
        char filename[fname.size() + 1];
        fname.copy(filename, fname.size() + 1);
        filename[fname.size()] = '\0';

        ifstream file;
        file.open(filename, ios::in | ios::binary);

        if (!file)
        {
            cout << "Could not open file to load BCs from checkpoint" << endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        file.read((char *)&inletP, sizeof(T));
        file.read((char *)&outletP, sizeof(T));

        file.close();
    }

    MPI_Bcast(&inletP, 1, MPI_DOUBLE, 0, world);
    MPI_Bcast(&outletP, 1, MPI_DOUBLE, 0, world);
}

/********* Routine that uses MUI to exchange solutions between different solvers ***********/

void interface_solution(std::vector<std::unique_ptr<mui::uniface<mui::one_dim>>> &ifs,
                        mui::chrono_sampler_exact<mui::one_dim> chrono_sampler)
{
    plint iter_ = iter - iter_checkpoint;

    mui::point1d push_point;
    push_point[0] = 0.;

    plint current;

    std::unordered_set<plint> tempBoolMask;
    boolMaskLKMC.swap(tempBoolMask);

    if (iter_ == 0)
    {
        ifs[0]->push("resolutionx", push_point, (T)nxlb);
        ifs[0]->push("resolutiony", push_point, (T)nylb);
        ifs[0]->push("resolutionz", push_point, (T)nzlb);
        ifs[0]->push("dx", push_point, dx);
        ifs[0]->push("dt", push_point, dt);
        ifs[0]->push("locationx", push_point, LBLocation[0] + 0.5 * dx);
        ifs[0]->push("locationy", push_point, LBLocation[1] + 0.5 * dx);
        ifs[0]->push("locationz", push_point, LBLocation[2] + 0.5 * dx);
    }

    // Push bound platelet centers to palabos
    for (current = 0; current < plateletList.size(); ++current)
    {
        platelet plt = plateletList[current];

        push_point[0] = (long long int)ny * nz * plt.center[0] +
                        (long long int)nz * plt.center[1] + (long long int)plt.center[2];
        Array<T, 3> realPos((T)plt.center[0] + 0.5, (T)plt.center[1] + 0.5, (T)plt.center[2] + 0.5);
        realPos = h * realPos + LBLocation;

        mui::sendplatelet sendplt;
        sendplt.px = realPos[0];
        sendplt.py = realPos[1];
        sendplt.pz = realPos[2];
        sendplt.release = plt.releaseTime;
        sendplt.calcInteg = plt.calcInteg;
        sendplt.loc = (long long int)ny * nz * plt.center[0] +
                      (long long int)nz * plt.center[1] + (long long int)plt.center[2];
        sendplt.rank = procid;

        ifs[0]->push("plt", push_point, sendplt);
    }

    ifs[0]->commit(iter_);
    ifs[0]->forget(iter_ - 1);

    pcout << "KMC pushed platelet granule release times to OpenFOAM" << endl;

    if (iter > 0)
    {
        if (iter_ > 0)
        {
            MPI_File_close(&filevel);
            MPI_File_close(&fileshear);
            MPI_File_close(&fileflag);
        }

        pcout << "Running LB solver" << endl;
        global::timer("LB").restart();
        runLB(0, LB.lattice.get());
        // runLB();
        lbtime = lbtime + global::timer("LB").getTime();
        pcout << "LB solver converged successfully" << endl;

        push_point[0] = 0;
        ifs[1]->push("signal", push_point, 1.);
    }

    ifs[1]->commit(iter_);
    ifs[1]->forget(iter_ - 1);

    ifs[2]->commit(iter_);

    auto recvplt = ifs[2]->fetch_values<mui::recvplatelet>("plt", iter_, chrono_sampler);
    pcout << "KMC/NN received updated agonist concentrations from OpenFOAM" << endl;

    ifs[2]->forget(iter_);

    std::map<long long int, T> adpMap;
    std::map<long long int, T> txa2Map;

    for (plint i = 0; i < recvplt.size(); ++i)
    {
        adpMap[recvplt[i].loc] = recvplt[i].adpC;
        txa2Map[recvplt[i].loc] = recvplt[i].txa2C;
    }

    pcout << recvplt.size() << endl;
    pcout << adpMap.size() << endl;

    for (current = 0; current < plateletList.size(); ++current)
    {
        platelet plt = plateletList[current];
        long long int loc = (long long int)ny * nz * plt.center[0] +
                            (long long int)nz * plt.center[1] + (long long int)plt.center[2];
        if (adpMap.find(loc) != adpMap.end())
        {
            plt.concentrations[0] = adpMap[loc];
        }
        if (txa2Map.find(loc) != txa2Map.end())
        {
            plt.concentrations[3] = txa2Map[loc];
        }

        plateletList[current] = plt;
    }

    MPI_File_open(MPI_COMM_SELF, filevelName.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &filevel);
    MPI_File_open(MPI_COMM_SELF, fileshearName.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fileshear);
    MPI_File_open(MPI_COMM_SELF, fileflagName.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fileflag);

    pcout << "KMC/NN received updated velocity field from LB" << endl;

    ++iter;
}
