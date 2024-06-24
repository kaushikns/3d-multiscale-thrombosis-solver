/*--------------------------------------------------------------------------*\
    Author: Kaushik N. Shankar
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "fvOptions.H"
#include "simpleControl.H"
#include "dynamicFvMesh.H"
#include "meshSearch.H"
#include "mui.h"
#include <array>
#include "PstreamGlobals.H"
#include <set>

using namespace mui;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

std::string fileflagName;
std::string filevelName;
MPI_File fileflag;
MPI_File filevel;
long int nxlb, nylb, nzlb;
std::array<double, 3> LBLocation;
double dx, dt;
double fvmTime = 0.;

// Reads the directory locations where velocity output files from palabos will be stored
void getDirectoryInfo();

struct flagMatrix
{
    int val;
    flagMatrix();
    int get(long int px, long int py, long int pz);
};

struct platelet
{
    double val;
    int rank;
};

void defineMPIStruct(MPI_Datatype &type)
{
    const int nitems = 7;
    int blocklengths[nitems] = {1, 1, 1, 1, 1, 1, 1};
    MPI_Datatype types[nitems] = {
        MPI_DOUBLE,
        MPI_DOUBLE,
        MPI_DOUBLE,
        MPI_DOUBLE,
        MPI_DOUBLE,
        MPI_LONG_LONG,
        MPI_INT};
    MPI_Aint offsets[nitems];
    offsets[0] = offsetof(struct sendplatelet, px);
    offsets[1] = offsetof(struct sendplatelet, py);
    offsets[2] = offsetof(struct sendplatelet, pz);
    offsets[3] = offsetof(struct sendplatelet, release);
    offsets[4] = offsetof(struct sendplatelet, calcInteg);
    offsets[5] = offsetof(struct sendplatelet, loc);
    offsets[6] = offsetof(struct sendplatelet, rank);
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &type);
    MPI_Type_commit(&type);
}

MPI_Datatype mpi_sendplatelet;

flagMatrix flagMatrixInside;

int procid = 0;

// Reads velocity from the file output by Palabos
std::array<double, 3> readVelocity(long int px, long int py, long int pz);

// Peforms trilinear interpolation of velocity at a given location in lattice units
std::array<double, 3> interpolateVelocity(double px, double py, double pz);

int main(int argc, char *argv[])
{
    argList::addNote(
        "Passive scalar transport equation solver.");

#include "setRootCaseLists.H"
#include "createTime.H"
#include "createDynamicFvMesh.H"
#include "createFields.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info << "\n Calculating scalar transport\n"
         << endl;

#include "CourantNo.H"

    simpleControl simple(mesh);

    if (argc == 1)
        mui::mpi_split_by_app();

    if (UPstream::parRun())
        MPI_Comm_rank(PstreamGlobals::MPI_COMM_FOAM, &procid);

    defineMPIStruct(mpi_sendplatelet);

    // Init of MUI interface
    std::string dom = "fvm";
    std::vector<std::string> interfaces;
    interfaces.emplace_back("comma/ifs");
    interfaces.emplace_back("commb/ifs");
    interfaces.emplace_back("commc/ifs");

    auto ifs = mui::create_uniface<mui::one_dim>(dom, interfaces);

    getDirectoryInfo();

    double h = 6.e-6;

    mui::chrono_sampler_exact<mui::one_dim> chrono_sampler;

    mui::point1d push_point;

    mui::point1d fetch_point;

    long int iter = 0;

    // Pre-exponenetial for source term
    double adp_pre_exponential = 2.5e-17; // ADP
    double adp_time_const = 5.;
    adp_pre_exponential = adp_pre_exponential / adp_time_const;
    double txa2_pre_exponential = 1.e-18; // TXA2
    double txa2_time_const = 100.;
    txa2_pre_exponential = txa2_pre_exponential / txa2_time_const;

    double dtLB = 0.1;
    double dtFVM = runTime.deltaTValue();
    long int factor = (long int)(dtLB / dtFVM);

    long int count = 0;

    while (runTime.run())
    {
        Info << "Time = " << runTime.timeName() << endl;
        runTime++;
        Info << "FVM run time = " << fvmTime << endl;

        if (count % factor == 0)
        {
            // Get LB params to do velocity interpolation when needed
            if (iter == 0)
            {
                auto resolutionx = ifs[0]->fetch_values<double>("resolutionx", iter, chrono_sampler);
                nxlb = (long int)resolutionx[0];
                auto resolutiony = ifs[0]->fetch_values<double>("resolutiony", iter, chrono_sampler);
                nylb = (long int)resolutiony[0];
                auto resolutionz = ifs[0]->fetch_values<double>("resolutionz", iter, chrono_sampler);
                nzlb = (long int)resolutionz[0];
                auto dx_ = ifs[0]->fetch_values<double>("dx", iter, chrono_sampler);
                dx = dx_[0];
                auto dt_ = ifs[0]->fetch_values<double>("dt", iter, chrono_sampler);
                dt = dt_[0];
                auto LBLocx = ifs[0]->fetch_values<double>("locationx", iter, chrono_sampler);
                LBLocation[0] = LBLocx[0];
                auto LBLocy = ifs[0]->fetch_values<double>("locationy", iter, chrono_sampler);
                LBLocation[1] = LBLocy[0];
                auto LBLocz = ifs[0]->fetch_values<double>("locationz", iter, chrono_sampler);
                LBLocation[2] = LBLocz[0];
            }

            // Get list of platelet centers and release times from KMC

            auto plts = ifs[0]->fetch_values<sendplatelet>("plt", iter, chrono_sampler);

            ifs[0]->forget(iter);

            if (UPstream::parRun())
            {
                int sz = plts.size();
                MPI_Bcast(plts.data(), sz, mpi_sendplatelet, 0, PstreamGlobals::MPI_COMM_FOAM);
            }

            Info << "Number of platelets that KMC needs information for: " << plts.size() << endl;

            isPlatelet = dimensionedScalar("zeroP", Foam::dimensionSet(scalar(0), scalar(0), scalar(0), scalar(0), scalar(0), scalar(0), scalar(0)), scalar(0.));

            double currentTime = runTime.value();

            meshSearch ms(mesh);
            ms.tol_ = 3.e-6;

            std::vector<double> localAdp{};
            std::vector<double> localTxa2{};
            localAdp.resize(plts.size());
            localTxa2.resize(plts.size());

            std::vector<platelet> localplt{};
            std::vector<platelet> globalplt{};
            localplt.resize(plts.size());
            globalplt.resize(plts.size());

            std::vector<label> cellind{};
            cellind.resize(plts.size());

            std::set<long long> locs_{};

            for (long int i = 0; i < plts.size(); ++i)
            {
                Foam::point p(plts[i].px, plts[i].py, plts[i].pz);

                label celli = ms.findNearestCell(p, -1, true);

                double px = mesh.C()[celli].x() - plts[i].px;
                double py = mesh.C()[celli].y() - plts[i].py;
                double pz = mesh.C()[celli].z() - plts[i].pz;

                localplt[i].val = std::sqrt(px * px + py * py + pz * pz);
                localplt[i].rank = procid;
                globalplt[i].rank = procid;

                localAdp[i] = adpC[celli];
                localTxa2[i] = txa2C[celli];

                cellind[i] = celli;
            }

            int nPlatelets = plts.size();

            if (UPstream::parRun())
            {
                MPI_Allreduce(localplt.data(), globalplt.data(), nPlatelets, MPI_DOUBLE_INT, MPI_MINLOC, PstreamGlobals::MPI_COMM_FOAM);
            }

            for (long int i = 0; i < plts.size(); ++i)
            {
                if (plts[i].calcInteg > 0.005 && procid == globalplt[i].rank)
                {
                    isPlatelet[cellind[i]] += 1;
                    if (plts[i].release > 0. && (currentTime - plts[i].release) > 10.)
                    {
                        isPlatelet[cellind[i]] -= 1;
                    }
                }

                push_point[0] = plts[i].loc;

                if (procid == globalplt[i].rank)
                {
                    mui::recvplatelet plt;
                    plt.adpC = localAdp[i];
                    plt.txa2C = localTxa2[i];
                    plt.loc = plts[i].loc;
                    plt.rank = plts[i].rank;
                    ifs[2]->push("plt", push_point, plt);
                }
            }

            ifs[2]->commit(iter);
            ifs[2]->forget(iter - 1);

            // Start mesh refinement
            // Do any mesh changes
            if (iter > 0)
            {
                mesh.update();
            }
            adpR = dimensionedScalar("zeroadpR", Foam::dimensionSet(scalar(0), scalar(-3), scalar(-1), scalar(0), scalar(1), scalar(0), scalar(0)), scalar(0.));

            txa2R = dimensionedScalar("zerotxa2R", Foam::dimensionSet(scalar(0), scalar(-3), scalar(-1), scalar(0), scalar(1), scalar(0), scalar(0)), scalar(0.));

            meshSearch ms_(mesh);
            ms_.tol_ = 3.e-6;

            // Set source terms based on platelet centers, activation states and release times
            for (long int i = 0; i < plts.size(); ++i)
            {
                if (procid != globalplt[i].rank)
                {
                    continue;
                }
                if (currentTime >= plts[i].release && plts[i].release > 0.)
                {
                    Foam::point p(plts[i].px, plts[i].py, plts[i].pz);
                    label celli = ms_.findNearestCell(p, -1, true);

                    if (mesh.V()[celli] < 1.e-21)
                    {
                        continue;
                    }

                    adpR[celli] += adp_pre_exponential * std::exp((plts[i].release - currentTime) / adp_time_const) / mesh.V()[celli];
                    txa2R[celli] += txa2_pre_exponential * std::exp((plts[i].release - currentTime) / txa2_time_const) / mesh.V()[celli];

                    if (std::isnan(adpR[celli]))
                    {
                        adpR[celli] = 0.;
                    }
                    if (std::isnan(txa2R[celli]))
                    {
                        txa2R[celli] = 0.;
                    }
                    if (adpC[celli] < 0.)
                    {
                        adpC[celli] = 0.;
                    }
                    if (txa2C[celli] < 0.)
                    {
                        txa2C[celli] = 0.;
                    }
                }
            }

            auto sig = ifs[1]->fetch_values<double>("signal", iter, chrono_sampler);
            ifs[1]->forget(iter);

            MPI_File_open(MPI_COMM_SELF, filevelName.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &filevel);
            MPI_File_open(MPI_COMM_SELF, fileflagName.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fileflag);

            forAll(mesh.C(), celli)
            {
                push_point[0] = celli;
                double px = mesh.C()[celli].x();
                double py = mesh.C()[celli].y();
                double pz = mesh.C()[celli].z();
                px = (px - LBLocation[0]) / dx;
                py = (py - LBLocation[1]) / dx;
                pz = (pz - LBLocation[2]) / dx;
                auto vel = interpolateVelocity(px, py, pz);
                U[celli].x() = vel[0];
                U[celli].y() = vel[1];
                U[celli].z() = vel[2];
            }

            MPI_File_close(&filevel);
            MPI_File_close(&fileflag);

            Info << "OpenFOAM received velocity field from LB" << endl;

            iter++;

            U.correctBoundaryConditions();
            phi = fvc::flux(U);
        }

        fvmTime -= MPI_Wtime();
        while (simple.correctNonOrthogonal())
        {
            fvScalarMatrix adpEqn(
                fvm::ddt(adpC) + fvm::div(phi, adpC) - fvm::laplacian(adpD, adpC) ==
                adpR // Source term added
            );

            adpEqn.solve();

            fvScalarMatrix txa2Eqn(
                fvm::ddt(txa2C) + fvm::div(phi, txa2C) - fvm::laplacian(txa2D, txa2C) ==
                txa2R // Source term added
            );

            txa2Eqn.solve();
        }

        count++;

        runTime.write();

        fvmTime += MPI_Wtime();
    }

    MPI_Type_free(&mpi_sendplatelet);

    Info << "End\n"
         << endl;

    return 0;
}

// Reads the directory locations where velocity output files from palabos will be stored
void getDirectoryInfo()
{
    std::string line;
    ifstream file("directoryinfo.txt");
    getline(file, line);
    filevelName = line;
    getline(file, line);
    fileflagName = line;
    file.close();
    fileflagName = fileflagName + ".dat";
    filevelName = filevelName + ".dat";
    return;
}

std::array<double, 3> readVelocity(long int px, long int py, long int pz)
{
    MPI_Status stat;
    MPI_Offset offset;
    long int loc = pz + py * nzlb + px * nzlb * nylb;
    offset = loc * sizeof(double) * 3;
    std::array<double, 3> u;
    MPI_File_read_at(filevel, offset, &u, 3, MPI_DOUBLE, &stat);
    return u;
}

std::array<double, 3> interpolateVelocity(double px, double py, double pz)
{
    std::array<double, 3> velocity;
    velocity[0] = 0.;
    velocity[1] = 0.;
    velocity[2] = 0.;
    long int x0, y0, z0, x1, y1, z1;
    x0 = (long int)floor(px);
    y0 = (long int)floor(py);
    z0 = (long int)floor(pz);
    x1 = x0 + 1;
    y1 = y0 + 1;
    z1 = z0 + 1;
    double xd, yd, zd;
    xd = px - (double)x0;
    yd = py - (double)y0;
    zd = pz - (double)z0;

    auto v000 = readVelocity(x0, y0, z0);
    auto v001 = readVelocity(x0, y0, z1);
    auto v010 = readVelocity(x0, y1, z0);
    auto v011 = readVelocity(x0, y1, z1);
    auto v100 = readVelocity(x1, y0, z0);
    auto v101 = readVelocity(x1, y0, z1);
    auto v110 = readVelocity(x1, y1, z0);
    auto v111 = readVelocity(x1, y1, z1);

    std::array<double, 3> v00, v01, v10, v11, v0, v1;

    for (int i = 0; i < 3; ++i)
    {
        v00[i] = (1. - xd) * v000[i] + xd * v100[i];
        v01[i] = (1. - xd) * v001[i] + xd * v101[i];
        v10[i] = (1. - xd) * v010[i] + xd * v110[i];
        v11[i] = (1. - xd) * v011[i] + xd * v111[i];
    }

    for (int i = 0; i < 3; ++i)
    {
        v0[i] = (1. - yd) * v00[i] + yd * v10[i];
        v1[i] = (1. - yd) * v01[i] + yd * v11[i];
    }

    for (int i = 0; i < 3; ++i)
    {
        velocity[i] = (dx / dt) * ((1. - zd) * v0[i] + zd * v1[i]);
    }

    return velocity;
}

int flagMatrix::get(long int px, long int py, long int pz)
{
    MPI_Status stat;
    MPI_Offset offset;
    long int loc = pz + py * nzlb + px * nzlb * nylb;
    offset = loc * sizeof(int);
    MPI_File_read_at(fileflag, offset, &val, 1, MPI_INT, &stat);
    return val;
}

flagMatrix::flagMatrix()
{
    val = 0;
    return;
}

// ************************************************************************* //
