/*--------------------------------------------------------------------------*\
    Author: Kaushik N. Shankar
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "fvOptions.H"
#include "simpleControl.H"
#include "dynamicFvMesh.H"
#include "meshSearch.H"
#include "mui.h"

using namespace mui;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    MPI_Comm world = mui::mpi_split_by_app();
    // Init of MUI interface
    std::string dom = "fvm";
    std::vector<std::string> interfaces;
    interfaces.emplace_back("mpi://kmc_fvm/ifs");
    interfaces.emplace_back("mpi://fvm_nn/ifs");
    interfaces.emplace_back("mpi://fvm_lb/ifs");
    interfaces.emplace_back("mpi://lb_fvm/ifs");
    auto ifs = mui::create_uniface<mui::config_1d>(dom, interfaces);

    mui::chrono_sampler_exact1d chrono_sampler;
    mui::sampler_exact1d<double> spatial_sampler;

    argList::addNote
    (
        "Passive scalar transport equation solver."
    );

    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createDynamicFvMesh.H"

    simpleControl simple(mesh);

    #include "createFields.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\n Calculating scalar transport\n" << endl;

    #include "CourantNo.H"

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
    long int factor = (long int) (dtLB/dtFVM);

    long int count = 0;

    meshSearch ms(mesh);

    while (runTime.run())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;
        runTime++;

	
        if (count % factor == 0)
        {
            // Get list of platelet centers and release times from KMC 
            auto pltCenterx = ifs[0] -> fetch_values<double>("px", iter, chrono_sampler);
            auto pltCentery = ifs[0] -> fetch_values<double>("py", iter, chrono_sampler);
            auto pltCenterz = ifs[0] -> fetch_values<double>("pz", iter, chrono_sampler); 
            auto releaseTimes = ifs[0] -> fetch_values<double>("release", iter, chrono_sampler);
	    auto calcInteg = ifs[0] -> fetch_values<double>("calcinteg", iter, chrono_sampler);
	    ifs[0] -> forget(iter);


            isPlatelet = dimensionedScalar("zeroP", Foam::dimensionSet( scalar(0), scalar(0), scalar(0), 
                scalar(0), scalar(0), scalar(0), scalar(0) ), scalar(0.));

            double localAdp, localTxa2, localThrombin;
	        double currentTime = runTime.value();

            for (long int i = 0; i < pltCenterx.size(); ++i)
            {
                Foam::point p(pltCenterx[i], pltCentery[i], pltCenterz[i]);
                label celli = ms.findNearestCell(p, 0, true);

                if (calcInteg[i] > 0.005)
                {
                    isPlatelet[celli] = 1;
		    if (releaseTimes[i] > 0. && (currentTime - releaseTimes[i]) > 10.)
		    {
			isPlatelet[celli] = 0.;
		    } 
                }
                localAdp = adpC[celli];
                localTxa2 = txa2C[celli];
                localThrombin = thrombinC[celli];

                push_point[0] = i;
                ifs[1] -> push("adp", push_point, localAdp);
                ifs[1] -> push("txa2", push_point, localTxa2);
                ifs[1] -> push("thrombin", push_point, localThrombin);
            }
	
	    ifs[1] -> commit(iter);
            ifs[1] -> forget(iter-1);

            // Do any mesh changes
            mesh.update();

            adpR = dimensionedScalar("zeroadpR", Foam::dimensionSet( scalar(0), scalar(-3), scalar(-1), scalar(0), 
                scalar(1), scalar(0), scalar(0) ), scalar(0.));

            txa2R = dimensionedScalar("zerotxa2R", Foam::dimensionSet( scalar(0), scalar(-3), scalar(-1), scalar(0), 
                scalar(1), scalar(0), scalar(0) ), scalar(0.));


            // Set source terms based on platelet centers, activation states and release times
            for (long int i = 0; i < pltCenterx.size(); ++i)
            {
                if (currentTime >= releaseTimes[i] && releaseTimes[i] > 0.)
                {
                    Foam::point p(pltCenterx[i], pltCentery[i], pltCenterz[i]);
                    label celli = ms.findNearestCell(p, 0, true);
                    adpR[celli] = adp_pre_exponential * std::exp ( (
                        releaseTimes[i] - currentTime ) / adp_time_const) / mesh.V()[celli];
                    txa2R[celli] = txa2_pre_exponential * std::exp ( (
                        releaseTimes[i] - currentTime ) / txa2_time_const) / mesh.V()[celli];
                }
            }

            forAll(mesh.C(), celli)
            {
                push_point[0] = celli;
                ifs[2] -> push("ofgridx", push_point, mesh.C()[celli].x() );
                ifs[2] -> push("ofgridy", push_point, mesh.C()[celli].y() );
                ifs[2] -> push("ofgridz", push_point, mesh.C()[celli].z() );
            }

            ifs[2] -> commit(iter);
            ifs[2] -> forget(iter-1);

            Info << "Pushed OpenFOAM grid points to MUI interface" << endl;

	        ifs[3] -> barrier(iter);

            auto ux = ifs[3] -> fetch_values<double>("ux", iter, chrono_sampler);
            auto uy = ifs[3] -> fetch_values<double>("uy", iter, chrono_sampler);
            auto uz = ifs[3] -> fetch_values<double>("uz", iter, chrono_sampler);
            auto cellindices = ifs[3] -> fetch_points<double>("ux", iter, chrono_sampler); 

	        ifs[3] -> forget(iter);

            for(long int i = 0; i < ux.size(); ++i)
            {
                label celli = cellindices[i][0];
                U[celli].x() = ux[i];
                U[celli].y() = uy[i];
                U[celli].z() = uz[i];
            }

            iter++;

            U.correctBoundaryConditions();
	    phi = fvc::flux(U);
            fvc::makeAbsolute(phi,  U);
        }

        while (simple.correctNonOrthogonal())
        {
            fvScalarMatrix adpEqn
            (
                fvm::ddt(adpC)
                + fvm::div(phi, adpC)
                - fvm::laplacian(adpD, adpC)
                ==
                adpR // Source term added
            );

            adpEqn.solve();

            fvScalarMatrix txa2Eqn
            (
                fvm::ddt(txa2C)
                + fvm::div(phi, txa2C)
                - fvm::laplacian(txa2D, txa2C)
                ==
                txa2R // Source term added
            );

            txa2Eqn.solve();
        }

        count++;

        runTime.write();
    }

Info<< "End\n" << endl;

return 0;

}


// ************************************************************************* //
