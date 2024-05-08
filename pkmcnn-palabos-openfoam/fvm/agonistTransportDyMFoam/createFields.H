volScalarField adpC
(
    IOobject
    (
        "adpC",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
	    IOobject::AUTO_WRITE
    ),
    mesh
);

volScalarField txa2C
(
    IOobject
    (
        "txa2C",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
	    IOobject::AUTO_WRITE
    ),
    mesh
);

volScalarField thrombinC
(
    IOobject
    (
        "thrombinC",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
	    IOobject::AUTO_WRITE
    ),
    mesh
);

volScalarField adpR
(
    IOobject
    (
        "adpR",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE

    ),
    mesh,
    dimensionedScalar("adpR", Foam::dimensionSet( scalar(0), scalar(-3), scalar(-1), scalar(0), scalar(1),
     scalar(0), scalar(0) ), scalar(0.))
);

volScalarField txa2R
(
    IOobject
    (
        "txa2R",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE

    ),
    mesh,
    dimensionedScalar("txa2R", Foam::dimensionSet( scalar(0), scalar(-3), scalar(-1), scalar(0), scalar(1),
     scalar(0), scalar(0) ), scalar(0.))
);

volScalarField isPlatelet
(
    IOobject
    (
        "isPlatelet",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE

    ),
    mesh,
    dimensionedScalar("isPlatelet", Foam::dimensionSet( scalar(0), scalar(0), scalar(0), scalar(0), scalar(0),
     scalar(0), scalar(0) ), scalar(0.))
);

volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


IOdictionary transportProperties
(
    IOobject
    (
        "transportProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ_IF_MODIFIED,
        IOobject::NO_WRITE
    )
);


dimensionedScalar adpD
(
	transportProperties.lookup("adpD")
);

dimensionedScalar txa2D
(
	transportProperties.lookup("txa2D")
);

dimensionedScalar thrombinD
(
	transportProperties.lookup("thrombinD")
);

#include "createPhi.H"

#include "createFvOptions.H"
