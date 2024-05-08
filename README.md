## 3d-multiscale-thrombosis-solver

Thrombus growth model decribed in the publications:

1. Shankar KN, Diamond SL, Sinno T. Development of a parallel multiscale 3D model for thrombus growth under flow. Frontiers in Physics. 2023 Sep 5;11:1256462.
2. Shankar KN, Zhang Y, Sinno T, Diamond SL. A three-dimensional multiscale model for the prediction of thrombus growth under flow with single-platelet resolution. PLoS computational biology. 2022 Jan 28;18(1):e1009850.

#### Code
The framework includes two main programs, each residing in their respective directories along with their makefiles, located in 
(i) `pkmcnn-palabos-openfoam/pkmcnnlb`, this program can be compiled using make and is designed to be run using MPI MPMD launch mode.
(ii) `pkmcnn-palabos-openfoam/pkmcnnlb/fvm/agonistTransportDyMFoam`, this program should be compiled using wmake and is also intended for execution using MPI MPMD launch mode.

#### Prerequisites
Before compiling and running the programs, ensure the following prerequisites are met:
Modified versions of the Palabos, OpenFOAM, and MUI libraries are provided in the repository. Update the directory paths for these libraries in the respective Makefiles and code files as needed. Additionally, the OpenFOAM package needs to be compiled and the OpenFOAM environment needs to be loaded. 

#### Input Parameters
The `param.xml` file in the `case` directory contains a list of input simulation parameters that can be changed depending on the example being studied. 
An example meshed stenotic geometry is provided in the `case` directory.

The geometries directory contains stl files of some geometries that can be utilized for simulations. Each geometry must be appropriately meshed, e.g., using `snappyHexMesh`, and the `case` directory needs to be updated accordingly. Additionally, the stl file input is also needed in the `param.xml` file. 

#### Usage
To run the example, `cd` to `case` and run:
````mpirun -np n1 pkmcnn-palabos-openfoam/pkmcnnlb : -np n2 agonistTransportDyMFoam #-parallel ````

#### Output 
The simulations output *.csv files that contain platelet locations and activation levels after different time intervals, *.vti files that have velocity profiles, and 'time' directories containing agonist concentration field data. These outputs can all be visualized using Paraview. 

#### Note
Ensure all dependencies are properly configured and paths are updated before compiling and running. For any further assistance or inquiries, please contact me.
