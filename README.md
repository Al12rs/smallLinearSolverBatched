# GPU Linear Solver Small Batched

This project contains functions to solve large quantities of small square linear systems (NxN with N<32, single precision, dense), on GPU though the CUDA programming model. 

## Building
The software is written for Linux, built in Release mode though the use of make files (debug building is not supported).

The software requires a number of dependencies to be installed to build:
* gcc version 6.1.0+, lower versions might work but have not been tested.
* CUDA toolkit version 9 or more, lower might work but have not been testet. For optimal performance use CUDA 10+. 
  A guide on installing CUDA on linux is available here: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
  Particular care should be given to post installation actions https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions which are required to make the cuda compiler and libraries visible.
For the Tester part, and not for the actual function these are also needed (stubbing of code is needed to avoid these dependencies).
* Lapack 3.8.0+, lower versions might work but have not been tested.
* Blas 3.8.0+, lower versions might work but have not been tested.
* gfortran (if using lapack and blas)
* OpenMP, This can easily be avoided by commenting the few lines of code where it's used.


After the dependencies are installed the following files will need to be edited to match the system configurations:
* Release/objects.mk  Here the Lapack, blas and intel64(if using intel version of lapack and blas) library paths are to be redefined. In general the libraries are defined here, so if something changes regarding them this file needs to be edited accordgly.
* Release/makefile  This file needs to be edited to select the correct Nvidia target achitecture. Specifically in line 61, arch= and code= need to be changed according to these guidelines https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation .
  If OpenMP is to be removed then also remove the -Xcompiler -fopenmp options from the line (passes the -fopenmp line to the gcc compiler after nvcc completes compilation of kernels).
* Release/src/subdir.mk This file contains all the files that needs to be built and their dependencies, as well as the nvcc compilation invocations. Lapack linbrary paths need to be properly defined here. 
  Nvidia target architecture needs to be defined here as well. 

To build the program cd to Release and execute
`make clean`
followed by
`make`
The binary will show up as Release/gpulinearsolversmallbatched .

## Using the current tester
The binary can be compiled in two configurations: manual and automatic tester.
In manual mode usage is
`gpulinearsolversmallbatched <matrix size> <number of linear systems> <number of openMP threads for CPU test>`
eg:
`gpulinearsolversmallbatched 4 10000 8`

In automatic mode the tester will generate a file results.csv with all the test results. The test parameters need to be configured from code before build.

## Usage on Galileo supercomputer.
The program needs some modules to be loaded:
`module load autoload git cuda gnu lapack blas`
Clone the repository or copy it someway and cd to Release
`cd Release`
To build use
`make clean`
`make`
To run the program on the GPU nodes use the following command:
`srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --mem=6G --time=20 --partition=gll_usr_gpuprod --gres=gpu:kepler:1 ./gpulinearsolversmallbatched`
adding command line parameters at the end if in manual mode.
Change --cpus-per-task= to increase the number of phisical cpu cores available for openMP to use.
Change --mem= to ingrease the maximum usable memory (gpu is limited to a little less than 12G) and currently the program is limited by int memory pointers.
Change --time= to set the time limit in minutes of of the test. For autotester 20-30 minutes might be needed. For manual 3 minutes is already plenty. Mostly to ensure that the program does not go into infinite loop while on a work node.
Change --gres=gpu:kepler: to change the number of GPUs requested (in case of adding multi gpu support).

To peform profiling run add `cudaProfilerStart()` and `cudaProfilerStop()` before and after the interested section, rebuild and use the following command:
`srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --mem=6G --time=3 --partition=gll_usr_gpuprod --gres=gpu:kepler:1  nvprof --export-profile timeline.prof -f --profile-from-start off --cpu-profiling off  ./gpulinearsolversmallbatched 16 1000 1`
Which will export the profiling data to ./timeline.prof which can be imported into Visual Profile. Visual profile can import the file even if running on a windows machine (use scp to get the file).

## Code structure and configurations
`main` is situated in `testing_sgesv_batched.cpp`, where also all the testing code is situated.
This is the file that needs to be edited to remove lapack, blas and openmp dependencies. 
To change the mode to manual tester the macro MANUAL_TEST needds to be defined in testing_sgesv_batched.cpp and to use the automatic tester comment the macro definition.
Various macros with comments are present at the beginning of the file to change the manual test behavior.
The manual test is performed by the function `gpuLinearSolverBatched_tester` while the autotester is performed by `gpuCSVTester` which is the function that needs to be edited to change the parameters of the autotest.

To perform GPU solution the user needs prepare the linear systems in host memory and call `gpuLinearSolverBatched` which is in `linearSolverSLU_batched.cpp`. 
This function allocates the device memory and transfers the data to the call `linearSolverSLU_batched` (same file) wich takes pointers to device memory to start executing the different phases.
To perform LU decomposition the function `linearDecompSLU_batched` (in file `linearDecompSLU_batched.cpp`) is called, which in turn calls `magma_sgetrf_batched_smallsq_shfl` (`tinySLUfactorization_batched.cu`).
Only now `magma_sgetrf_batched_smallsq_shfl` executes the kernel  `sgetrf_batched_smallsq_shfl_kernel` which contains the main calculations of the program to do the LU factorization.
After the factorization is complete control returns to `linearSolverSLU_batched`which then calls `linearSolverFactorizedSLU_batched` (linearSolverFactorizedSLU_batched.cpp)
This function uses various inexpensive other functions to manipulate the factorized data and obtain the resolution to the linear systems, by using forwards and backwards substitutions. 
These functions are located in `set_pointer.cu`, `strsv_batched.cu`, `linearSolverFactorizedSLUutils.cu`  

For the remiaining files we have `operation_batched.h` which contains the declaration of most host batched functions listed above, `utils.cpp` `utilscu.cuh` `utils.h` contain utility functions that are used thoughought the code.
`testing.h`, `flops.h`, `magma_types.h` instead contain important magma definitions that are used throughout the code.



