################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/linearSolverFactorizedSLUutils.cu \
../src/set_pointer.cu \
../src/strsv_batched.cu \
../src/tinySLUfactorization_batched.cu 

CPP_SRCS += \
../src/linearDecompSLU_batched.cpp \
../src/linearSolverFactorizedSLU_batched.cpp \
../src/linearSolverLU_batched.cpp \
../src/testing_sgesv_batched.cpp \
../src/utils.cpp 

OBJS += \
./src/linearDecompSLU_batched.o \
./src/linearSolverFactorizedSLU_batched.o \
./src/linearSolverFactorizedSLUutils.o \
./src/linearSolverLU_batched.o \
./src/set_pointer.o \
./src/strsv_batched.o \
./src/testing_sgesv_batched.o \
./src/tinySLUfactorization_batched.o \
./src/utils.o 

CU_DEPS += \
./src/linearSolverFactorizedSLUutils.d \
./src/set_pointer.d \
./src/strsv_batched.d \
./src/tinySLUfactorization_batched.d 

CPP_DEPS += \
./src/linearDecompSLU_batched.d \
./src/linearSolverFactorizedSLU_batched.d \
./src/linearSolverLU_batched.d \
./src/testing_sgesv_batched.d \
./src/utils.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -O3 -gencode arch=compute_37,code=sm_37 --ptxas-options=-v --include-path /cineca/prod/opt/libraries/lapack/3.8.0/intel--pe-xe-2018--binary/include/  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -O3 --compile --ptxas-options=-v --include-path /cineca/prod/opt/libraries/lapack/3.8.0/intel--pe-xe-2018--binary/include/  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -O3 -gencode arch=compute_37,code=sm_37 --ptxas-options=-v --include-path /cineca/prod/opt/libraries/lapack/3.8.0/intel--pe-xe-2018--binary/include/  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -O3 --compile --ptxas-options=-v --include-path /cineca/prod/opt/libraries/lapack/3.8.0/intel--pe-xe-2018--binary/include/  --relocatable-device-code=true -gencode arch=compute_37,code=compute_37 -gencode arch=compute_37,code=sm_37  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

#lapack
#--include-path /cineca/prod/opt/libraries/lapack/3.8.0/intel--pe-xe-2018--binary/include/
