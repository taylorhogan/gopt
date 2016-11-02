################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/goptMain.cu \
../src/util.cu 

OBJS += \
./src/goptMain.o \
./src/util.o 

CU_DEPS += \
./src/goptMain.d \
./src/util.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.0/bin/nvcc -G -g -O0 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_35,code=sm_35  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.0/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_20,code=compute_20 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_32,code=sm_32  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


