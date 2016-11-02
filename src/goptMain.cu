/*
 * basic genetic optimization in CUDA
 */

#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>



// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "individual.h"

/* this GPU kernel function is used to initialize the random states */

__global__ void init(unsigned int seed, Individual *I,
		int numElements)
		{
	int c = blockDim.x * blockIdx.x + threadIdx.x;

	if (c < numElements) {
		/* we have to initialize the state */
		curand_init(seed, blockIdx.x,

		0, &I[c].state);
	}
}

__global__ void shake(Individual *I, int chromosomeLength, int numElements) {
	int c = blockDim.x * blockIdx.x + threadIdx.x;

	if (c < numElements) {
		curandState_t* state = &I[c].state;
		for (int m = 0; m < 1; m++) {
			int i = curand(state) % chromosomeLength;
			int j = curand(state) % chromosomeLength;
			int temp = I[c].chromosome[i];
			I[c].chromosome[i] = I[c].chromosome[j];
			I[c].chromosome[j] = temp;
		}

	}

}

__global__ void mutateAndScore(Individual *I,
		bool justMutate, int chromosomeLength, int numElements) {
	int c = blockDim.x * blockIdx.x + threadIdx.x;

	if (c < numElements) {
		curandState_t* state = &I[c].state;
		int i = curand(state) % chromosomeLength;
		int j = curand(state) % chromosomeLength;
		int temp = I[c].chromosome[i];
		I[c].chromosome[i] = I[c].chromosome[j];
		I[c].chromosome[j] = temp;
		if (!justMutate) {

			I[c].fitness = c;
		}
	}

}

void initGPURandoms(int num, Individual *I, int numElements) {
	/*
	 * For each core initialize a random state, and return this block of states
	 */
	curandState_t* states;
	cudaError_t err = cudaSuccess;

	/* allocate space on the GPU for the random states */
	cudaMalloc((void**) &states, num * sizeof(curandState_t));

	/* invoke the GPU to initialize all of the random states */
	init<<<num, 1>>>(time(0), I, numElements);

	err = cudaGetLastError();

	if (err != cudaSuccess) {
		fprintf(stderr, "failed to init random states (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

}

void printPopulation(Individual *list, int n, int l) {
	for (int i = 0; i < n; ++i) {
		fprintf(stdout, "%d  %lf ", list[i].id, list[i].fitness);
		for (int j = 0; j < l; j++) {
			fprintf(stdout, "%d ", list[i].chromosome[j]);

		}
		fprintf(stdout, "\n");

	}
}
int main(void) {
// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

// Print the vector length to be used, and compute its size
	int chromosomeLength = 128;
	int numIndividuals = 10;

	size_t individualSize = sizeof(struct Individual);
	size_t allIndividualsSize = individualSize * numIndividuals;

	struct Individual *h_I = (struct Individual *) malloc(allIndividualsSize);

// Verify that allocations succeeded
	if (h_I == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	fprintf(stdout, "Allocated %d individuals for a population size of %u\n",
			numIndividuals, allIndividualsSize);

// Initialize the host input vectors
	for (int i = 0; i < numIndividuals; ++i) {
		h_I[i].fitness = -1;
		h_I[i].id = i;
		for (int j = 0; j < chromosomeLength; j++) {
			h_I[i].chromosome[j] = j;
		}

	}

	printPopulation(h_I, 3, 10);
// Allocate the device output vector C
	Individual *d_I = NULL;
	err = cudaMalloc((void **) &d_I, allIndividualsSize);

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

// Copy the host input vectors A and B in host memory to the device input vectors in
// device memory
	printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_I, h_I, allIndividualsSize, cudaMemcpyHostToDevice);

	if (err != cudaSuccess) {
		fprintf(stderr,
				"Failed to copy vector A from host to device (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (numIndividuals + threadsPerBlock - 1)
			/ threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
			threadsPerBlock);

	initGPURandoms(blocksPerGrid, d_I, numIndividuals);
	shake<<<blocksPerGrid, threadsPerBlock>>>(d_I, chromosomeLength,
			numIndividuals);

	mutateAndScore<<<blocksPerGrid, threadsPerBlock>>>( d_I, true,
			chromosomeLength, numIndividuals);

	err = cudaGetLastError();

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

// Copy the device result vector in device memory to the host result vector
// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_I, d_I, allIndividualsSize, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess) {
		fprintf(stderr,
				"Failed to copy vector C from device to host (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printPopulation(h_I, 3, 10);

	printf("Test PASSED\n");

// Free device global memory
	err = cudaFree(d_I);

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

// Free host memory
	free(h_I);

	printf("Done\n");
	return 0;
}

