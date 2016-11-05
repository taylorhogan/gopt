/*
 * basic genetic optimization in CUDA
 * Sorry, I do not know c
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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort =
		true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
				line);
		if (abort)
			exit(code);
	}
}

__global__ void initRandomVariableInChromosome(unsigned int seed, Individual *I,
		int numElements)
{

	int c = blockDim.x * blockIdx.x + threadIdx.x;

	if (c < numElements)
		curand_init(seed, c, 0, &(I[c].state));

}

__global__ void shake(Individual *I, int chromosomeLength, int numElements)
{
	int c = blockDim.x * blockIdx.x + threadIdx.x;

	if (c < numElements)
	{
		curandState_t* state = &I[c].state;
		for (int m = 0; m < chromosomeLength*2; m++)
		{
			int i = curand(state) % chromosomeLength;
			int j = curand(state) % chromosomeLength;

			long x = I[c].chromosome[i].x;
			long y = I[c].chromosome[i].y;
			I[c].chromosome[i].x = I[c].chromosome[j].x;
			I[c].chromosome[i].y = I[c].chromosome[j].y;
			I[c].chromosome[j].x=x;
			I[c].chromosome[j].y=y;


		}

	}

}

__global__ void fitness(Individual *I, int chromosomeLength, int numElements)
{

	int c = blockDim.x * blockIdx.x + threadIdx.x;

	if (c < numElements)
	{
		I[c].fitness = 0;
	}
}

__global__ void mutate(Individual *I, int chromosomeLength, int numElements)
{
	int c = blockDim.x * blockIdx.x + threadIdx.x;

	if (c < numElements)
	{
		curandState_t* state = &I[c].state;
		int i = curand(state) % chromosomeLength;
		int j = curand(state) % chromosomeLength;
		coord2D temp = I[c].chromosome[i];
		I[c].chromosome[i] = I[c].chromosome[j];
		I[c].chromosome[j] = temp;

	}

}

void printPopulation(Individual *list, int n, int l)
{
	for (int i = 0; i < n; ++i)
	{
		fprintf(stdout, "%d  %lf ", list[i].id, list[i].fitness);
		for (int j = 0; j < l; j++)
		{
			fprintf(stdout, "(%ld,%ld) ", list[i].chromosome[j].x,
					list[i].chromosome[j].y);
		}
		fprintf(stdout, "\n");
	}
}

int individualCompare(const void * a, const void * b)
{
	Individual *i0 = (Individual *) a;
	Individual *i1 = (Individual *) b;
	if (i1->fitness < i0->fitness)
		return -1;
	if (i0->fitness < i1->fitness)
		return 1;
	else
		return 0;
}

Individual *mergePopulations(Individual *current, Individual *next,
		int populationSize, size_t individualSize)
{
	struct Individual *total = (struct Individual *) malloc(populationSize * 2);

	memcpy(total, current, populationSize);
	memcpy(total + populationSize, next, populationSize);

	return total;
}

Individual* best(Individual *current, Individual *next, int populationSize,
		size_t individualSize)
{

	Individual *total = mergePopulations(current, next, populationSize,
			individualSize);

	qsort(total, populationSize * 2, individualSize, individualCompare);

	return total;

}

 void evolve (Individual *curGenerationLocal,
		Individual *curGenerationOnGPU, int chromosomeLength,
		int populationSize, size_t populationSizeInBytes, int blocksPerGrid,
		int threadsPerBlock, unsigned long generations)
{

	return;

/*
	for (unsigned long generation = 0; generation < generations; generation++)
	{
		fprintf(stdout, "1");
		// mutate population on GPU
		mutate<<<blocksPerGrid, threadsPerBlock>>>(curGenerationOnGPU,
				chromosomeLength, populationSize);
		fprintf(stdout, "2");
		// copy mutated population to local
		struct Individual *newGenerationLocal = (struct Individual *) malloc(
				populationSizeInBytes);
		fprintf(stdout, "3");
		gpuErrchk(
				cudaMemcpy(newGenerationLocal, curGenerationOnGPU,
						populationSizeInBytes, cudaMemcpyDeviceToHost));
		fprintf(stdout, "4");
		// find the best of the old and mutated
		Individual* nextGenerationLocal = best(curGenerationLocal,
				newGenerationLocal, populationSizeInBytes, populationSize);
		fprintf(stdout, "5");
		curGenerationLocal = newGenerationLocal;
		fprintf(stdout, "6");
		gpuErrchk(
				cudaMemcpy(curGenerationOnGPU, curGenerationLocal,
						populationSizeInBytes, cudaMemcpyHostToDevice));
		fprintf(stdout, "7");
	}

	*/

}

void makeFirstIndividuals(Individual *population, int poulationSize,
		int chromosomeLength)
{
	for (int i = 0; i < poulationSize; ++i)
	{
		population[i].fitness = -1;
		population[i].id = i;
		for (int j = 0; j < chromosomeLength; j++)
		{
			population[i].chromosome[j].x = j;
			population[i].chromosome[j].y = 0;
		}

	}
}
int main(void)
{
// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

// Print the vector length to be used, and compute its size
	int chromosomeLength = 128;
	int poulationSize = 10;

	size_t individualSizeInBytes = sizeof(struct Individual);
	size_t populationSizeInBytes = individualSizeInBytes * poulationSize;

	struct Individual *h_I = (struct Individual *) malloc(
			populationSizeInBytes );


	if (h_I == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}
	printf("allocated population size of %ul", populationSizeInBytes);


	fprintf(stdout, "Allocated %d individuals for a population size of %lu\n",
			poulationSize, (unsigned long) populationSizeInBytes);

// Initialize the host input vectors
	makeFirstIndividuals(h_I, poulationSize, chromosomeLength);

	printPopulation(h_I, 3, 10);

// Allocate the device output vector C
	Individual *d_I = NULL;
	err = cudaMalloc((void ** ) &d_I, populationSizeInBytes);
	if (err != cudaSuccess)
		{
			fprintf(stderr,
					"Failed to copy vector C from device to host (error code %s)!\n",
					cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}


// Copy the host input vectors A and B in host memory to the device input vectors in
// device memory
	printf("Copy input data from the host memory to the CUDA device\n");

	err = cudaMemcpy(d_I, h_I, populationSizeInBytes, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr,
				"Failed to copy vector C from device to host (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (poulationSize + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
			threadsPerBlock);


	evolve(h_I, d_I, chromosomeLength, poulationSize, populationSizeInBytes,
			blocksPerGrid, threadsPerBlock, 1);

	initRandomVariableInChromosome<<<blocksPerGrid, threadsPerBlock>>>(time(0),
	d_I, poulationSize);

	fprintf(stdout, "2");
	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr,
				"Failed to launch <<<initRandomVariableInChromosome>>> (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	shake<<<blocksPerGrid, threadsPerBlock>>>(d_I, chromosomeLength,
			poulationSize);
	gpuErrchk(cudaGetLastError());

	//evolve(h_I, d_I, chromosomeLength, poulationSize, populationSizeInBytes,
			//blocksPerGrid, threadsPerBlock, 1);


// Copy the device result vector in device memory to the host result vector
// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_I, d_I, populationSizeInBytes, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr,
				"Failed to copy vector C from device to host (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printPopulation(h_I, 3, 10);

	printf("Test PASSED\n");

// Free device global memory
	err = cudaFree(d_I);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

// Free host memory
	free(h_I);

	printf("Done\n");
	return 0;
}

