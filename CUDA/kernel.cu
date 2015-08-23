#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>

#define N 100
#define RAND_MAX 100
#define TILE_WIDTH 2

void matrixMultiplicationCPU(int* inputA, int* inputB, int* output)
{
	int i, j, k;

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k < N; k++)
				output[i * N + j] = output[i * N + j] + inputA[i * N + k] * inputB[k * N + j];
}

__global__ void matrixMultiplicationGPU(int *inputA, int *inputB, int *output, int size)
{
	int i, sum = 0;
	int columns = threadIdx.x + blockDim.x * blockIdx.x;
	int rows = threadIdx.y + blockDim.y * blockIdx.y;

	if (columns < size && rows < size)
	{
		for (i = 0; i < size; i++)
			sum += inputA[rows * size + i] * inputB[i * size + columns];
		output[rows * size + columns] = sum;
	}
}

__global__ void matrixMultiplicationGPUSharedMemeory(int *inputA, int *inputB, int *output, int size)
{
	__shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ int Nds[TILE_WIDTH][TILE_WIDTH];
	int bIdX = blockIdx.x;
	int bIdY = blockIdx.y;
	int tIdX = threadIdx.x;
	int tIdY = threadIdx.y;
	int row = bIdY * TILE_WIDTH + tIdY;
	int column = bIdX * TILE_WIDTH + tIdX;
	int i, j, sum = 0;
	
	for (i = 0; i < size / TILE_WIDTH; ++i)
	{
		Mds[tIdY][tIdX] = inputA[row * size + (i * TILE_WIDTH + tIdX)];
		Nds[tIdY][tIdX] = inputB[(i * TILE_WIDTH + tIdY) * size + column];
		__syncthreads();
		for (j = 0; j < TILE_WIDTH; ++j)
			sum += Mds[tIdY][j] * Nds[j][tIdX];
		__syncthreads();
	}
	output[row * size + column] = sum;
}

int* generateArray(int count)
{
	int *array;
	array = (int*)calloc(count,sizeof(int));
	srand(time(NULL));
	for (int i = 0; i < count; i++)
	{
		(array)[i] = rand() / RAND_MAX;
	}
	return array;
}

int main() {
	int *a, *b;

	a = generateArray(N*N);
	b = generateArray(N*N);

	int* c;
	c = (int*)calloc(N*N, sizeof(int));

	for (int i = 0; i < N*N; i++)
		c[i] = 0;


	
	int i, j;
	int *dev_a, *dev_b, *dev_c;
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	LARGE_INTEGER frequency;
	LARGE_INTEGER startCPU;
	LARGE_INTEGER endCPU;
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&startCPU);

	matrixMultiplicationCPU(a, b, c);
	
	QueryPerformanceCounter(&endCPU);
	
	printf("matrixMultiplicationCPU\n");
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++)
		{
		//	printf("%d\t", c[i*N+j]);
		}
		//printf("\n");
	}
	printf("time %f ms\n", ((double)(endCPU.QuadPart - startCPU.QuadPart) / frequency.QuadPart) * 1000);

	int size = N * N * sizeof(int);
	cudaMalloc((void **)&dev_a, size);
	cudaMalloc((void **)&dev_b, size);
	cudaMalloc((void **)&dev_c, size);

	cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

	//dim for shared
	taki jest wynik dim3 dimGrid(N / TILE_WIDTH, N / TILE_WIDTH);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

	//dim non shared
	//dim3 dimBlock(N, N);
	//dim3 dimGrid(1, 1);
	//dim3 dimGrid((int)ceil(N/dimBlock.x),(int)ceil(N/dimBlock.y));

	int blok = (int)ceil(N / dimBlock.x);
	int watek = (int)ceil(N / dimBlock.y);

	printf("Blok\n%d X\t%d Y\n\n", dimBlock.x, dimBlock.y);
	printf("Grid\n%d Blok\t%d Watki\n", blok, watek);

	cudaEventRecord(start, 0);

	//matrixMultiplicationGPU << <dimGrid, dimBlock >> >(dev_a, dev_b, dev_c, N);
	matrixMultiplicationGPUSharedMemeory <<<dimGrid, dimBlock >>>(dev_a, dev_b, dev_c, N);

	cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	printf("\n\matrixMultiplicationGPU\n");
	for (i = 0; i < N; ++i){
		for (j = 0; j < N; ++j)
		{
		//	printf("%d\t", c[i*N+j]);
		}
		//printf("\n");
	}
	printf("time %g ms\n", time);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	system("PAUSE");
	return 0;
}