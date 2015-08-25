#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include "cublas.h"
#include <math.h>
#include "cuda_runtime.h"

#define N 100
#define RAND_MAX 100
#define TILE_WIDTH 2

void matrixMultiplicationCPU(float* inputA, float* inputB, float* output)
{
	int i, j, k;

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k < N; k++)
				output[i * N + j] = output[i * N + j] + inputA[i * N + k] * inputB[k * N + j];
}

__global__ void matrixMultiplicationGPU(float *inputA, float *inputB, float *output, int size)
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

__global__ void matrixMultiplicationGPUSharedMemeory(float *inputA, float *inputB, float *output, int size)
{
	__shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ int Nds[TILE_WIDTH][TILE_WIDTH];

	int tIdX = threadIdx.x;
	int tIdY = threadIdx.y;
	int row = blockIdx.y * TILE_WIDTH + tIdY;
	int column = blockIdx.x * TILE_WIDTH + tIdX;
	int i, j, sum = 0;
	
	for (i = 0; i < size / TILE_WIDTH; i++)
	{
		Mds[tIdY][tIdX] = inputA[row * size + (i * TILE_WIDTH + tIdX)];
		Nds[tIdY][tIdX] = inputB[(i * TILE_WIDTH + tIdY) * size + column];
		__syncthreads();
		for (j = 0; j < TILE_WIDTH; j++)
			sum += Mds[tIdY][j] * Nds[j][tIdX];
		__syncthreads();
	}
	output[row * size + column] = sum;
}

float* generateArray(int count)
{
	float *array;
	array = (float*)malloc(count * sizeof(int));
	srand(time(NULL));
	for (int i = 0; i < count; i++)
	{
		(array)[i] = rand() % RAND_MAX;
	}
	return array;
}

void saveToFile(float* array, char* name, int size)
{
	int i, j;
	FILE *file = fopen(name, "a");
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
			fprintf(file, "%d\t", array[i * size + j]);
		fprintf(file, "\n");
	}
}


int main() {
	float *inputA, *inputB, *output, *dev_inputA, *dev_inputB, *dev_output, i, j, size = N * N * sizeof(int);
	float time;
	cudaEvent_t start, stop;
	LARGE_INTEGER frequency, startCPU, endCPU;
	FILE *fileTime = fopen("outTime.txt", "a");
	FILE *fileMatrixCPU = fopen("outMatrixCPU.txt", "a");
	FILE *fileMatrixGPU = fopen("outMatrixGPU.txt", "a");
	FILE *fileMatrixGPUSM = fopen("outMatrix.GPUSM.txt", "a");
	
	//prepare array
	inputA = generateArray(N * N);
	inputB = generateArray(N * N);
	output = (float*)malloc(size);
	for (int i = 0; i < N * N; i++)
		output[i] = 0;
	
	//CPU
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&startCPU);
	//CPU calculations
	matrixMultiplicationCPU(inputA, inputB, output);
	QueryPerformanceCounter(&endCPU);
	
	//save to file
	saveToFile(output, "outMatrixCPU.txt", N);
	
	fprintf(fileTime, "matrixMultiplicationCPU time %f ms\n", ((double)(endCPU.QuadPart - startCPU.QuadPart) / frequency.QuadPart) * 1000);

	//GPU
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	cudaMalloc((void **)&dev_inputA, size);
	cudaMalloc((void **)&dev_inputB, size);
	cudaMalloc((void **)&dev_output, size);

	cudaMemcpy(dev_inputA, inputA, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_inputB, inputB, size, cudaMemcpyHostToDevice);

	dim3 dimBlock(N, N);
	dim3 dimGrid(1, 1);
	//dim3 dimGrid((int)ceil(N/dimBlock.x),(int)ceil(N/dimBlock.y));

	int block = (int)ceil(N / dimBlock.x);
	int thread = (int)ceil(N / dimBlock.y);

	printf("Blok\n%d X\t%d Y\n\n", dimBlock.x, dimBlock.y);
	printf("Grid\n%d Blok\t%d Watki\n", block, thread);

	cudaEventRecord(start, 0);

	//GPU calculations
	matrixMultiplicationGPU << <dimGrid, dimBlock >> >(dev_inputA, dev_inputB, output, N);
	
	cudaMemcpy(output, dev_output, size, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	saveToFile(output, "outMatrixGPU.txt", N);
	fprintf(fileTime, "matrixMultiplicationGPU time %g ms\n", time);

	cudaFree(dev_inputA);
	cudaFree(dev_inputB);
	cudaFree(dev_output);

	//GPU + SM
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	cudaMalloc((void **)&dev_inputA, size);
	cudaMalloc((void **)&dev_inputB, size);
	cudaMalloc((void **)&dev_output, size);

	cudaMemcpy(dev_inputA, inputA, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_inputB, inputB, size, cudaMemcpyHostToDevice);

	dim3 dimGridSM(N / TILE_WIDTH, N / TILE_WIDTH);
	dim3 dimBlockSM(TILE_WIDTH, TILE_WIDTH);

	int blockSM = (int)ceil(N / dimBlockSM.x);
	int threadSM = (int)ceil(N / dimBlockSM.y);

	printf("Blok\n%d X\t%d Y\n\n", dimBlockSM.x, dimBlockSM.y);
	printf("Grid\n%d Blok\t%d Watki\n", blockSM, threadSM);

	cudaEventRecord(start, 0);
	
	//GPU calculations
	matrixMultiplicationGPUSharedMemeory <<<dimGridSM, dimBlockSM >>>(dev_inputA, dev_inputB, output, N);
	
	cudaMemcpy(output, dev_output, size, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	saveToFile(output, "outMatrixGPUSM.txt", N);
	fprintf(fileTime, "matrixMultiplication GPU SM time %g ms\n", time);

	cudaFree(dev_inputA);
	cudaFree(dev_inputB);
	cudaFree(dev_output);

	system("PAUSE");
	return 0;
}