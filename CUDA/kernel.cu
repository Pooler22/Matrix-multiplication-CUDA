#include "device_launch_parameters.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include <math.h>
#include "cuda_runtime.h"

#define M 7
#define K 7
#define N 7

#define ROW_1 100
#define COL_1 100
#define ROW_2 100
#define ROW_1 100

#define RAND_MAX 100
#define TILE_WIDTH 2

void matrixMultiplicationCPU(int* inputA, int* inputB, int* output)
{
	int i, j, k, sum;

	for (i = 0; i < M; i++)
		for (j = 0; j < N; j++){
			sum = 0;
			for (k = 0; k < K; k++)
				sum += inputA[i * K + k] * inputB[k * N + j];
			output[i * N + j] = sum;
		}
}

__global__ void matrixMultiplicationGPU(int *inputA, int *inputB, int *output, int size)
{
	int i, sum ;
	int columns = threadIdx.x + blockDim.x * blockIdx.x;
	int rows = threadIdx.y + blockDim.y * blockIdx.y;

	if (columns < size && rows < size)
	{
		sum = 0;
		for (i = 0; i < size; i++)
			sum += inputA[rows * size + i] * inputB[i * size + columns];
		output[rows * size + columns] = sum;
	}
}

__global__ void matrixMultiplicationGPUSharedMemeory(int *inputA, int *inputB, int *output, int size)
{
	__shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ int Nds[TILE_WIDTH][TILE_WIDTH];
	int tIdX = threadIdx.x;
	int tIdY = threadIdx.y;
	int row = blockIdx.y * TILE_WIDTH + tIdY;
	int column = blockIdx.x * TILE_WIDTH + tIdX;
	int i, j, sum;
	
	for (i = 0; i < size / TILE_WIDTH; i++)
	{
		sum = 0;
		Mds[tIdY][tIdX] = inputA[row * size + (i * TILE_WIDTH + tIdX)];
		Nds[tIdY][tIdX] = inputB[(i * TILE_WIDTH + tIdY) * size + column];
		__syncthreads();
		for (j = 0; j < TILE_WIDTH; j++)
			sum += Mds[tIdY][j] * Nds[j][tIdX];
		__syncthreads();
	}
	output[row * size + column] = sum;
}

int* generateArray(int count)
{
	int *array;
	srand(time(NULL));
	array = (int*)malloc(count * sizeof(int));
	for (int i = 0; i < count; i++)
		(array)[i] = rand() % RAND_MAX;
	return array;
}

void saveToFile(int* array, char* name, int size)
{
	int i, j;
	FILE *file = fopen(name, "a");
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
			fprintf(file, "%d\t", array[i * size + j]);
		fprintf(file, "\n");
	}
	fclose(file);
}

void CPU(int* inputA, int* inputB, int* output, FILE *fileTime)
{
	LARGE_INTEGER frequency, startCPU, endCPU;

	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&startCPU);
	
	matrixMultiplicationCPU(inputA, inputB, output);
	
	QueryPerformanceCounter(&endCPU);

	saveToFile(output, "outMatrixCPU.txt", N);
	fprintf(fileTime, "CPU time %f ms\n", ((double)(endCPU.QuadPart - startCPU.QuadPart) / frequency.QuadPart) * 1000);
}

void GPU(int* inputA, int* inputB, int* output, FILE *fileTime)
{
	cudaEvent_t start, stop;
	int *dev_inputA, *dev_inputB, *dev_output;
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMalloc((void **)&dev_inputA, N * N * sizeof(int));
	cudaMalloc((void **)&dev_inputB, N * N * sizeof(int));
	cudaMalloc((void **)&dev_output, N * N * sizeof(int));

	cudaMemcpy(dev_inputA, inputA, N * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_inputB, inputB, N * N * sizeof(int), cudaMemcpyHostToDevice);

	dim3 dimBlock(N, N);
	dim3 dimGrid(1, 1);
	//dim3 dimGrid((int)ceil(N/dimBlock.x),(int)ceil(N/dimBlock.y));

	cudaEventRecord(start, 0);

	matrixMultiplicationGPU << <dimGrid, dimBlock >> >(dev_inputA, dev_inputB, dev_output, N);
	cudaMemcpy(output, dev_output, N * N * sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	saveToFile(output, "outMatrixGPU.txt", N);
	fprintf(fileTime, "GPU time %f ms\n", time);

	cudaFree(dev_inputA);
	cudaFree(dev_inputB);
	cudaFree(dev_output);
}

void GPUSM(int* inputA, int* inputB, int* output, FILE *fileTime)
{
	cudaEvent_t start, stop;
	int *dev_inputA, *dev_inputB, *dev_output;
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaMalloc((void **)&dev_inputA, N * N * sizeof(int));
	cudaMalloc((void **)&dev_inputB, N * N * sizeof(int));
	cudaMalloc((void **)&dev_output, N * N * sizeof(int));

	cudaMemcpy(dev_inputA, inputA, N * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_inputB, inputB, N * N * sizeof(int), cudaMemcpyHostToDevice);

	dim3 dimGridSM(N / TILE_WIDTH, N / TILE_WIDTH);
	dim3 dimBlockSM(TILE_WIDTH, TILE_WIDTH);

	cudaEventRecord(start, 0);

	matrixMultiplicationGPUSharedMemeory << <dimGridSM, dimBlockSM >> >(dev_inputA, dev_inputB, output, N);
	cudaMemcpy(output, dev_output, N * N * sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	saveToFile(output, "outMatrixGPUSM.txt", N);
	fprintf(fileTime, "GPU SM time %g ms\n", time);

	cudaFree(dev_inputA);
	cudaFree(dev_inputB);
	cudaFree(dev_output);
}

int main() {
	int *inputA, *inputB, *output;
	float time;
	cudaEvent_t startSM, stopSM;
	FILE *fileTime = fopen("outTime.txt", "a");
	
	//init
	inputA = generateArray(N * N);
	inputB = generateArray(N * N);
	output = (int*)malloc(N * N * sizeof(int));
	for (int i = 0; i < N * N; i++)
		output[i] = 0;

	//CPU
	CPU(inputA, inputB, output, fileTime);

	//GPU
	GPU(inputA, inputB, output, fileTime);

	//GPU + SM
	GPUSM(inputA, inputB, output, fileTime);

	system("PAUSE");
	return 0;
}