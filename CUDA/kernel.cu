#include "device_launch_parameters.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include<Windows.h>
#include "cuda_runtime.h"


#define RAND_MAX 100
#define TILE_WIDTH 8
int size;

void matrixMultiplicationCPU(int* inputA, int* inputB, int* output)
{
	int i, j, k, sum;
	for (i = 0; i < size; i++)
		for (j = 0; j < size; j++)
		{
			sum = 0;
			for (k = 0; k < size; k++)
				sum += inputA[i * size + k] * inputB[k * size + j];
			output[i * size + j] = sum;
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
	int i, j, sum = 0;;
	
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
	LARGE_INTEGER frequency, start, end;

	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&start);
	
	matrixMultiplicationCPU(inputA, inputB, output);
	
	QueryPerformanceCounter(&end);

	saveToFile(output, "outMatrixCPU.txt", size);
	fprintf(fileTime, "%f\t", ((double)(end.QuadPart - start.QuadPart) / frequency.QuadPart) * 1000);
}

void GPU(int* inputA, int* inputB, int* output, FILE *fileTime)
{
	int *dev_inputA, *dev_inputB, *dev_output;
	float time;
	LARGE_INTEGER frequency, start, end;

	cudaMalloc((void **)&dev_inputA, size * size * sizeof(int));
	cudaMalloc((void **)&dev_inputB, size * size * sizeof(int));
	cudaMalloc((void **)&dev_output, size * size * sizeof(int));

	cudaMemcpy(dev_inputA, inputA, size * size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_inputB, inputB, size * size * sizeof(int), cudaMemcpyHostToDevice);

	dim3 dimGrid(size / TILE_WIDTH, size / TILE_WIDTH);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&start);
	matrixMultiplicationGPU << <dimGrid, dimBlock >> >(dev_inputA, dev_inputB, dev_output, size);
	cudaMemcpy(output, dev_output, size * size * sizeof(int), cudaMemcpyDeviceToHost);
	QueryPerformanceCounter(&end);

	saveToFile(output, "outMatrixGPU.txt", size);
	fprintf(fileTime, "%f\t", ((double)(end.QuadPart - start.QuadPart) / frequency.QuadPart) * 1000);

	cudaFree(dev_inputA);
	cudaFree(dev_inputB);
	cudaFree(dev_output);
}

void GPUSM(int* inputA, int* inputB, int* output, FILE *fileTime)
{
	int *dev_inputA, *dev_inputB, *dev_output;
	float time;
	LARGE_INTEGER frequency, start, end;

	cudaMalloc((void **)&dev_inputA, size * size * sizeof(int));
	cudaMalloc((void **)&dev_inputB, size * size * sizeof(int));
	cudaMalloc((void **)&dev_output, size * size * sizeof(int));

	cudaMemcpy(dev_inputA, inputA, size * size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_inputB, inputB, size * size * sizeof(int), cudaMemcpyHostToDevice);

	dim3 dimGridSM(size / TILE_WIDTH, size / TILE_WIDTH);
	dim3 dimBlockSM(TILE_WIDTH, TILE_WIDTH);
	
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&start);
	
	matrixMultiplicationGPUSharedMemeory << <dimGridSM, dimBlockSM >> >(dev_inputA, dev_inputB, dev_output, size);
	cudaMemcpy(output, dev_output, size * size * sizeof(int), cudaMemcpyDeviceToHost);
	
	QueryPerformanceCounter(&end);

	saveToFile(output, "outMatrixGPUSH.txt", size);
	fprintf(fileTime, "%f\t", ((double)(end.QuadPart - start.QuadPart) / frequency.QuadPart) * 1000);

	cudaFree(dev_inputA);
	cudaFree(dev_inputB);
	cudaFree(dev_output);
}

void init(int** inputA, int** inputB, int** output)
{
	*inputA = generateArray(size * size);
	*inputB = generateArray(size * size);
	*output = (int*)malloc(size * size * sizeof(int));
	for (int i = 0; i < size * size; i++)
		(*output)[i] = 0;
}

int main(int argc, char *argv[]) {

	size  = (int)atoi(argv[1]);
	int *inputA, *inputB, *output;
	FILE *fileTime = fopen("outTime.txt", "a");
	cudaError_t cudaStatus;
	
	fprintf(fileTime, "\n%d\t", size);
	init(&inputA, &inputB, &output);
	CPU(inputA, inputB, output, fileTime);
	
	for (int k = 0; k < size * size; k++)
		output[k] = 0;
	GPU(inputA, inputB, output, fileTime);

	for (int k = 0; k < size * size; k++)
		output[k] = 0;
	GPUSM(inputA, inputB, output, fileTime);
	
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "error: %s\n", cudaGetErrorString(cudaStatus));
	}

	free(inputA);
	free(inputB);
	free(output);	
	return 0;
}