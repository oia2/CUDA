#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 32 

__global__ void matrixMult(const __int64* Am, const __int64* Bm, __int64* result, int size) {
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int ia = size * (gridDim.y * by + ty);
	int ib = gridDim.x * bx + tx;
	int ic = ia + ib;

	__int64 sum = 0;

	for (int k = 0; k < size; k++) {
		sum += Am[ia + k] * Bm[k * size + ib];
	}
	result[ic] = sum;
}


void compareMatrix(const __int64* f, const __int64* s, int size) {
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			if (f[i * size + j] != s[i * size + j]) {
				printf("Matrixes not equal!\n");
				return;
			}
		}
	}
	printf("Matrices are equal!\n");
}

int main()
{
	int size = 1024;

	size_t byte_size = size * size * sizeof(__int64);
	__int64* Am = (__int64*)malloc(byte_size);
	__int64* Bm = (__int64*)malloc(byte_size);
	__int64* GPU_C = (__int64*)malloc(byte_size);
	__int64* CPU_C = (__int64*)malloc(byte_size);

	for (int i = 0; i < size * size; ++i) {
		Am[i] = rand() % 10;
		Bm[i] = rand() % 10;
		CPU_C[i] = 0;
	}

	printf("Scalar: \n");
	auto start = std::chrono::system_clock::now();
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			for (int k = 0; k < size; ++k) {
				CPU_C[i * size + j] += Am[i * size + k] * Bm[k * size + j];
			}
		}
	}
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	printf("Time: %f seconds\n", diff);

	printf("GPU: \n");

	__int64* d_A = NULL;
	cudaMalloc((void**)&d_A, byte_size);
	cudaMemcpy(d_A, Am, byte_size, cudaMemcpyHostToDevice);

	__int64* d_B = NULL;
	cudaMalloc((void**)&d_B, byte_size);
	cudaMemcpy(d_B, Bm, byte_size, cudaMemcpyHostToDevice);

	__int64* d_C = NULL;
	cudaMalloc((void**)&d_C, byte_size);

	start = std::chrono::system_clock::now();

	const dim3 block(32, 32);
	const dim3 grid((size) / block.x, (size) / block.y);
	matrixMult << < grid, block >> > (d_A, d_B, d_C, size);

	cudaDeviceSynchronize();

	end = std::chrono::system_clock::now();
	diff = end - start;
	cudaMemcpy(GPU_C, d_C, byte_size, cudaMemcpyDeviceToHost);


	printf("Time: %f seconds\n", diff);
	compareMatrix(GPU_C, CPU_C, size);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(Am);
	free(Bm);
	free(GPU_C);
	free(CPU_C);
	return 0;
}
