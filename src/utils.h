#ifndef UTILS_H_
#define UTILS_H_
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include "helper_cuda.h"

//row major
void out_hmat_rmj(float *A, int n, int m)
{
	printf("\n---------------\n");
	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			printf("%f%c", A[i * m + j], j == m - 1 ? '\n' : ' ');
	printf("\n---------------\n");
}
//column major
void out_hmat_cmj(float *A, int n, int m)
{
	printf("\n---------------\n");
	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			printf("%f%c", A[j * n + i], j == m - 1 ? '\n' : ' ');
	printf("\n---------------\n");
}
//column major on device
void out_device_mat(float *d_A, int n, int m)
{
	float *A = new float[n * m];
	checkCudaErrors(
			cudaMemcpy(A, d_A, n * m * sizeof(float), cudaMemcpyDeviceToHost));
	out_hmat_cmj(A, n, m);
	delete[] A;
}




#endif /* UTILS_H_ */
