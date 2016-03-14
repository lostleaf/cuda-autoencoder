#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <random>
#include <vector>
#include <string>
#include <sys/time.h>
using namespace std;

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "helper_cuda.h"

#include "utils.h"
#include "nn_kernels.h"
#include "readmnist.h"

default_random_engine gen;

float *d_X;
const int N_TRAIN = 1000;
const int IMAGE_H = 28;
const int IMAGE_W = 28;

class SparseAutoEncoder
{
public:
	SparseAutoEncoder(int _visible_size, int _hidden_size)
	{
		checkCublasErrors(cublasCreate(&handle));

		visible_size = _visible_size;
		hidden_size = _hidden_size;
		n_batch = -1;

		//W1(h*v) W2(v*h) b1(h*1) b2(v*1)
		allocate(&d_W1, hidden_size, visible_size);
		allocate(&d_W2, visible_size, hidden_size);
		allocate(&d_b1, hidden_size, 1);
		allocate(&d_b2, visible_size, 1);

		//grad_W1(h*v) grad_b1(h*1) grad_W2(v*h) grad_b2(v*1)
		allocate(&d_grad_W1, hidden_size, visible_size);
		allocate(&d_grad_b1, hidden_size, 1);
		allocate(&d_grad_W2, visible_size, hidden_size);
		allocate(&d_grad_b2, visible_size, 1);

		//rho_hat(h*1) delta_sparse(h*1)
		allocate(&d_rho_hat, hidden_size, 1);
		allocate(&d_kl_val, hidden_size, 1);
		allocate(&d_delta_sparse, hidden_size, 1);

		//variables related to n_batch, set to NULL
		d_ones = 0;
//		d_X = 0;
		d_z2 = 0;
		d_z3 = 0;
		d_a2 = 0;
		d_a3 = 0;
		d_delta2 = 0;
		d_delta3 = 0;
		d_err = 0;
		d_ap_z2 = 0;
		d_ap_z3 = 0;
		d_prod_W2T_delta3 = 0;

		//Initialize parameters
		init_parameters(d_W1, hidden_size, visible_size);
		init_parameters(d_W2, visible_size, hidden_size);
//		init_from_file("W1.txt", d_W1, hidden_size, visible_size);
//		init_from_file("W2.txt", d_W2, visible_size, hidden_size);
		checkCudaErrors(cudaMemset(d_b1, 0, sizeof(float) * hidden_size * 1));
		checkCudaErrors(cudaMemset(d_b2, 0, sizeof(float) * visible_size * 1));
	}
	~SparseAutoEncoder()
	{
		checkCudaErrors(cudaFree(d_W1));
		checkCudaErrors(cudaFree(d_W2));
		checkCudaErrors(cudaFree(d_b1));
		checkCudaErrors(cudaFree(d_b2));

		checkCudaErrors(cudaFree(d_grad_W1));
		checkCudaErrors(cudaFree(d_grad_W2));
		checkCudaErrors(cudaFree(d_grad_b1));
		checkCudaErrors(cudaFree(d_grad_b2));

		checkCudaErrors(cudaFree(d_rho_hat));
		checkCudaErrors(cudaFree(d_delta_sparse));

		free_others();
	}

	void set_n_batch(int n)
	{
		free_others();
		n_batch = n;
		//Helper: ones(1*n_batch)
		allocate(&d_ones, 1, n_batch);
		fill_ones(d_ones, n_batch);

//Forward: z2(h*n_batch) a2(h*n_batch) z3(v*n_batch) a3(v*n_batch)
		allocate(&d_z2, hidden_size, n_batch);
		allocate(&d_a2, hidden_size, n_batch);
		allocate(&d_z3, visible_size, n_batch);
		allocate(&d_a3, visible_size, n_batch);

		//Backward: delta2(h*n_batch) delta3(v*n_batch) err(v*n_batch)
		allocate(&d_delta2, hidden_size, n_batch);
		allocate(&d_delta3, visible_size, n_batch);
		allocate(&d_err, visible_size, n_batch);

		//Activation prime: of z2(h*n_batch) and z3(v*n_batch)
		allocate(&d_ap_z2, hidden_size, n_batch);
		allocate(&d_ap_z3, visible_size, n_batch);

		//Temporary variable: W2.T*delta3, size z2(h*n_batch)
		allocate(&d_prod_W2T_delta3, hidden_size, n_batch);
	}

	//Forward: z2(h*n_batch) a2(h*n_batch) z3(v*n_batch) a3(v*n_batch) ones(1*n_batch)
	void forward(float *d_Xtrain)
	{
		float f_one = 1, f_zero = 0;

		//z2 = W1*X
		checkCublasErrors(
				cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, hidden_size, n_batch, visible_size, &f_one, d_W1, hidden_size, d_Xtrain, visible_size, &f_zero, d_z2, hidden_size));

		//z2 += b1*ones
		checkCublasErrors(
				cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, hidden_size, n_batch, 1, &f_one, d_b1, hidden_size, d_ones, 1, &f_one, d_z2, hidden_size));
		//a2 = f(z2)
		active(d_z2, d_a2, hidden_size, n_batch);

		//z3 = W2*a2
		checkCublasErrors(
				cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, visible_size, n_batch, hidden_size, &f_one, d_W2, visible_size, d_a2, hidden_size, &f_zero, d_z3, visible_size));
		//z3 += b2*ones
		checkCublasErrors(
				cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, visible_size, n_batch, 1, &f_one, d_b2, visible_size, d_ones, 1, &f_one, d_z3, visible_size));

		//a3 = f(z3)
		active(d_z3, d_a3, visible_size, n_batch);
	}

	void backward(float sparse_param, float beta, float *d_Xtrain)
	{
		float f_one = 1, f_zero = 0, rho = sparse_param;

		//rho_hat = a2 * ones = sum(a2, 'row')
		float rho_hat_alpha = 1.0 / n_batch;
		checkCublasErrors(
				cublasSgemv(handle, CUBLAS_OP_N, hidden_size, n_batch, &f_one, d_a2, hidden_size, d_ones, 1, &f_zero, d_rho_hat, 1));
		checkCublasErrors(
				cublasSscal(handle, hidden_size * 1, &rho_hat_alpha, d_rho_hat, 1));

		//delta_sparse = kl'(rho, rho_hat) / m
		kl_prime(d_rho_hat, d_delta_sparse, rho, hidden_size * 1);

		// err = a3 - y_train
		square_loss_prime(d_a3, d_Xtrain, d_err, visible_size, n_batch);

		// ap_z3 = f'(z3)
		active_prime(d_z3, d_ap_z3, visible_size, n_batch);

		// delta3 = err .* ap_z3 = err .* f'(z3)
		element_mul(d_err, d_ap_z3, d_delta3, visible_size, n_batch);

		// ap_z2 = f'(z2)
		active_prime(d_z2, d_ap_z2, hidden_size, n_batch);
		// prod_W2T_delta3 = W2.T * delta3
		checkCublasErrors(
				cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, hidden_size, n_batch, visible_size, &f_one, d_W2, visible_size, d_delta3, visible_size, &f_zero, d_prod_W2T_delta3, hidden_size));

		//prod_W2T_delta3 += beta * broadcast(delta_sparse)
		checkCublasErrors(
				cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, hidden_size, n_batch, 1, &beta, d_delta_sparse, hidden_size, d_ones, 1, &f_one, d_prod_W2T_delta3, hidden_size));

		// delta2 = prod_W2T_delta3 .* ap_z2 = (W2.T * delta3) .* f'(z2)
		element_mul(d_prod_W2T_delta3, d_ap_z2, d_delta2, hidden_size, n_batch);
//		out_device_mat(d_delta2, hidden_size, n_batch);
	}

	void compute_grad(float *d_Xtrain)
	{
		float f_one = 1, f_zero = 0;

		// grad_W2 = delta3 * a2.T
		checkCublasErrors(
				cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, visible_size, hidden_size, n_batch, &f_one, d_delta3, visible_size, d_a2, hidden_size, &f_zero, d_grad_W2, visible_size));

		// grad_b2 = sum(delta3, 'row') = delta3 * ones
		checkCublasErrors(
				cublasSgemv(handle, CUBLAS_OP_N, visible_size, n_batch, &f_one, d_delta3, visible_size, d_ones, 1, &f_zero, d_grad_b2, 1));

		// grad_W1 = delta2 * X.T
		checkCublasErrors(
				cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, hidden_size, visible_size, n_batch, &f_one, d_delta2, hidden_size, d_Xtrain, visible_size, &f_zero, d_grad_W1, hidden_size));
		// grad_b1 = sum(delta2, 'row') = delta2 * ones
		checkCublasErrors(
				cublasSgemv(handle, CUBLAS_OP_N, hidden_size, n_batch, &f_one, d_delta2, hidden_size, d_ones, 1, &f_zero, d_grad_b1, 1));

	}

	void gradient_descent(float rate, float lambda, float sparse_param,
			float beta, float *d_Xtrain, int n_batch, float *cost_ = 0)
	{
		if (n_batch != this->n_batch)
		{
			fprintf(stderr,
					"n_batch of input in gradient descent does not match pre-set n_batch\n");
			exit(-1);
		}

		forward(d_Xtrain);

		backward(sparse_param, beta, d_Xtrain);

		compute_grad(d_Xtrain);

		float weight_decay_rate = 1 - lambda, alpha_rate = -rate / n_batch;

		//weight decay W1
		checkCudaErrors(
				cublasSscal(handle, hidden_size * visible_size, &weight_decay_rate, d_W1, 1));

		//weight decay W2
		checkCudaErrors(
				cublasSscal(handle, visible_size * hidden_size, &weight_decay_rate, d_W2, 1));

		// W1 = W1 - rate / m * grad_W1
		checkCublasErrors(
				cublasSaxpy(handle, hidden_size * visible_size, &alpha_rate, d_grad_W1, 1, d_W1, 1));

		// b1 = b1 - rate / m * grad_b1
		checkCublasErrors(
				cublasSaxpy(handle, hidden_size * 1, &alpha_rate, d_grad_b1, 1, d_b1, 1));

		//W2 = W2 - rate / m * grad_W2
		checkCublasErrors(
				cublasSaxpy(handle, visible_size * hidden_size, &alpha_rate, d_grad_W2, 1, d_W2, 1));

		//b2 = b2 - rate / m * grad_b2
		checkCublasErrors(
				cublasSaxpy(handle, visible_size * 1, &alpha_rate, d_grad_b2, 1, d_b2, 1));
		if (cost_)
		{

			float tmp;
			double cost;
			checkCublasErrors(
					cublasSnrm2(handle, visible_size * n_batch, d_err, 1, &tmp));
			cost = tmp * (double) tmp / (2 * n_batch);

			checkCublasErrors(
					cublasSnrm2(handle, hidden_size * visible_size, d_W1, 1, &tmp));
			cost += tmp * (double) tmp * lambda / 2;

			checkCublasErrors(
					cublasSnrm2(handle, visible_size * hidden_size, d_W2, 1, &tmp));
			cost += tmp * (double) tmp * lambda / 2;
//			*cost_ = cost;

			kl(d_rho_hat, d_kl_val, sparse_param, hidden_size * 1);

//			out_device_mat(d_rho_hat, hidden_size, 1);
//			out_device_mat(d_kl_val, hidden_size, 1);

			checkCublasErrors(
					cublasSasum(handle, hidden_size * 1, d_kl_val, 1, &tmp));
			cost += (double) tmp * beta;

			*cost_ = (float) cost;
		}
	}

	void reconstruct(float *d_X, float *h_result, int _n_batch)
	{
		set_n_batch(_n_batch);
		forward(d_X);
		checkCudaErrors(
				cudaMemcpy(h_result, d_a3, _n_batch * IMAGE_W * IMAGE_H * sizeof(float),
						cudaMemcpyDeviceToHost));
	}
private:
	void read_mat_trans(const char *fname, float *A, int n_row, int n_col)
	{
		FILE *fin = fopen(fname, "r");
		for (int i = 0; i < n_row; i++)
			for (int j = 0; j < n_col; j++)
				fscanf(fin, "%f", &A[j * n_row + i]);
		fclose(fin);
	}

	void init_from_file(const char *fname, float *d_A, int n_row, int n_col)
	{
		float *h_A = new float[n_row * n_col];
		read_mat_trans(fname, h_A, n_row, n_col);
		checkCudaErrors(
				cudaMemcpy(d_A, h_A, n_row * n_col * sizeof(float),
						cudaMemcpyHostToDevice));

		delete[] h_A;
	}

	void free_others()
	{
		if (n_batch > -1)
		{
			checkCudaErrors(cudaFree(d_ones));
//			checkCudaErrors(cudaFree(d_X));
			checkCudaErrors(cudaFree(d_z2));
			checkCudaErrors(cudaFree(d_z3));
			checkCudaErrors(cudaFree(d_a2));
			checkCudaErrors(cudaFree(d_a3));
			checkCudaErrors(cudaFree(d_delta2));
			checkCudaErrors(cudaFree(d_delta3));
			checkCudaErrors(cudaFree(d_err));
			checkCudaErrors(cudaFree(d_ap_z2));
			checkCudaErrors(cudaFree(d_ap_z3));
			checkCudaErrors(cudaFree(d_prod_W2T_delta3));
			n_batch = -1;
		}
	}

	void allocate(float **p, int n_rows, int n_cols)
	{
		checkCudaErrors(cudaMalloc(p, n_rows * n_cols * sizeof(float)));
	}

	void init_parameters(float *d_M, int n_rows, int n_cols)
	{
		int n = n_rows * n_cols;
		float r = sqrt(6.0 / (hidden_size + visible_size + 1));
		float *M = new float[n_rows * n_cols];
		uniform_real_distribution<float> distribution(-r, r);
		for (int i = 0; i < n; i++)
			M[i] = distribution(gen);

		checkCudaErrors(
				cudaMemcpy(d_M, M, n_rows * n_cols * sizeof(float),
						cudaMemcpyHostToDevice));
		delete[] M;
	}
	int visible_size, hidden_size, n_batch;

	//Parameters: W1(h*v) W2(v*h) b1(h*1) b2(v*1)
	float *d_W1, *d_b1, *d_W2, *d_b2;

	//Gradients: grad_W1(h*v) grad_b1(h*1) grad_W2(v*h) grad_b2(v*1)
	float *d_grad_W1, *d_grad_b1, *d_grad_W2, *d_grad_b2;

	//Sparse parameters: rho_hat(h*1) delta_sparse(h*1)
	float *d_rho_hat, *d_delta_sparse, *d_kl_val;

	//Helper: ones(1*n_batch)
	float *d_ones;

	//Forward: z2(h*n_batch) a2(h*n_batch) z3(v*n_batch) a3(v*n_batch)
	float *d_z2, *d_z3, *d_a2, *d_a3;

	//Backward: delta2(h*n_batch) delta3(v*n_batch) err(v*n_batch)
	float *d_delta2, *d_delta3, *d_err;

	//Activation prime: of z2(h*n_batch) and z3(v*n_batch)
	float *d_ap_z2, *d_ap_z3;

	//Temporary variable: W2.T*delta3, size same as z2(h*n_batch)
	float *d_prod_W2T_delta3;

	cublasHandle_t handle;
};

void read_data()
{
	vector<vector<float> > vec;
	string file_name = "train-images-idx3-ubyte";
	read_Mnist(file_name, vec);
	float *h_X = new float[IMAGE_W * IMAGE_H * N_TRAIN];
	int idx = 0;
	for (int i = 0; i < N_TRAIN; i++)
		for (int j = 0; j < IMAGE_H * IMAGE_W; j++)
			h_X[idx++] = vec[i][j] / 255.0;
	checkCudaErrors(
			cudaMemcpy(d_X, h_X, IMAGE_W * IMAGE_H * N_TRAIN * sizeof(float),
					cudaMemcpyHostToDevice));
	delete[] h_X;
}

void train()
{
	const int n_hidden = 196;
	const int max_it = 10000;
	float rate = 0.5, lambda = 3e-3, rho = 0.1, beta = 3;
	SparseAutoEncoder ae(IMAGE_H * IMAGE_H, n_hidden);
	ae.set_n_batch(N_TRAIN);
	timeval t1, t2;
	gettimeofday(&t1, 0);
	for (int i = 1; i <= max_it; i++)
		if (i % 100 == 0)
		{
			gettimeofday(&t2, 0);
			float cost;
			ae.gradient_descent(rate, lambda, rho, beta, d_X, N_TRAIN, &cost);
			//rate *= 0.95;
			printf(
					"Epoch %d, cost %f, time elapsed in ms %ld, learning rate %f\n",
					i, cost,
					(t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec,
					rate);
			gettimeofday(&t1, 0);
		}
		else
			ae.gradient_descent(rate, lambda, rho, beta, d_X, N_TRAIN, NULL);
	float *result = new float[IMAGE_W * IMAGE_H];
	//reconstruct image[0]
	ae.reconstruct(d_X + IMAGE_W * IMAGE_H, result, 1);
	FILE *fout = fopen("result.txt", "w");
	for (int i = 0; i < IMAGE_H; i++)
		for (int j = 0; j < IMAGE_W; j++)
			fprintf(fout, "%f%c", result[i * IMAGE_W + j],
					j == IMAGE_W - 1 ? '\n' : ' ');
	fclose(fout);
	delete[] result;
}

int main()
{
	checkCudaErrors(
			cudaMalloc(&d_X, IMAGE_W * IMAGE_H * N_TRAIN * sizeof(float)));
	read_data();
	train();
	checkCudaErrors(cudaFree(d_X));
	return 0;
}
