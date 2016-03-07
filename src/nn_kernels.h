/*
 * nn_kernels.h
 *
 *  Created on: Mar 4, 2016
 *      Author: lostleaf
 */

#ifndef NN_KERNELS_H_
#define NN_KERNELS_H_
#include <cuda.h>
#include "helper_cuda.h"

void fill_ones(float *d_arr, int n);
void active(float *d_in, float *d_out, int n_rows, int n_cols);
void active_prime(float *d_in, float *d_out, int n_rows, int n_cols);
//Y_pred - Y_true
void square_loss_prime(float *d_Ypred, float *d_Ytrue, float *d_err, int n_rows, int n_cols);
void element_mul(float *d_a, float *d_b, float *d_c, int n_rows, int n_cols);

void kl_prime(float *d_in, float *d_out, float rho, int n);
void kl(float *d_in, float *d_out, float rho, int n);
#endif /* NN_KERNELS_H_ */
