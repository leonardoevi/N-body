#include "kernels.h"

__global__ void calculate_pairwise_acceleration_component(const double* pos, const double* mass, const unsigned int component, double* matrix, const unsigned int n_particles, const unsigned int blocks_per_row) {
    unsigned int i, j; mapIndexTo2D(blockIdx.x, blocks_per_row, i, j);
    i = i * blockDim.x + threadIdx.x;
    j = j * blockDim.y + threadIdx.y;

    if (i < n_particles && j < n_particles && i < j)
        if (j > i) {
            double di[DIM];
            for (unsigned int k = 0; k < DIM; ++k)
                di[k] = pos[k * n_particles + j] - pos[k * n_particles + i];

            double d = 0;
            for (unsigned int k = 0; k < DIM; ++k)
                d += di[k] * di[k];
            d = std::sqrt(d);

            d = d * d * d;
            if (d > 0) {
                matrix[i * n_particles + j] = G * di[component] / (d > D_MIN ? d : D_MIN) * mass[j];
                matrix[j * n_particles + i] = -matrix[i * n_particles + j];
            }
            free(di);
        }
}

__host__ __device__ void
mapIndexTo2D(unsigned int index, const unsigned int n, unsigned int &i, unsigned int &j) {
    // Find row and column
    int row = 0;
    while (index >= n - row) {
        index -= (n - row);
        row++;
    }
    unsigned int col = row + index;

    i = row;
    j = col;
}

__global__ void sum_over_rows(const double* mat, double* arr, const unsigned int matrix_edge_size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < matrix_edge_size) {
        arr[i] = 0;
        for (unsigned int j = 0; j < matrix_edge_size; j++)
            arr[i] += mat[i * matrix_edge_size + j];
    }
}

__global__ void x_plus_by(const double* x, double* y, double b, double* res, unsigned int size) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
        for (int k = 0; k < DIM; k++) {
            res[i + size * k] = x[i + size * k] + b * y[i + size * k];
        }
}

__global__ void reduceSum_rows_parallel(double *input, int size, int A) {
    extern __shared__ double sharedData[];

    const unsigned int t_idx = threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.x * 2 + threadIdx.x;

    for (unsigned int k = 0; k < A; k++) {
        const unsigned int i = blockIdx.x * A + k;

        if (i >= size)
            return;

        sharedData[t_idx] = 0.0;
        sharedData[t_idx + blockDim.x] = 0.0;
        if (j < size)
            sharedData[t_idx] = input[i * size + j];
        if (j + blockDim.x < size)
            sharedData[t_idx + blockDim.x] = input[i * size + j + blockDim.x];

        __syncthreads();

        for (unsigned int stride = blockDim.x; stride > 0; stride >>= 1) {
            if (t_idx < stride) {
                sharedData[t_idx] += sharedData[t_idx + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0)
            input[i * size + j] = sharedData[0];

    }
}

__global__ void sumRowsInterleaved(double *input, double *out, int size, int step) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size) {
        double sum = 0.0;
        for (unsigned int j = 0; j < size; j += step)
            sum += input[row * size + j];

        out[row] = sum;
    }
}

__global__ void setZeroDiag(double *input, const int size) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
        input[i * size + i] = 0.0;
}
