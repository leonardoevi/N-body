#include "kernels.h"

__global__ void calculate_pairwise_force_component(const double* pos, const double* mass, const unsigned int component, double* matrix, const unsigned int n_particles, const unsigned int blocks_per_row) {
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
                matrix[i * n_particles + j] = G * di[component] / (d > D_MIN ? d : D_MIN) * mass[j] * mass[i];
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
    // Shared memory for block-level reduction
    extern __shared__ double shared_data[];

    unsigned int row = blockIdx.x;  // Each block processes one row
    unsigned int col = threadIdx.x; // Thread within the row

    unsigned int index = row * matrix_edge_size + col; // Global memory index

    // Load matrix data into shared memory
    shared_data[col] = (col < matrix_edge_size) ? mat[index] : 0.0;

    __syncthreads(); // Ensure all threads finish loading data

    // Perform parallel reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (col < stride && (col + stride) < matrix_edge_size) {
            shared_data[col] += shared_data[col + stride];
        }
        __syncthreads(); // Ensure all threads complete their additions
    }

    // Write the result to the output array
    if (col == 0) {
        arr[row] = shared_data[0];
    }
}

__global__ void apply_motion(double* pos, double* vel, const double* mass, const double* force, const unsigned int n_particles, const integration_type integration, const double dt) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_particles && mass[i] > 0.0) {
        if (integration == forwardEuler)
            for (int k = 0; k < DIM; k++) {
                vel[n_particles * k + i] += force[n_particles * k + i] / mass[i] * dt;
                pos[n_particles * k + i] += vel[n_particles * k + i] * dt;
            }
    }
}