#ifndef KERNELS_H
#define KERNELS_H

#include "../include/defines.h"
#include "../include/integration_type.h"

/**
 * @param index
 * @param n size of the 2D matrix
 * @return the pair of indexes (i, j) that correspond to the index-th element
 * in a 2D square matrix of size n, numbered from 0, row major, skipping elements
 * under the main diagonal. In a 3x3 the numbering would be:
 * [0 1 2]
 * [- 3 4]
 * [- - 5]
 */
__host__ __device__ void
mapIndexTo2D(unsigned int index, unsigned int n, unsigned int &i, unsigned int &j);

/**
 * This kernel computes one component of all the forces between the particles and writes it in the matrix.
 * @param pos a dim x n_particles matrix, each column contains the coordinates of a particle: (x, y, z) in the case dim=3.
 * @param mass an n_particles array containing the mass of each particle.
 * @param component value between 0 and dim-1 specifies the component of the force needed (x, y, or z).
 * @param matrix [i][j] will contain the value of the force along the specified axis on particle i caused by particle j.
 * It has to be an n_particles x n_particles matrix.
 */
__global__ void calculate_pairwise_acceleration_component( const double* pos,
                                                    const double* mass,
                                                    unsigned int component,
                                                    double* matrix,
                                                    unsigned int n_particles,
                                                    unsigned int blocks_per_row );

__global__ void sum_over_rows(const double* mat, double* arr, unsigned int matrix_edge_size);

/**
 * Calculates res = x + y * b, element wise
 * @param x DIM x size matrix
 * @param y DIM x size matrix
 * @param b multiply factor
 * @param res DIM x size matrix
 */
__global__ void x_plus_by(const double* x, double* y, double b, double* res, unsigned int size);

__global__ void reduceSum_rows_parallel(double *input, int size, int A);

__global__ void sumRowsInterleaved(double *input, double *out, int size, int step);

__global__ void setZeroDiag(double *input, int size);

#endif //KERNELS_H
