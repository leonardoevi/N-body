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

/**
 * Perform the sum of all the elements in each row of the matrix, stores the result in
 * the corresponding element of the array.
 * 1 thread per row, 1D grid, 1D blocks.
 * @param mat pointer to the square matrix of size matrix_edge_size x matrix_edge_size
 * @param arr pointer to the array of length matrix_edge_size
 * @param matrix_edge_size size of the matrix
*/
__global__ void sum_over_rows(const double* mat, double* arr, unsigned int matrix_edge_size);

/**
 * Calculates res = x + y * b, element wise
 * 1 thread per column, 1D grid, 1D blocks.
 * @param x DIM x size matrix
 * @param y DIM x size matrix
 * @param b multiply factor
 * @param res DIM x size matrix
 * @param size n of columns of the matrices.
 */
__global__ void x_plus_by(const double* x, double* y, double b, double* res, unsigned int size);

/**
 * Perform the sum of all the elements in each row of the matrix, with a parallel implementation
 * on each row.
 * !WARNING! modifies the matrix to store partial results, and requires additional computation to
 * accumulate the partial results.
 */
__global__ void reduceSum_rows_parallel(double *input, int size, int A);

/**
 * Performs the accumulation of the partial results left by the previous kernel and
 * stores them in the array.
 * @param input pointer to the matrix : size x size
 * @param out pointer to the array
 * @param size size of the array
 * @param step distance between elements left by the previous kernel
 */
__global__ void sumRowsInterleaved(double *input, double *out, int size, int step);

__global__ void setZeroDiag(double *input, int size);

#endif //KERNELS_H
