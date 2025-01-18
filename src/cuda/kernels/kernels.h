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
 * Update positions and velocities according to the force applied to each particle.
 * All three matrices must be DIM * n_particles, and contain on each row the components parallel to one axis.
 * Execute with a 1D grid of 1D blocks. (Might be faster serially)
 */
__global__ void apply_motion(   double* pos,
                                double* vel,
                                const double* force,
                                unsigned int n_particles,
                                double dt   );

#endif //KERNELS_H
