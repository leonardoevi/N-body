#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>

#include "include/defines.h"
#include "kernels/kernels.h"
#include "futils/futils.h"

int main() {
    std::cout << N_PARTICLES << " particles." << std::endl;

    // calculate the number of blocks needed
    constexpr int blocks_per_row = N_PARTICLES % BLOCK_SIZE == 0 ?
                                       N_PARTICLES / BLOCK_SIZE : N_PARTICLES / BLOCK_SIZE + 1;
    constexpr int n_blocks = blocks_per_row * (blocks_per_row + 1) / 2;

    // allocate pos, vel
    auto *pos = static_cast<double *>(malloc(N_PARTICLES * DIM * sizeof(double)));
    auto *vel = static_cast<double *>(malloc(N_PARTICLES * DIM * sizeof(double)));

    // allocate matrix, pos, and vel array on device
    double *device_matrix, *device_pos, *device_vel;
    cudaMalloc(&device_matrix, N_PARTICLES*N_PARTICLES*sizeof(double)); checkCudaError("cudaMalloc1");
    cudaMalloc(&device_pos, N_PARTICLES * DIM * sizeof(double));        checkCudaError("cudaMalloc2");
    cudaMalloc(&device_vel, N_PARTICLES * DIM * sizeof(double));        checkCudaError("cudaMalloc2.5");

    // fill initial conditions
    fill_array(pos, N_PARTICLES*DIM);
    fill_array(vel, N_PARTICLES*DIM);

    // copy initial condition on deviuce
    cudaMemcpy(device_pos, pos, N_PARTICLES * DIM * sizeof(double), cudaMemcpyHostToDevice); checkCudaError("cudaMalloc4");
    cudaMemcpy(device_vel, vel, N_PARTICLES * DIM * sizeof(double), cudaMemcpyHostToDevice); checkCudaError("cudaMalloc4.5");

    // allocate force array on device
    double *device_force;
    cudaMalloc(&device_force, N_PARTICLES * DIM * sizeof(double));

    // launch kernel
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim(n_blocks);

    for (int cmp = 0; cmp < DIM; cmp++) {
        calculate_pairwise_force_component<<<grid_dim, block_dim>>>(device_pos, cmp, device_matrix, N_PARTICLES, blocks_per_row);
        checkCudaError("kernel 1 launch");
        cudaDeviceSynchronize();

        // calculate total force (1 component) on each particle
        sum_over_rows<<<n_blocks, BLOCK_SIZE>>>(device_matrix, device_force + (cmp * N_PARTICLES), N_PARTICLES);
        checkCudaError("kernel 2 launch");
        cudaDeviceSynchronize();
    }

    // apply force on particles
    apply_motion<<<N_PARTICLES % 1024 == 0 ? N_PARTICLES / 1024 : N_PARTICLES / 1024 + 1,1024>>>(device_pos, device_vel, device_force, N_PARTICLES, forwardEuler, D_T);
    cudaDeviceSynchronize();

    // copy back new position
    cudaMemcpy(pos, device_pos, N_PARTICLES * DIM * sizeof(double), cudaMemcpyDeviceToHost); checkCudaError("cudaMalloc9");
    cudaMemcpy(vel, device_vel, N_PARTICLES * DIM * sizeof(double), cudaMemcpyDeviceToHost); checkCudaError("cudaMalloc9.5");


    // free space on Host and device
    cudaFree(device_matrix);

    free(pos);  cudaFree(device_pos);
    free(vel);  cudaFree(device_vel);

    cudaFree(device_force);

    return EXIT_SUCCESS;
}