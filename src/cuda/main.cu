#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>

#include "include/defines.h"
#include "kernels/kernels.h"
#include "futils/futils.h"

#define COMPONENT 1     // x = 0, y = 1, z = 2. Must be < DIM

int main() {
    std::cout << N_PARTICLES << " particles." << std::endl;

    // calculate the number of blocks needed
    constexpr int blocks_per_row = N_PARTICLES % BLOCK_SIZE == 0 ?
                                       N_PARTICLES / BLOCK_SIZE : N_PARTICLES / BLOCK_SIZE + 1;
    constexpr int n_blocks = blocks_per_row * (blocks_per_row + 1) / 2;

    // allocate pos array
    auto *pos = static_cast<double *>(malloc(N_PARTICLES * DIM * sizeof(double)));

    // allocate matrix and pos array on device
    double *device_matrix, *device_pos;
    cudaMalloc(&device_matrix, N_PARTICLES*N_PARTICLES*sizeof(double)); checkCudaError("cudaMalloc1");
    cudaMalloc(&device_pos, N_PARTICLES * DIM * sizeof(double)); checkCudaError("cudaMalloc2");

    fill_array(pos, N_PARTICLES * DIM);
    cudaMemcpy(device_pos, pos, N_PARTICLES * DIM * sizeof(double), cudaMemcpyHostToDevice); checkCudaError("cudaMalloc4");

    // launch kernel
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim(n_blocks);
    calculate_pairwise_force_component<<<grid_dim, block_dim>>>(device_pos, COMPONENT, device_matrix, N_PARTICLES, blocks_per_row);
    checkCudaError("kernel 1 launch");
    cudaDeviceSynchronize();

    // allocate force array on device
    double *device_f_tot_x;
    cudaMalloc(&device_f_tot_x, N_PARTICLES * sizeof(double)); checkCudaError("cudaMalloc8");
    sum_over_rows<<<n_blocks, BLOCK_SIZE>>>(device_matrix, device_f_tot_x, N_PARTICLES);
    checkCudaError("kernel 2 launch");
    cudaDeviceSynchronize();

    // allocate force array on host
    auto *f_tot = static_cast<double *>(malloc(N_PARTICLES * sizeof(double)));
    cudaMemcpy(f_tot, device_f_tot_x, N_PARTICLES * sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "Kernel is done! Checking result:" << std::endl;
    std::flush(std::cout);

    // check correctness of result
    const double error = check_array_force_component(f_tot, pos, COMPONENT, N_PARTICLES);
    std::cout << "Total result error is: " << error << std::endl;

    // free space on Host and device
    cudaFree(device_matrix);

    free(pos);
    cudaFree(device_pos);

    free(f_tot);
    cudaFree(device_f_tot_x);

    return EXIT_SUCCESS;
}