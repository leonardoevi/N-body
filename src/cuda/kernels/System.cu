#include "System.h"

int System::initialize_device() {
    int errors = 0;

    // allocate memory
    std::cout << "Allocating device memory..." << std::endl;
    cudaMalloc(&this->d_pos, sizeof(double) * DIM * n_particles);   errors += checkCudaError("cudaMalloc d_pos");
    cudaMalloc(&this->d_vel, sizeof(double) * DIM * n_particles);   errors += checkCudaError("cudaMalloc d_vel");

    cudaMalloc(&this->d_mass, sizeof(double) * n_particles);        errors += checkCudaError("cudaMalloc d_mass");

    for (int i = 0; i < DIM; i++) {
        cudaMalloc(&this->d_force_matrix[i], sizeof(double) * n_particles * n_particles);   errors += checkCudaError(("cudaMalloc d_force_matrix[" + std::to_string(i) + "]").c_str());
    }

    cudaMalloc(&this->d_force_tot, sizeof(double) * DIM * n_particles);    errors += checkCudaError("cudaMalloc d_force_tot");

    // copy data to device
    std::cout << "Moving initial data to device..." << std::endl;
    cudaMemcpy(this->d_pos, pos.get(), sizeof(double) * DIM * n_particles, cudaMemcpyHostToDevice); errors += checkCudaError("cudaMemcpy pos");
    cudaMemcpy(this->d_vel, vel.get(), sizeof(double) * DIM * n_particles, cudaMemcpyHostToDevice); errors += checkCudaError("cudaMemcpy vel");

    cudaMemcpy(this->d_mass, mass.get(), sizeof(double) * n_particles, cudaMemcpyHostToDevice);     errors += checkCudaError("cudaMemcpy mess");

    // initialize cuda streams
    for (auto & stream : streams)
        cudaStreamCreate(& stream);

    std::cout << "Done initializing, total errors: " << errors << std::endl;

    return errors;
}

void System::simulate(const std::string &out_file_name) {
    while (this->t_curr < this->t_max) {

        // compute dimensions for kernel launch
        const int blocks_per_row = n_particles % BLOCK_SIZE == 0 ?
                                   n_particles / BLOCK_SIZE : n_particles / BLOCK_SIZE + 1;
        const int n_blocks = blocks_per_row * (blocks_per_row + 1) / 2;

        // allocate DIM cuda streams for parallel computation of force components

        // calculate grid and block size for each kernel launch
        dim3 grid_dim_2D(n_blocks);
        dim3 block_dim_2D(BLOCK_SIZE, BLOCK_SIZE);

        int grid_dim_1D = n_particles % 1024 == 0 ? n_particles / 1024 : n_particles / 1024 + 1;
        int block_dim_1D = 1024;

        for (int i = 0; i < DIM; i++) {
            calculate_pairwise_force_component<<<grid_dim_2D, block_dim_2D, 0, streams[i]>>>
                (d_pos, d_mass, i, d_force_matrix[i], n_particles, blocks_per_row);

            sum_over_rows<<<grid_dim_1D, block_dim_1D, 0 ,streams[i]>>>
                (d_force_matrix[i], (d_force_tot + i * n_particles), n_particles);
        }

        // using the blocking behaviour of the default stream, all the force components should be computed before
        // this kernel is starting to be executed
        apply_motion<<<grid_dim_1D, block_dim_1D>>>(d_pos, d_vel, d_mass, d_force_tot, n_particles, forwardEuler, dt);

        cudaDeviceSynchronize();

        // copy back pos and vel matrices
        cudaMemcpy(pos.get(), d_pos, sizeof(double) * DIM * n_particles, cudaMemcpyDeviceToHost);
        cudaMemcpy(vel.get(), d_vel, sizeof(double) * DIM * n_particles, cudaMemcpyDeviceToHost);

        this->t_curr += this->dt;
    }
}

void System::print_state() const {
    for (int i = 0; i < DIM ; i++)
        printArray("pos[" + std::to_string(i) + "]", pos.get() + i * n_particles, n_particles);
    for (int i = 0; i < DIM ; i++)
        printArray("vel[" + std::to_string(i) + "]", vel.get() + i * n_particles, n_particles);

}

System::~System() {
    // memory held by std::unique_ptr should be automatically released

    // destroy cuda streams
    for (const auto & stream : streams)
        cudaStreamDestroy(stream);

    cudaFree(this->d_pos);
    cudaFree(this->d_vel);
    cudaFree(this->d_mass);

    for (int i = 0; i < DIM ; i++)
        cudaFree(this->d_force_matrix[i]);

    cudaFree(this->d_force_tot);

    std::cout << "Successfully released System Object" << std::endl;
}
