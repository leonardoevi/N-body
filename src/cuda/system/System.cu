#include "System.h"

int System::initialize_device() {
    int errors = 0;

    // allocate memory
    cudaMalloc(&this->d_pos, sizeof(double) * DIM * n_particles);   errors += checkCudaError("cudaMalloc d_pos");
    cudaMalloc(&this->d_vel, sizeof(double) * DIM * n_particles);   errors += checkCudaError("cudaMalloc d_vel");

    cudaMalloc(&this->d_mass, sizeof(double) * n_particles);        errors += checkCudaError("cudaMalloc d_mass");

    for (int i = 0; i < DIM; i++) {
        cudaMalloc(&this->d_acc_matrix[i], sizeof(double) * n_particles * n_particles);   errors += checkCudaError(("cudaMalloc d_force_matrix[" + std::to_string(i) + "]").c_str());
    }

    cudaMalloc(&this->d_acc_tot, sizeof(double) * DIM * n_particles);    errors += checkCudaError("cudaMalloc d_force_tot");

    if (errors != 0) return errors;

    // copy data to device
    cudaMemcpy(this->d_pos, pos.get(), sizeof(double) * DIM * n_particles, cudaMemcpyHostToDevice); errors += checkCudaError("cudaMemcpy pos");
    cudaMemcpy(this->d_vel, vel.get(), sizeof(double) * DIM * n_particles, cudaMemcpyHostToDevice); errors += checkCudaError("cudaMemcpy vel");

    cudaMemcpy(this->d_mass, mass.get(), sizeof(double) * n_particles, cudaMemcpyHostToDevice);     errors += checkCudaError("cudaMemcpy mess");

    // initialize cuda streams
    for (auto & stream : streams)
        cudaStreamCreate(& stream);

    std::cout << "Done initializing, total errors: " << errors << std::endl;

    return errors;
}

void System::device_compute_acceleration(const dim3 grid_dim_2D,const dim3 block_dim_2D, const int grid_dim_1D, const int block_dim_1D, const int blocks_per_row) {
    for (int i = 0; i < DIM; i++) {
        if constexpr (BETTER_MATRIX_CALC) {
            calculate_pairwise_acceleration_component_opt<<<grid_dim_2D, block_dim_2D, DIM * block_dim_2D.x * sizeof(double), streams[i]>>>(d_pos, d_mass, i, d_acc_matrix[i], n_particles, blocks_per_row);
        } else
            calculate_pairwise_acceleration_component<<<grid_dim_2D, block_dim_2D, 0, streams[i]>>>
                (d_pos, d_mass, i, d_acc_matrix[i], n_particles, blocks_per_row);

        if constexpr (!BETTER_REDUCTION) {
            sum_over_rows<<<grid_dim_1D, block_dim_1D, 0 ,streams[i]>>>
                (d_acc_matrix[i], (d_acc_tot + i * n_particles), n_particles);
        } else {
            const int B = 512; // Threads per block
            const int A = 16;
            const dim3 gridSize((n_particles + A - 1) / A, (n_particles + B*2 -1) / (B*2)); // Number of blocks

            reduceSum_rows_parallel<<<gridSize, B, B * 2 * sizeof(double), streams[i]>>>(d_acc_matrix[i], n_particles, A);
            sumRowsInterleaved<<<grid_dim_1D , block_dim_1D, 0, streams[i]>>>(d_acc_matrix[i], (d_acc_tot + i * n_particles), n_particles, B * 2);
        }
    }
}

void System::write_state() {
    outFile << t_curr << std::endl;

    // on each row (assuming DIM = 3) we expect
    // pos_x pos_y pos_z vel_x vel_y vel_z of the j-th particle
    for (int j = 0; j < n_particles; j++) {
        for (int k = 0; k < DIM; k++)
            outFile << pos[j + k * n_particles] << " ";
        for (int k = 0; k < DIM; k++)
            outFile << vel[j + k * n_particles] << " ";
        outFile << "\n";
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
        cudaFree(this->d_acc_matrix[i]);

    cudaFree(this->d_acc_tot);

    // request the slave thread to terminate
    pthread_mutex_lock(&mutex);
    kys = true;
    pthread_mutex_unlock(&mutex);
    pthread_cond_signal(&cond);

    std::cout << "Waiting for threads to finish" << std::endl;

    // wait for slave thead to terminate
    pthread_join(system_printer, nullptr);

    std::cout << "Successfully released System Object" << std::endl;
}
