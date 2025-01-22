#include "SystemLF.h"

void SystemLF::simulate(const std::string &out_file_name) {

    std::cout << "Running LEAP FROG simulation" << std::endl;

    // open output file
    outFile.open(out_file_name);

    // Check if the file was opened successfully
    if (!outFile) {
        std::cerr << "Error: Could not open the file " << out_file_name << " for writing." << std::endl;
        throw std::runtime_error("Error: Could not open file " + out_file_name + " for writing.");
    }

    // summon the slave thread
    pthread_create(&system_printer, nullptr, write_system_state, (void*)this);

    // write the number of particles in the system and the space dimensions.
    outFile << std::fixed << std::setprecision(std::numeric_limits<double>::digits10 + 1);
    outFile << n_particles << " " << DIM << std::endl;

    // write the mass array
    for (int i = 0; i < n_particles; i++)
        outFile << mass[i] << " ";
    outFile << std::endl;

    // ask slave thread to print the state of the system
    pthread_mutex_lock(&mutex);
    print_system = true;
    pthread_cond_signal(&cond);
    pthread_mutex_unlock(&mutex);

    // compute dimensions for kernel launch
    const int blocks_per_row = n_particles % BLOCK_SIZE == 0 ?
                               n_particles / BLOCK_SIZE : n_particles / BLOCK_SIZE + 1;
    const int n_blocks = blocks_per_row * (blocks_per_row + 1) / 2;

    // calculate grid and block size for each kernel launch
    const dim3 grid_dim_2D(n_blocks);
    const dim3 block_dim_2D(BLOCK_SIZE, BLOCK_SIZE);

    const int grid_dim_1D = n_particles % 1024 == 0 ? n_particles / 1024 : n_particles / 1024 + 1;
    const int block_dim_1D = 1024;

    // compute a_0
    device_compute_acceleration(grid_dim_2D, block_dim_2D, grid_dim_1D, block_dim_1D, blocks_per_row);
    cudaDeviceSynchronize();

    int iter = 0;
    while (this->t_curr < this->t_max) {

        // integration
        {
            // v_(i + 0.5) = v_(i) + a_i * dt / 2
            x_plus_by<<<grid_dim_1D, block_dim_1D>>>(d_vel, d_acc_tot, dt / 2.0, d_vel, n_particles);

            // x_(i + 1) = x_(i) + v_(i + 0.5) * dt
            x_plus_by<<<grid_dim_1D, block_dim_1D>>>(d_pos, d_vel, dt, d_pos, n_particles);

            // compute new acceleration vector a_(i+1)
            device_compute_acceleration(grid_dim_2D, block_dim_2D, grid_dim_1D, block_dim_1D, blocks_per_row);

            // v_(i + 1) = v_(i + 0.5) + a_(i + 0.5) * dt / 2
            x_plus_by<<<grid_dim_1D, block_dim_1D>>>(d_vel, d_acc_tot, dt / 2.0, d_vel, n_particles);
        }
        cudaDeviceSynchronize();

        // wait for consumer to end its job, acquire lock
        pthread_mutex_lock(&mutex);
        {
            // copy back pos and vel matrices from device
            cudaMemcpy(pos.get(), d_pos, sizeof(double) * DIM * n_particles, cudaMemcpyDeviceToHost);
            cudaMemcpy(vel.get(), d_vel, sizeof(double) * DIM * n_particles, cudaMemcpyDeviceToHost);

            this->t_curr += this->dt;

            print_system = true;
        }
        pthread_mutex_unlock(&mutex);
        // signal consumer memory is ready to be output
        pthread_cond_signal(&cond);

        iter ++;
        if (iter % 100 == 0) std::cout << "Iteration " << iter << "/" << static_cast<int>(t_max / dt)<< std::endl;
    }
}