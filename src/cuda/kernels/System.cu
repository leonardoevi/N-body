#include "System.h"

void* write_system_state(void* system) {
    auto* system_obj = static_cast<System *>(system);

    while (true) {
        pthread_mutex_lock(&system_obj->mutex);

        // wait for the need of writing memory to file
        while (system_obj->print_system == false) {
            // check if termination has been requested
            if (system_obj->kys == true) {
                pthread_mutex_unlock(&system_obj->mutex);
                return nullptr;
            }

            // release the lock and wait for the condition to be signaled
            pthread_cond_wait(&system_obj->cond, &system_obj->mutex);
        }

        // now it is time to write the system to file
        system_obj->write_state();

        // set the flag
        system_obj->print_system = false;

        // job is done, release the lock
        pthread_mutex_unlock(&system_obj->mutex);
        pthread_cond_signal(&system_obj->cond);
    }
}

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

void System::simulate(const std::string &out_file_name) {

    // open output file
    outFile.open(out_file_name);

    // Check if the file was opened successfully
    if (!outFile) {
        std::cerr << "Error: Could not open the file " << out_file_name << " for writing." << std::endl;
        throw std::runtime_error("Error: Could not open file " + out_file_name + " for writing.");
    }

    // summon the slave thread
    pthread_create(&system_printer, nullptr, write_system_state, (void*)this);

    // write the number of particles in the system
    outFile << std::fixed << std::setprecision(std::numeric_limits<double>::digits10 + 1);
    outFile << n_particles << std::endl;

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

    int iter = 0;
    while (this->t_curr < this->t_max) {

        for (int i = 0; i < DIM; i++) {
            calculate_pairwise_acceleration_component<<<grid_dim_2D, block_dim_2D, 0, streams[i]>>>
                (d_pos, d_mass, i, d_acc_matrix[i], n_particles, blocks_per_row);

            sum_over_rows<<<grid_dim_1D, block_dim_1D, 0 ,streams[i]>>>
                (d_acc_matrix[i], (d_acc_tot + i * n_particles), n_particles);
        }

        // using the blocking behaviour of the default stream, all the force components should be computed before
        // this kernel is starting to be executed
        apply_motion<<<grid_dim_1D, block_dim_1D>>>(d_pos, d_vel, d_acc_tot, n_particles, dt);

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

    std::cout << "Waiting for threads to finish..." << std::endl;

    // wait for slave thead to terminate
    pthread_join(system_printer, nullptr);

    std::cout << "Successfully released System Object" << std::endl;
}
