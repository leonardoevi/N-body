#include "System.h"

int System::initialize_device() {
    int errors = 0;

    // allocate memory
    std::cout << "Allocating device memory." << std::endl;
    cudaMalloc(&this->d_pos, sizeof(double) * DIM * n_particles);   errors += checkCudaError("cudaMalloc d_pos");
    cudaMalloc(&this->d_vel, sizeof(double) * DIM * n_particles);   errors += checkCudaError("cudaMalloc d_vel");

    cudaMalloc(&this->d_mass, sizeof(double) * n_particles);        errors += checkCudaError("cudaMalloc d_mass");

    for (int i = 0; i < DIM; i++) {
        cudaMalloc(&this->d_force_matrix[i], sizeof(double) * n_particles * n_particles);   errors += checkCudaError(("cudaMalloc d_force_matrix[" + std::to_string(i) + "]").c_str());
        cudaMalloc(&this->d_force_tot[i], sizeof(double) * n_particles);                    errors += checkCudaError(("cudaMalloc d_force_tot[" + std::to_string(i) + "]").c_str());
    }

    // copy data to device
    std::cout << "Moving data to device." << std::endl;
    cudaMemcpy(this->d_pos, pos.get(), sizeof(double) * DIM * n_particles, cudaMemcpyHostToDevice); errors += checkCudaError("cudaMemcpy pos");
    cudaMemcpy(this->d_vel, vel.get(), sizeof(double) * DIM * n_particles, cudaMemcpyHostToDevice); errors += checkCudaError("cudaMemcpy vel");

    cudaMemcpy(this->d_mass, mass.get(), sizeof(double) * n_particles, cudaMemcpyHostToDevice);     errors += checkCudaError("cudaMemcpy mess");

    std::cout << "Done initializing, total errors: " << errors << std::endl;

    return errors;
}

System::~System() {
    // memory held by std::unique_ptr should be automatically released

    cudaFree(this->d_pos);
    cudaFree(this->d_vel);
    cudaFree(this->d_mass);

    for (int i = 0; i < DIM ; i++) {
        cudaFree(this->d_force_matrix[i]);
        cudaFree(this->d_force_tot[i]);
    }

    std::cout << "Successfully released System Object" << std::endl;
}
