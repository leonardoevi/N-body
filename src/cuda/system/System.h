#ifndef SYSTEM_H
#define SYSTEM_H

#include <memory>
#include "../include/defines.h"
#include "../include/integration_type.h"
#include <cuda_runtime.h>
#include <iostream>
#include "../futils/futils.h"
#include "../kernels/kernels.h"
#include <fstream>

#include <pthread.h>

class System {

protected:

    // number of particles in the system
    const unsigned int n_particles;

    // maximum time of simulation, timestep, current time of simulation
    const double t_max, dt;
    double t_curr;

    // array of CUDA streams, DIM in total, used to compute force components in parallel
    cudaStream_t streams[DIM];

    // output file stream
    std::ofstream outFile;

    // =========================== POINTERS TO DATA ======================================= //

    // pointers to the position and velocity MATRICES : DIM x N_PARTICLES
    std::unique_ptr<double[]> pos, vel;

    // pointer to mass vector
    std::unique_ptr<double[]> mass;

    // pos, vel, mass, on device
    double *d_pos, *d_vel;
    double *d_mass;

    // pointer to pairwise force components MATRIX on device : N_PARTICLES x N_PARTICLES
    double* d_acc_matrix[DIM];

    // pointer to resulting force MATRIX on device : DIM x N_PARTICLES
    double* d_acc_tot;

    // =========================== THREAD ======================================= //

    // slave thread that will print the system state while the new one is being computed
    pthread_t system_printer;

    // synchronization objects
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

    // variable to signal that the system state has to be printed
    bool print_system = false;
    bool kys = false;

public:
    /**
     * @param n_particles_ number of particles in the system
     * @param t_max_ maximum simulation time
     * @param dt_ time step
     * @param pos_ position matrix
     * @param vel_ velocity matrix
     * @param mass_ mass vector
     */
    System(const unsigned int n_particles_, const double t_max_, const double dt_,
           std::unique_ptr<double[]> pos_,
           std::unique_ptr<double[]> vel_,
           std::unique_ptr<double[]> mass_)
        : n_particles(n_particles_),
          t_max(t_max_),
          dt(dt_),
          t_curr(0.0),
          pos(std::move(pos_)),
          vel(std::move(vel_)),
          mass(std::move(mass_)) {}

    ~System();

    /**
     * @return 0 if no error occurred during the allocation of the necessary memory on the GPU
     */
    int initialize_device();

    /**
     * Simulates the system of particles, write the output file with the specified name.
     */
    void simulate(const std::string &out_file_name);

    void print_state() const;

    friend void* write_system_state(void* system);

protected:

    void write_state();

    void device_compute_acceleration(dim3 grid_dim_2D, dim3 block_dim_2D, int grid_dim_1D, int block_dim_1D, int blocks_per_row);

};

#endif //SYSTEM_H
