#ifndef SYSTEM_H
#define SYSTEM_H

#include <memory>
#include "../include/defines.h"
#include "../include/integration_type.h"
#include <cuda_runtime.h>
#include <iostream>
#include "../futils/futils.h"

class System {

    // number of particles in the system
    const unsigned int n_particles;

    // maximum time of simulation, timestep, current time of simulation
    const double t_max, dt;
    double t_curr;

    const enum integration_type integration_type;

    // =========================== POINTERS TO DATA ======================================= //

    // pointers to the position and velocity MATRICES : DIM x N_PARTICLES
    std::unique_ptr<double[]> pos, vel;

    // pointer to mass vector
    std::unique_ptr<double[]> mass;

    // pos, vel, mass, on device
    double *d_pos, *d_vel;
    double *d_mass;

    // pointer to pairwise force components MATRIX on device : N_PARTICLES x N_PARTICLES
    double* d_force_matrix[DIM];

    // pointer to resulting force components VECTORS on device : N_PARTICLES
    double* d_force_tot[DIM];

public:
    /**
     * @param n_particles_ number of particles in the system
     * @param t_max_ maximum simulation time
     * @param dt_ time step
     * @param integration_ type of integration to be used
     * @param pos_ position matrix
     * @param vel_ velocity matrix
     * @param mass_ mass vector
     */
    System(const unsigned int n_particles_, const double t_max_, const double dt_,
                    const enum integration_type integration_,
                    std::unique_ptr<double[]> pos_,
                    std::unique_ptr<double[]> vel_,
                    std::unique_ptr<double[]> mass_)
        : n_particles(n_particles_),
          t_max(t_max_),
          dt(dt_),
          t_curr(0.0),
          integration_type(integration_),
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
    void simulate(std::string &out_file_name);
};

#endif //SYSTEM_H
