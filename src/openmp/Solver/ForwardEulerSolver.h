#ifndef FORWARD_EULER_SOLVER_H
#define FORWARD_EULER_SOLVER_H

#include "Solver.h"
#include "../Vector.h"

/**
 * Child class of Solver that implements Forward Euler integration
 * v^(k+1) = v^(k) + a(k) * delta_t
 * p^(k+1) = p^(k) + v^(k+1) * delta_t
 * @tparam number_particles total number of particles of the system
 * @tparam dimension n dimension of each vector of the system (either 2 or 3)
 */

template <unsigned int number_particles, unsigned int dimension> class ForwardEulerSolver : public Solver<number_particles, dimension> {

    public:
    /**
     * Constructor of ForwardEulerSolver class
     * @param total_time amount of time to run the simulation
     * @param delta_time time step of each iteration of the simulation
     * @param mass array of particles' masses
     * @param initial_positions array of particles' initial positions
     * @param initial_velocities array of particles' initial velocities
     */
    ForwardEulerSolver(double total_time, double delta_time,
             const vector<double>& mass,
             const vector<Vector<dimension>>& initial_positions,
             const vector<Vector<dimension>>& initial_velocities)
          : Solver<number_particles, dimension>(total_time, delta_time,mass, initial_positions, initial_velocities) {}

    /**
    * Overridden method to implement Forward Euler Integration
    */
    void applyMotion() override {
            #pragma omp for
            for (int i = 0; i < number_particles; ++i) {

                Vector<dimension> force;
                for (int j = 0; j < number_particles; ++j) {
                    force += this->accelerations[i][j];
                }

                this->velocities[i] += force * this->delta_time;
                this->positions[i] += this->velocities[i] * this->delta_time;
            }
          }

};
#endif //FORWARD_EULER_SOLVER_H

