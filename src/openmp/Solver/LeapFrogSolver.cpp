#include "Solver.h"
#include "../Vector.hpp"


template <unsigned int number_particles, unsigned int dimension> class LeapFrogSolver : public Solver<number_particles, dimension> {

    public:
        LeapFrogSolver(double total_time, double delta_time,
             const vector<double>& mass,
             const vector<Vector<dimension>>& initial_positions,
             const vector<Vector<dimension>>& initial_velocities)
          : Solver<number_particles, dimension>(total_time, delta_time, mass, initial_positions, initial_velocities){}

        void applyMotion() override {
        #pragma omp for
            for (int i = 0; i < number_particles; ++i) {
                Vector<dimension> force;
                for (int j = 0; j < number_particles; ++j) {
                    force += this->accelerations[i][j];
                }
                this->velocities[i] += force * 0.5 * this->delta_time;
                this->positions[i] += this->velocities[i] * this->delta_time;
            }
            this->computeMatrix();
        #pragma omp for
            for (int i = 0; i < number_particles; ++i) {
                Vector<dimension> force;
                for (int j = 0; j < number_particles; ++j) {
                    force += this->accelerations[i][j];
                }
                this->velocities[i] += force * 0.5 * this->delta_time;
            }
          }

};

