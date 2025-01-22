#ifndef FORWARDEULERSOLVER_H
#define FORWARDEULERSOLVER_H

#include "Solver.h"
#include "../Vector.h"

class ForwardEulerSolver : public Solver {
    public:
      ForwardEulerSolver(int dimension, int num_particles, double total_time, double delta_time,
                         const std::vector<double>& masses,
                         const std::vector<Vector>& init_positions,
                         const std::vector<Vector>& init_velocities)
      : Solver(dimension, num_particles, total_time, delta_time, masses, init_positions,init_velocities) {}

      void apply_motion() override;
      void simulate(const std::string &output_file_name) override;
      void end_timer() override;
};


#endif //FORWARDEULERSOLVER_H
