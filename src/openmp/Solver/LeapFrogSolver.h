#ifndef LEAPFROGSOLVER_H
#define LEAPFROGSOLVER_H

#include "Solver.h"
#include "../Vector.h"
//@note it's better to give extension .hpp to C++ header file
//to indicate that they are not C header file

class LeapFrogSolver : public Solver {
  public:
  //@note this constructor just calls the base one. You could have saved useless typing 
  // by delegating the constructor:
  // using Solver::Solver
    LeapFrogSolver(int dimension, int num_particles, double total_time, double delta_time,
                   const std::vector<double>& masses,
                   const std::vector<Vector>& init_positions,
                   const std::vector<Vector>& init_velocities)
    : Solver(dimension, num_particles, total_time, delta_time, masses, init_positions, init_velocities) {}

    void apply_motion() override;
    void simulate(const std::string &output_file_name) override;
    void end_timer() override;

};



#endif //LEAPFROGSOLVER_H
