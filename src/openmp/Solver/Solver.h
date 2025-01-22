#ifndef SOLVER_H
#define SOLVER_H

#include <iostream>
#include <chrono>
#include <fstream>
#include <omp.h>
#include "../Vector.h"


class Solver {
  protected:
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    constexpr static double G = 10.0;
    int dimension, num_particles;
    double total_time, delta_time, current_time;

    std::vector<double> masses;
    std::vector<Vector> positions, velocities;
    std::vector<std::vector<Vector>> accelerations;

  public:
    Solver(const int dimension, const int num_particles, const double total_time, const double delta_time,
           const std::vector<double>& masses,
           const std::vector<Vector>& init_positions,
           const std::vector<Vector>& init_velocities)
    :
          dimension(dimension),
          num_particles(num_particles),
          total_time(total_time),
          delta_time(delta_time),
          current_time(0.0),
          masses(masses),
          positions(init_positions),
          velocities(init_velocities)
    {
      accelerations.resize(num_particles,  std::vector<Vector>(num_particles, Vector(dimension)));

    }
    static double max(const double a, const double b){
        return a > b ? a : b;
      }

    std::vector<double> get_masses() const;

    std::vector<Vector> get_positions() const;

    std::vector<Vector> get_velocities() const;

    std::vector<std::vector<Vector>> get_accelerations() const;

    void compute_matrix();

    void write_masses(std::ofstream& out) const;

    void start_timer();

    void write_status_to_file(std::ofstream& out) const;

    long double compute_energy() const;

    virtual void apply_motion() = 0;

    virtual void simulate(const std::string& output_file_name) = 0;

    virtual void end_timer() = 0;

    virtual ~Solver() = default;

};



#endif //SOLVER_H
