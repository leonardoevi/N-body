#include "Solver.h"

std::vector<double> Solver::get_masses() const{
  return masses;
}

std::vector<Vector> Solver::get_positions() const{
  return positions;
}

std::vector<Vector> Solver::get_velocities() const{
  return velocities;
}

std::vector<std::vector<Vector>> Solver::get_accelerations() const{
  return accelerations;
}

void Solver::compute_matrix(){
  #pragma omp for
    for (int i = 0; i < num_particles; ++i) {
      for (int j = i + 1; j < num_particles; ++j) {
        Vector distance = positions[j] - positions[i];

        accelerations[i][j] = distance * G * masses[j] / max(distance.norm()*distance.norm()*distance.norm(), 1.e-3);
        accelerations[j][i] = - accelerations[i][j];
      }
    }
}


void Solver::write_masses(std::ofstream& file) const {
    // writing masses in output file
    for (int i = 0; i < num_particles -1 ; ++i) {
      file << masses[i] << " ";
    }
    file << masses[num_particles - 1] << "\n";
}

void Solver::start_timer(){
    start = std::chrono::high_resolution_clock::now();
}

void Solver::write_status_to_file(std::ofstream& file) const {
    file << current_time << "\n";
    for (int i = 0; i < num_particles; ++i) {
      // writing position of particle i
      for (int j = 0; j < dimension; ++j) {
        file << positions[i][j] << " ";
      }
      //writing velocity of particle i
      for (int j = 0; j < dimension - 1; ++j) {
        file << velocities[i][j] << " ";
      }
      file << velocities[i][dimension - 1] << "\n";
    }
}

long double Solver::compute_energy() const{
    long double kinetic_energy = 0.0;
    for (int i = 0; i < num_particles; ++i) {
      kinetic_energy += 0.5 * masses[i]  * velocities[i].norm() * velocities[i].norm();
    }

    long double potential_energy = 0.0;
    for (int i = 0; i < num_particles; ++i) {
      for (int j = i+1; j < num_particles; ++j) {
        potential_energy -= G * masses[i] * masses[j]/max((positions[i] - positions[j]).norm(), 1.e-3);
      }
    }

    return kinetic_energy + potential_energy;
}

