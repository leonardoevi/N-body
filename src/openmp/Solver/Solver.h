#ifndef SOLVER_H
#define SOLVER_H

#include <iostream>
#include <fstream>
#include "../Vector.hpp"

template <unsigned int number_particles, unsigned int dimension> class Solver {
  protected:
    chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
    chrono::high_resolution_clock::time_point end = chrono::high_resolution_clock::now();

    constexpr static double G = 10.0;
    double total_time, delta_time, current_time;

    vector<double> mass;
    vector<Vector<dimension>> positions, velocities;
    vector<vector<Vector<dimension>>> accelerations;

  public:
    Solver (double total_time, double delta_time,
             const vector<double>& mass,
             const vector<Vector<dimension>>& initial_positions,
             const vector<Vector<dimension>>& initial_velocities)
          :
            total_time(total_time),
            delta_time(delta_time),
            current_time(0.0),
            mass(mass),
            positions(initial_positions),
            velocities(initial_velocities)
            {
            accelerations.resize(number_particles, vector<Vector<dimension>>(number_particles));
            }


            [[nodiscard]] vector<double> get_mass() const  {
                return mass;
            }
        [[nodiscard]] vector<Vector<dimension>> get_positions() const {
                return positions;
            }
        [[nodiscard]] vector<Vector<dimension>> get_velocities() const{
                return velocities;
            }
        [[nodiscard]] vector<vector<Vector<dimension>>> get_accelerations() const{
                return accelerations;
            }

        static double max(const double a, const double b) {
                return a > b ? a : b;
            }

            void computeMatrix(){
            #pragma omp for
            for (int i = 0; i < number_particles; ++i) {
                for (int j = i + 1; j < number_particles; ++j) {
                    Vector distance = positions[j] - positions[i];

                    accelerations[i][j] = distance * G * mass[j] / max(distance.norm()*distance.norm()*distance.norm(), 1.e-3);
                    accelerations[j][i] = - accelerations[i][j];
                }
            }
          }

          void simulate(const string& output_file_name){
            //Generate output file
            ofstream file(output_file_name, ios::out);
            if (!file.is_open()) {
                throw std::runtime_error("Unable to open file");
            }
            file << fixed << std::setprecision(numeric_limits<double>::digits10 + 1);
            file << number_particles << "\n";

            writeMasses(file);
            startTimer();

            for (current_time = 0.0; current_time < total_time; current_time += delta_time){
                #pragma omp parallel
                {
                    computeMatrix();
                    applyMotion();
                    #pragma omp single
                    {
                        write_status_to_file(file);
                    }
                }
            }
            endTimer();
            file.close();
        }


        void writeMasses(std::ofstream& file) {
                // writing masses in output file
                for (int i = 0; i < number_particles -1 ; ++i) {
                    file << mass[i] << " ";
                }
                file << mass[number_particles - 1] << "\n";
            }

        void startTimer() {
                start = std::chrono::high_resolution_clock::now();
            }
        void endTimer() {
                end = std::chrono::high_resolution_clock::now();
                chrono::duration<double> elapsed = end - start;
                cout << "LeapFrog Integration Parallelized took " << elapsed.count() << " seconds to complete.\n";

            }

            void write_status_to_file(std::ofstream& file) const {

                file << current_time << "\n";
                for (int i = 0; i < number_particles; ++i) {
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

        void write_energy_to_file(std::ofstream& file) const {
                if (!file.is_open()) {
                    throw std::runtime_error("File stream is not open");
                }
                file << current_time << " " << compute_energy() << endl;

            }

        [[nodiscard]] long double compute_energy() const {
                long double kinetic_energy = 0.0;
                for (int i = 0; i < number_particles; ++i) {
                    kinetic_energy += 0.5 * mass[i]  * velocities[i].norm() * velocities[i].norm();
                }

                long double potential_energy = 0.0;
                for (int i = 0; i < number_particles; ++i) {
                    for (int j = i+1; j < number_particles; ++j) {
                        potential_energy -= G * mass[i] * mass[j]/max((positions[i] - positions[j]).norm(), 1.e-3);
                    }
                }

                return kinetic_energy + potential_energy;
            }

        virtual void applyMotion() = 0;

        virtual ~Solver() = default;
};



#endif //SOLVER_H
