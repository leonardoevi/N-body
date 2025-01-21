#ifndef SOLVER_H
#define SOLVER_H

#include <iostream>
#include <fstream>
#include "../Vector.h"

/**
 * Class Solver initializes the solver with the needed attributes
 * @tparam number_particles total number of particles of the system
 * @tparam dimension dimension of each vector of the system (either 2 or 3)
 */
template <unsigned int number_particles, unsigned int dimension> class Solver {
  protected:
    // Attributes needed for profiling
    chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
    chrono::high_resolution_clock::time_point end = chrono::high_resolution_clock::now();

    // Gravitational constant
    constexpr static double G = 10.0;
    double total_time, delta_time, current_time;

    vector<double> mass;
    vector<Vector<dimension>> positions, velocities;
    // Matrix of accelerations; accelerations[i][j] is the acceleration of particle i due to particle j, obviously is a upper triangular matrix
    vector<vector<Vector<dimension>>> accelerations;

  public:
    /**
     * Constructor of Solver class
     * @param total_time amount of time to run the simulation
     * @param delta_time time step of each iteration of the simulation
     * @param mass array of particles' masses
     * @param initial_positions array of particles' initial positions
     * @param initial_velocities array of particles' initial velocities
     */
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

    /**
     * Getter function
     * @return array of particles' masses
     */
    [[nodiscard]] vector<double> get_mass() const  {
                return mass;
            }

    /**
     * Getter function
     * @return array of particles' positions
     */
    [[nodiscard]] vector<Vector<dimension>> get_positions() const {
                return positions;
            }

    /**
     * Getter function
     * @return array of particles' positions
     */
    [[nodiscard]] vector<Vector<dimension>> get_velocities() const{
                return velocities;
            }

    /**
     * Getter function
     * @return matrix of accelerations
     */
    [[nodiscard]] vector<vector<Vector<dimension>>> get_accelerations() const{
                return accelerations;
            }

    /**
     * Function that return maximum
     * @param a first parameter
     * @param b second parameter
     * @return maximum between a or b
     */
    static double max(const double a, const double b) {
                return a > b ? a : b;
            }

    /**
     * Called at each time step, computes the matrix of accelarations
     */
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

    /**
     * Solves each time step of the system and writes positions and velocities in output file
     * @param output_file_name Name of the output file to write positions and velocities
     */
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


    /**
     * Writes masses in poutput file
     * @param file file already opened to write masses on
     */
    void writeMasses(std::ofstream& file) {
                // writing masses in output file
                for (int i = 0; i < number_particles -1 ; ++i) {
                    file << mass[i] << " ";
                }
                file << mass[number_particles - 1] << "\n";
            }

    /**
     * Starts the timer before the while loop of the solver starts in order to  profile
     */
    void startTimer() {
                start = std::chrono::high_resolution_clock::now();
            }
    /**
     * Ends the timer after the while loop of the solver ends in order to profile
     */
    void endTimer() {
                end = std::chrono::high_resolution_clock::now();
                chrono::duration<double> elapsed = end - start;
                cout << "Integration Parallelized took " << elapsed.count() << " seconds to complete.\n";

            }

    /**
     * Writes the current positions and velocities at time step current_time
     * @param file file already opened to write status on
     */
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

    /**
     * Computes the total energy of the system. Needed to compare between Leapfrog Integration and Forward Euler Integration
     * @return the sum of kinetic and potential energy
     */
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

    /**
     * Computes the next position and velocity of each particle. Will be overridden by the 2 child classes
     */
    virtual void applyMotion() = 0;

    /**
     * Destructor
     */
    virtual ~Solver() = default;
};

#endif //SOLVER_H
