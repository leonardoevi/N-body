#include <iostream>
#include <chrono>
#include <random>
#include <filesystem>

#include "Solver/Solver.h"
#include "Solver/ForwardEulerSolver.h"
#include "Solver/LeapFrogSolver.h"

#define NUM_PARTICLES 100
#define DIM 3

inline double random_number() {
    return static_cast<double>(std::rand()) / RAND_MAX;
}

void initialize_particles(std::vector<Vector>& positions, std::vector<Vector>& velocities,
                          const int num_particles, const int dimension, const int start, const int end,
                          const double radius, const double inner_radius, const double speed) {
    for (int i = start; i < end; i++) {
        // Random angles
        const double theta = random_number() * 2 * M_PI; // Angle around the torus (major radius)
        const double phi = random_number() * 2 * M_PI;   // Angle around the tube (minor radius)

        // Position in torus
        const double x = (radius + inner_radius * std::cos(phi)) * std::cos(theta);
        const double y = (radius + inner_radius * std::cos(phi)) * std::sin(theta);
        const double z = inner_radius * std::sin(phi);

        // Assign position
        positions[i][0] = x;
        positions[i][1] = y;
        if (dimension == 3) positions[i][2] = z;

        // Set velocity tangent to the torus
        const double velocity_x = -std::sin(theta) * speed;
        const double velocity_y = std::cos(theta) * speed;

        velocities[i][0] = velocity_x;
        velocities[i][1] = velocity_y;
        if (dimension == 3) velocities[i][2] = 0.0;
    }
}

void print_usage(const char* executable_name) {
    std::cout << "Usage with specific initial positions and velocities:" << std::endl;
    std::cout << "\t" << std::filesystem::path(executable_name).filename().string() << " delta_time total_time {fe | lf} output_file_name input.txt()" << std::endl;

    std::cout << "Usage with generated initial positions and velocities:" << std::endl;
    std::cout << "\t" << std::filesystem::path(executable_name).filename().string() << " delta_time total_time {fe | lf} output_file_name" << std::endl;
}

int main(int argc, char* argv[]) {

    if (argc != 5 && argc != 6) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        double delta_time = std::stod(argv[1]);
        double total_time = std::stod(argv[2]);

        std::string integration_name = argv[3];

        if (integration_name.size() != 2) {
            std::cout << "Integration type not supported." << std::endl;
            std::cout << "Integration supported: FE or BE" << std::endl;
            return 1;
        }
        std::string output_file_name = argv[4];
        if (argc == 5) {
            int dimension = DIM;
            int num_particles = NUM_PARTICLES;

            std::cout << std::fixed << std::setprecision(std::numeric_limits<double>::digits10 + 1);
            // Initializing positions and velocities randomly
            std::vector<Vector> positions(num_particles, Vector(dimension));
            std::vector<Vector> velocities(num_particles, Vector(dimension));
            std::vector<double> masses(num_particles);

            initialize_particles(positions, velocities, num_particles, dimension,0 ,num_particles/2, 5.0, 0.5, -55.0);
            initialize_particles(positions, velocities, num_particles, dimension, num_particles/2,num_particles, 2, 0.2, 40.0);
            //Initializing masses
            for (int i = 0; i < num_particles; i++) {
                masses[i] = random_number();
            }
            if (integration_name == "lf") {

                std::unique_ptr<Solver> leapFrogSolver = std::make_unique<LeapFrogSolver>(dimension, num_particles, total_time, delta_time, masses, positions, velocities);
                std::cout << "Leap Frog solver initial energy: " << leapFrogSolver->compute_energy()<< std::endl;
                leapFrogSolver->simulate(output_file_name);
                std::cout << "Leap Frog solver final energy: " << leapFrogSolver->compute_energy()<< std::endl;

            }else if (integration_name == "fe") {

                std::unique_ptr<Solver> forwardEulerSolver = std::make_unique<ForwardEulerSolver>(dimension, num_particles, total_time, delta_time, masses, positions, velocities);
                std::cout << "Forward Euler solver initial energy: " << forwardEulerSolver->compute_energy()<< std::endl;
                forwardEulerSolver->simulate(output_file_name);
                std::cout << "Forward Euler solver final energy: " << forwardEulerSolver->compute_energy()<< std::endl;

            }
        } else {
            std::string input_file = argv[5];
            std::ifstream file(input_file);

            if (!file.is_open()) {
                std::cout << "Error opening file" << std::endl;
                return 1;
            }

            int num_particles, dimension;
            file >> num_particles >> dimension;
            std::cout << std::fixed << std::setprecision(std::numeric_limits<double>::digits10 + 1);

            std::vector<Vector> positions(num_particles, Vector(dimension));
            std::vector<Vector> velocities(num_particles, Vector(dimension));
            std::vector<double> masses(num_particles);

            for (int i = 0; i < num_particles; i++) {
                file >> masses[i];
            }

            for (int i = 0; i < num_particles; i++) {

                for (int j = 0; j < dimension; j++) {
                    file >> positions[i][j];
                }
                for (int j = 0; j < dimension; j++) {
                    file >> velocities[i][j];
                }
            }

            if (integration_name == "lf") {

                std::unique_ptr<Solver> leapFrogSolver = std::make_unique<LeapFrogSolver>(dimension, num_particles, total_time, delta_time, masses, positions, velocities);
                std::cout << "Leap Frog solver initial energy: " << leapFrogSolver->compute_energy()<< std::endl;
                leapFrogSolver->simulate(output_file_name);
                std::cout << "Leap Frog solver final energy: " << leapFrogSolver->compute_energy()<< std::endl;

            }else if (integration_name == "fe") {

                std::unique_ptr<Solver> forwardEulerSolver = std::make_unique<ForwardEulerSolver>(dimension, num_particles, total_time, delta_time, masses, positions, velocities);
                std::cout << "Forward Euler solver initial energy: " << forwardEulerSolver->compute_energy()<< std::endl;
                forwardEulerSolver->simulate(output_file_name);
                std::cout << "Forward Euler solver final energy: " << forwardEulerSolver->compute_energy()<< std::endl;

            }
        }

    }
    catch (const std::exception& e) {
        std::cerr << "Error parsing input: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
