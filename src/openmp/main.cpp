#include <iostream>
#include <chrono>
#include <random>
#include <filesystem>

#include "Solver/Solver.h"
#include "Solver/ForwardEulerSolver.h"
#include "Solver/LeapFrogSolver.h"
#include "../../include/define.h"

void initialize_particles(std::vector<Vector>& positions, std::vector<Vector>& velocities, const int numParticles, const double radius, const int layers) {
    // Total number of particles per layer
    int particlesPerLayer = numParticles / layers;

    // Alternating velocity direction
    int direction = 1;

    for (int layer = 0; layer < layers; ++layer) {
        // Current radius for this layer
        double currentRadius = radius * (layer + 1) / layers;

        for (int j = 0; j < particlesPerLayer; ++j) {
            constexpr double gravitationalConstant = 9.81;
            // Particle index in the overall array
            int index = layer * particlesPerLayer + j;

            // Random azimuthal angle theta in the range [0, 2π)
            const double theta = (j / static_cast<double>(particlesPerLayer)) * 2 * M_PI;

            // Position on the circle/sphere
            if (DIM == 3) {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> dis(0.0, 1.0);
                double phi = std::acos(2 * dis(gen) - 1);
                positions[index][0] = currentRadius * std::sin(phi) * std::cos(theta);
                positions[index][1] = currentRadius * std::sin(phi) * std::sin(theta);
                positions[index][2] = currentRadius * std::cos(phi);
            } else {
                positions[index][0] = currentRadius * std::cos(theta); // x-coordinate
                positions[index][1] = currentRadius * std::sin(theta); // y-coordinate
            }

            // Tangential velocity
            double speed = std::sqrt(gravitationalConstant / currentRadius) * 4.0;
            if (DIM == 3) {
                velocities[index][0] = -direction * speed * sin(theta); // Tangential component in x
                velocities[index][1] = direction * speed * cos(theta);  // Tangential component in y
                velocities[index][2] = 0.0; // No z-component for tangential velocity
            } else {
                velocities[index][0] = -direction * speed * sin(theta); // Tangential component in x
                velocities[index][1] = direction * speed * cos(theta);  // Tangential component in y
            }
        }

        // Alternate direction for the next layer
        direction = -direction;
    }
}

void print_usage(const char* executable_name) {
    std::cout << "Usage with specific initial positions and velocities:" << std::endl;
    std::cout << "\t" << std::filesystem::path(executable_name).filename().string() << " delta_time total_time {FE | LF} output_file_name input.txt()" << std::endl;

    std::cout << "Usage with generated initial positions and velocities:" << std::endl;
    std::cout << "\t" << std::filesystem::path(executable_name).filename().string() << " delta_time total_time {FE | LF} output_file_name" << std::endl;
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
            std::vector<double> masses(num_particles, 0.1);

            //Generated initial positions and vectors
            const double radius = 10.0;
            const int layers = 2;
            initialize_particles(positions, velocities, num_particles, radius, layers);

            //Initializing masses
            for (int i = 0; i < num_particles; i++) {
                masses[i] = 1.0;
            }
            if (integration_name == "LF") {

                std::unique_ptr<Solver> leapFrogSolver = std::make_unique<LeapFrogSolver>(dimension, num_particles, total_time, delta_time, masses, positions, velocities);
                std::cout << "Leap Frog solver initial energy: " << leapFrogSolver->compute_energy()<< std::endl;
                leapFrogSolver->simulate(output_file_name);
                std::cout << "Leap Frog solver final energy: " << leapFrogSolver->compute_energy()<< std::endl;

            }else if (integration_name == "FE") {

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

            if (integration_name == "LF") {

                std::unique_ptr<Solver> leapFrogSolver = std::make_unique<LeapFrogSolver>(dimension, num_particles, total_time, delta_time, masses, positions, velocities);
                std::cout << "Leap Frog solver initial energy: " << leapFrogSolver->compute_energy()<< std::endl;
                leapFrogSolver->simulate(output_file_name);
                std::cout << "Leap Frog solver final energy: " << leapFrogSolver->compute_energy()<< std::endl;

            }else if (integration_name == "FE") {

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
