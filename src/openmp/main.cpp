#include <iostream>
#include <chrono>
#include <random>

#include "Solver/Solver.h"
#include "Solver/ForwardEulerSolver.h"
#include "Solver/LeapFrogSolver.h"
#include "../../include/define.h"

void initialize_particles(std::vector<Vector>& positions, std::vector<Vector>& velocities, int numParticles, double radius, int layers) {
    const double gravitationalConstant = 9.81;

    // Total number of particles per layer
    int particlesPerLayer = numParticles / layers;

    // Alternating velocity direction
    int direction = 1;

    for (int layer = 0; layer < layers; ++layer) {
        // Current radius for this layer
        double currentRadius = radius * (layer + 1) / layers;

        for (int j = 0; j < particlesPerLayer; ++j) {
            // Particle index in the overall array
            int index = layer * particlesPerLayer + j;

            // Random azimuthal angle theta in the range [0, 2π)
            double theta = (j / (double)particlesPerLayer) * 2 * M_PI;

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

int main() {

    int dimension = DIM;
    int num_particles = NUM_PARTICLES;

    constexpr double total_time = 20.0;
    constexpr double delta = 0.001;
    std::cout << std::fixed << std::setprecision(std::numeric_limits<double>::digits10 + 1);

    // Initializing positions and velocities randomly
    std::vector<Vector> positions(num_particles, Vector(dimension));
    std::vector<Vector> velocities(num_particles, Vector(dimension));
    std::vector<double> masses(num_particles, 0.1);

    const double radius = 10.0;
    const int layers = 2;
    initialize_particles(positions, velocities, num_particles, radius, layers);

    //Initializing masses
    for (int i = 0; i < num_particles; i++) {
        masses[i] = 1.0;
    }

    std::unique_ptr<Solver> forwardEulerSolver = std::make_unique<ForwardEulerSolver>(dimension, num_particles, total_time, delta, masses, positions, velocities);
    std::unique_ptr<Solver> leapFrogSolver = std::make_unique<LeapFrogSolver>(dimension, num_particles, total_time, delta, masses, positions, velocities);

    forwardEulerSolver->simulate("output_forward_euler.txt");
    leapFrogSolver->simulate("output_leapfrog.txt");


    return 0;
}
