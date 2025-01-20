#include <iostream>
#include <chrono>
#include <random>

#include "Solver/Solver.h"
#include "Solver/ForwardEulerSolver.cpp"
#include "Solver/LeapFrogSolver.cpp"
#include "../../include/define.h"

Vector<DIM> generateRandomVector() {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution dis(-1.0, 1.0);

    Vector<DIM> randomVector;
    for (int i = 0; i < DIM; ++i) {
        randomVector[i] = dis(gen);
    }

    return randomVector;
}

Vector<DIM> generateRandomPointOnUnitSphere(const double rho) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    // Random azimuthal angle theta in the range [0, 2π)
    double theta = dis(gen) * 2 * M_PI;

    // Random polar angle phi in the range [0, π]
    double phi = std::acos(2 * dis(gen) - 1);  // Uniform distribution for phi

    // Convert spherical coordinates to Cartesian coordinates
    Vector<DIM> randomPoint;
    randomPoint[0] = rho *  std::sin(phi) * std::cos(theta);  // x
    randomPoint[1] = rho * std::sin(phi) * std::sin(theta);  // y
    if (DIM == 3) randomPoint[2] = rho * std::cos(phi);                    // z

    return randomPoint;
}

void initializeParticles(vector<Vector<DIM>>& positions, vector<Vector<DIM>>& velocities, int numParticles, double radius, int layers) {
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
                random_device rd;
                mt19937 gen(rd());
                uniform_real_distribution<> dis(0.0, 1.0);
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

    constexpr double total_time = 20.0;
    constexpr double delta = 0.001;
    cout << fixed << setprecision(numeric_limits<double>::digits10 + 1);

    // Initializing positions and velocities randomly
    vector<Vector<DIM>> positions(NUM_PARTICLES);
    vector<Vector<DIM>> velocities(NUM_PARTICLES);
    vector<double> masses(NUM_PARTICLES);

    const double radius = 10.0;
    const int layers = 4;
    initializeParticles(positions, velocities, NUM_PARTICLES, radius, layers);

    //Initializing masses
    for (int i = 0; i < NUM_PARTICLES; i++) {
        masses[i] = 1.0;
    }

    std::unique_ptr<Solver<NUM_PARTICLES, DIM>> forwardEulerSolver = std::make_unique<ForwardEulerSolver<NUM_PARTICLES, DIM>>(total_time, delta, masses, positions, velocities);
    std::unique_ptr<Solver<NUM_PARTICLES, DIM>> leapFrogSolver = std::make_unique<LeapFrogSolver<NUM_PARTICLES, DIM>>(total_time, delta, masses, positions, velocities);
    std::cout << "Initial Energy "<< forwardEulerSolver->compute_energy() << endl;

    forwardEulerSolver->simulate("output_interface_forward_euler.txt");
    leapFrogSolver->simulate("output_interface_leapfrog.txt");

    std::cout << "Final Forward Euler Solver Energy " << forwardEulerSolver->compute_energy() << endl;
    std::cout << "Final LeapFrog Solver Energy " << leapFrogSolver->compute_energy() << endl;

    return 0;
}
