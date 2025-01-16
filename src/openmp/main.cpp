#include <iostream>
#include <chrono>
#include <random>

#include "Solver.cpp"
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

int main() {

    constexpr double total_time = 200.0;
    constexpr double delta = 0.01;
    cout << fixed << setprecision(numeric_limits<double>::digits10 + 1);

    // Initializing positions and velocities randomly
    vector<Vector<DIM>> positions(NUM_PARTICLES);
    vector<Vector<DIM>> velocities(NUM_PARTICLES);

    for (int i = 0; i < NUM_PARTICLES; i++) {
        positions[i] = generateRandomPointOnUnitSphere(5.0);
        velocities[i] = generateRandomVector() * 0.1;
    }
    //Initializing masses
    vector<double> masses(NUM_PARTICLES);
    for (int i = 0; i < NUM_PARTICLES; i++) {
        masses[i] = 0.1;
    }

    Solver<NUM_PARTICLES, DIM> solverForwardEuler(total_time, delta, masses, positions, velocities);
    Solver<NUM_PARTICLES, DIM> solverLeapFrog(total_time, delta, masses, positions, velocities);

    cout << "Initial Forward Euler Energy: " << solverForwardEuler.compute_energy() << endl;
    cout << "Initial Leapfrog Energy: " << solverLeapFrog.compute_energy() << endl;


    solverForwardEuler.simulateForwardEuler("output_forward_euler.txt");
    solverLeapFrog.simulateLeapFrog("output_leapfrog.txt");

    cout << "Final Forward Euler Energy: " << solverForwardEuler.compute_energy() << endl;
    cout << "Final Leapfrog Energy: " << solverLeapFrog.compute_energy() << endl;


    return 0;
}
