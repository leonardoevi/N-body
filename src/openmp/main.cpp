#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

// Function to compute gravitational forces
void computeForces(const double* positions, double* forces, size_t n) {
    // Initialize the output force matrix with zeros
    for (size_t i = 0; i < 2 * n; ++i) {
        forces[i] = 0.0;
    }

    // Compute gravitational forces
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i != j) {
                // Compute the difference in positions
                double dx = positions[j] - positions[i];
                double dy = positions[n + j] - positions[n + i];

                // Compute the distance squared
                double distSquared = dx * dx + dy * dy;

                // Compute the distance (avoiding division by zero)
                double dist = sqrt(distSquared) + 1e-8;

                // Compute the gravitational force magnitude (G = 1 for simplicity)
                double forceMagnitude = 1.0 / distSquared;

                // Compute force components and accumulate
                forces[i] += forceMagnitude * dx / dist;
                forces[n + i] += forceMagnitude * dy / dist;
            }
        }
    }
}

int main() {
    // Example input: 2x3 matrix represented as a flat array
    double positions[] = {1.0, 3.0, 1.0, 3.0, 1.0, 1.0, 3.0, 3.0}; // x1, x2, x3, y1, y2, y3
    size_t n = 4; // Number of particles

    // Output array for forces
    double forces[8]; // 2 * n elements

    // Compute the forces
    computeForces(positions, forces, n);

    // Output the forces
    for (size_t i = 0; i < n; ++i) {
        cout << "Particle " << i << ": Force = ("
             << forces[i] << ", " << forces[n + i] << ")\n";
    }

    return 0;
}
