#include <iostream>

#include "include/defines.h"
#include "kernels/System.h"

int main() {
    std::cout << N_PARTICLES << " particles." << std::endl;

    auto pos = std::make_unique<double[]>(DIM * N_PARTICLES);
    pos[0] = 1.0; pos[1] = 3.0; pos[2] = 1.0; pos[3] = 1.0;

    auto vel = std::make_unique<double[]>(DIM * N_PARTICLES);

    auto mass = std::make_unique<double[]>(N_PARTICLES);

    System system(N_PARTICLES, 1.0, 0.1, forwardEuler, (std::move(pos)), (std::move(vel)), (std::move(mass)));
    system.initialize_device();

    return EXIT_SUCCESS;
}
