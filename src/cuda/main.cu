#include <iostream>

#include "include/defines.h"
#include "kernels/System.h"

int main() {
    std::cout << N_PARTICLES << " particles." << std::endl;

    auto pos = std::make_unique<double[]>(DIM * N_PARTICLES);
    pos[0] = 1.0; pos[1] = 2.0; pos[2] = 3.0;

    auto vel = std::make_unique<double[]>(DIM * N_PARTICLES);

    auto mass = std::make_unique<double[]>(N_PARTICLES);
    mass[0] = 1.0; mass[1] = 0.0; mass[2] = 1.0;

    System system(N_PARTICLES, 0.1, 0.1, forwardEuler, (std::move(pos)), (std::move(vel)), (std::move(mass)));
    system.initialize_device();
    system.simulate("helo");
    system.print_state();

    return EXIT_SUCCESS;
}
