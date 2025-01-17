#include <iostream>

#include "include/defines.h"
#include "kernels/System.h"

int main() {
    std::cout << N_PARTICLES << " particles." << std::endl;

    auto pos = std::make_unique<double[]>(DIM * N_PARTICLES);
    //pos[0] = 0.0; pos[1] = 2.0; pos[2] = 10.0;
    for (int i = 0; i < DIM; i++)
        fill_array(pos.get(), i * N_PARTICLES);

    auto vel = std::make_unique<double[]>(DIM * N_PARTICLES);

    auto mass = std::make_unique<double[]>(N_PARTICLES);
    for (int i = 0; i < N_PARTICLES; i++)
        mass[i] = 1.0;

    System system(N_PARTICLES, 20.0, 0.01, forwardEuler, (std::move(pos)), (std::move(vel)), (std::move(mass)));
    if (system.initialize_device() != 0)
        return EXIT_FAILURE;
    system.simulate("../out/CUDA_out.txt");


    system.print_state();

    return EXIT_SUCCESS;
}
