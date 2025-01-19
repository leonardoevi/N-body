#include <iostream>

#include "include/defines.h"
#include "kernels/System.h"

#include <chrono>

int main() {
    std::cout << N_PARTICLES << " particles." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    auto pos = std::make_unique<double[]>(DIM * N_PARTICLES);

    auto vel = std::make_unique<double[]>(DIM * N_PARTICLES);

    fill_spiral_3D(pos.get(), vel.get(), 0, N_PARTICLES, 0, 2, 10, 0.5, -10, N_PARTICLES);

    auto mass = std::make_unique<double[]>(N_PARTICLES);
    for (int i = 0; i < N_PARTICLES; i++)
        mass[i] = random_r();

    System system(N_PARTICLES, 2, 0.001, leapFrog, (std::move(pos)), (std::move(vel)), (std::move(mass)));
    if (system.initialize_device() != 0)
        return EXIT_FAILURE;
    system.simulate("../out/out.txt");

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000l << " seconds" << std::endl;


    return EXIT_SUCCESS;
}
