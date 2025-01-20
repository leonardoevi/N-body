#include <iostream>

#include "include/defines.h"
#include "system/System.h"
#include "system/SystemLF.h"
#include "system/SystemFE.h"

#include <chrono>

int main() {
    std::cout << N_PARTICLES << " particles." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    auto pos = std::make_unique<double[]>(DIM * N_PARTICLES);

    auto vel = std::make_unique<double[]>(DIM * N_PARTICLES);

    fill_spiral_3D(pos.get(), vel.get(), 0, N_PARTICLES, 0, 3, 0.75, 4, 0.15, -4, N_PARTICLES);

    auto mass = std::make_unique<double[]>(N_PARTICLES);
    for (int i = 0; i < N_PARTICLES; i++)
        mass[i] = 1.0;

    System* system;

    {
        integration_type int_t = forwardEuler;
        if (int_t == forwardEuler)
            system = new SystemFE(N_PARTICLES, 1, 0.001, (std::move(pos)), (std::move(vel)), (std::move(mass)));
        else
            system = new SystemLF(N_PARTICLES, 1, 0.001, (std::move(pos)), (std::move(vel)), (std::move(mass)));

        if (system->initialize_device() != 0)
            return EXIT_FAILURE;
        system->simulate("../out/out.txt");

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000l << " seconds" << std::endl;
    }

    delete system;

    return EXIT_SUCCESS;
}
