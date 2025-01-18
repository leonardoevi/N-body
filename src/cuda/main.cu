#include <iostream>

#include "include/defines.h"
#include "kernels/System.h"

#include <chrono>

int main() {
    std::cout << N_PARTICLES << " particles." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    auto pos = std::make_unique<double[]>(DIM * N_PARTICLES);
    pos[0] = -1.0; pos[1] = 0.0; pos[2] = 1.0; pos[3] = 0.0;
    pos[4] = 0.0; pos[5] = -1.0; pos[6] = 0.0; pos[7] = 1.0;
    //fill_array(pos.get(), DIM * N_PARTICLES);

    auto vel = std::make_unique<double[]>(DIM * N_PARTICLES);
    vel[0] = 0.0; vel[1] = 1.0; vel[2] = 0.0; vel[3] = -1.0;
    vel[4] = -1.0; vel[5] = 0.0; vel[6] = 1.0; vel[7] = 0.0;

    auto mass = std::make_unique<double[]>(N_PARTICLES);
    for (int i = 0; i < N_PARTICLES; i++)
        mass[i] = 1.5;

    System system(N_PARTICLES, 80.0, 0.04, forwardEuler, (std::move(pos)), (std::move(vel)), (std::move(mass)));
    if (system.initialize_device() != 0)
        return EXIT_FAILURE;
    system.simulate("../out/out.txt");

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000l << " seconds" << std::endl;


    return EXIT_SUCCESS;
}
