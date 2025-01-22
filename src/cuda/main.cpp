#include <iostream>
#include <filesystem>
#include <optional>
#include <fstream>
#include <chrono>

#include "include/defines.h"
#include "system/System.h"
#include "system/SystemLF.h"
#include "system/SystemFE.h"




int main(const int argc, char* argv[]) {
    /*
    const int N_PARTICLES = 1000;
    std::cout << N_PARTICLES << " particles" << std::endl << std::endl;
    std::cout << "BETTER REDUCTION  : " << BETTER_REDUCTION << std::endl << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    auto pos = std::make_unique<double[]>(DIM * N_PARTICLES);

    auto vel = std::make_unique<double[]>(DIM * N_PARTICLES);

    fill_spiral_3D(pos.get(), vel.get(), 0, N_PARTICLES, 0, 4, 1, 6, 0.15, -20, N_PARTICLES);

    auto mass = std::make_unique<double[]>(N_PARTICLES);
    for (int i = 0; i < N_PARTICLES; i++)
        mass[i] = random_r();

    System* system;

    {
        integration_type int_t = leapFrog;
        if (int_t == forwardEuler)
            system = new SystemFE(N_PARTICLES, 0.3, 0.001, (std::move(pos)), (std::move(vel)), (std::move(mass)));
        else
            system = new SystemLF(N_PARTICLES, 0.3, 0.001, (std::move(pos)), (std::move(vel)), (std::move(mass)));

        if (system->initialize_device() != 0)
            return EXIT_FAILURE;
        system->simulate("../out/out.txt");

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "\nElapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0l << " seconds" << std::endl << std::endl;
    }

    delete system;
    */

    double d_time, total_time;
    std::string int_type;
    std::string out_file_name;
    std::optional<std::string> input_file_name;


    if (argc == 6) {
        // Try simulating with a provided starting system state.
        d_time = std::stod(argv[1]);
        total_time = std::stod(argv[2]);
        int_type = argv[3];
        out_file_name = argv[4];

        input_file_name = argv[5];

    } else if (argc == 5) {
        // Simple simulation with a default system initialization
        d_time = std::stod(argv[1]);
        total_time = std::stod(argv[2]);
        int_type = argv[3];
        out_file_name = argv[4];

    } else {
        std::cerr << "Usage: ./" << std::filesystem::path(argv[0]).filename().string() << " d_time total_time {fe | lf} out_file_name [input.txt]" << std::endl;
        std::cerr << "try  : ./" << std::filesystem::path(argv[0]).filename().string() << " 0.001 0.5 lf out_test.txt" << std::endl;
        return EXIT_FAILURE;
    }

    if (int_type != "fe" && int_type != "lf") {
        std::cerr << "Third parameter must be either fe or lf" << std::endl;
        return EXIT_FAILURE;
    }

    // if a file was given, read the number of particles in the system
    std::ifstream in;
    int n_particles;
    if (input_file_name.has_value()) {
        in.open(input_file_name.value());

        if (!in.is_open()) {
            std::cerr << "Error opening file " << input_file_name.value() << std::endl;
            return EXIT_FAILURE;
        }

        in >> n_particles;

        // check compatibility with DIM parameter
        int dim;
        in >> dim;
        if (dim != DIM) {
            std::cerr << "Dimension in file: " << dim << " must be equal to: " << DIM << std::endl;
            return EXIT_FAILURE;
        }
    } else {
        n_particles = N_PARTICLES_DEFAULT;
    }

    // allocate system state variables
    auto pos = std::make_unique<double[]>(DIM * n_particles);
    auto vel = std::make_unique<double[]>(DIM * n_particles);
    auto mass = std::make_unique<double[]>(n_particles);

    for (int i = 0; i < n_particles; i++)
        in >> mass[i];

    if (in.is_open())
        for (int i = 0; i < n_particles; i++) {
            for (int j = 0; j < DIM; j++) {
                in >> pos[j * DIM + i];
            }
            for (int j = 0; j < DIM; j++) {
                in >> vel[j * DIM + i];
            }
        }
    else {
        // default system state
        fill_donut_3D(pos.get(), vel.get(), 0, N_PARTICLES_DEFAULT/2, 0, 4, 0.4, 25, N_PARTICLES_DEFAULT);
        fill_donut_3D(pos.get(), vel.get(), N_PARTICLES_DEFAULT/2, N_PARTICLES_DEFAULT, 0, 1, 0.2, -30, N_PARTICLES_DEFAULT);

        for (int i = 0; i < N_PARTICLES_DEFAULT; i++)
            mass[i] = 10 * random_r();
    }

    System* system;
    {
        auto start = std::chrono::high_resolution_clock::now();

        if (int_type == "fe")
            system = new SystemFE(n_particles, total_time, d_time, (std::move(pos)), (std::move(vel)), (std::move(mass)));
        else
            system = new SystemLF(n_particles, total_time, d_time, (std::move(pos)), (std::move(vel)), (std::move(mass)));

        if (system->initialize_device() != 0)
            return EXIT_FAILURE;

        system->simulate(out_file_name);

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "\nElapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0l << " seconds" << std::endl << std::endl;
    }
    delete system;


    return EXIT_SUCCESS;
}
