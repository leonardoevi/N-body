#include "ForwardEulerSolver.h"

void ForwardEulerSolver::apply_motion(){
    #pragma omp for
        for (int i = 0; i < num_particles; ++i) {

            Vector force(dimension);
            for (int j = 0; j < num_particles; ++j) {
                force += this->accelerations[i][j];
            }

            this->velocities[i] += force * this->delta_time;
            this->positions[i] += this->velocities[i] * this->delta_time;
        }
    }

void ForwardEulerSolver::simulate(const std::string &output_file_name) {
    std::ofstream file(output_file_name);
    if (!file.is_open()){
        throw std::runtime_error("Could not open file " + output_file_name);
    }

    file << std::fixed << std::setprecision(std::numeric_limits<double>::digits10 + 1);
    file << num_particles  << " " << dimension << "\n";

    write_masses(file);
    start_timer();

    for (current_time = 0.0; current_time < total_time; current_time += delta_time){
        #pragma omp parallel
        {
            compute_matrix();
            apply_motion();
            #pragma omp single
            {
                write_status_to_file(file);
            }
        }
    }
    end_timer();
    file.close();
}

void ForwardEulerSolver::end_timer(){
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Forward Euler simulation took " << elapsed_seconds.count() << " seconds to complete" << std::endl;
}