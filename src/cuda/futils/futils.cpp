#include "futils.h"

double check_array_force_component(const double* array, const double* pos, const unsigned int component, const unsigned int size) {
    double error = 0.0;
    std::vector<double> f_tot(size);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i == j) continue;
            double di[DIM];
            double d = 0;
            for (unsigned int k = 0; k < DIM; ++k) {
                di[k] = pos[k * size + j] - pos[k * size + i];
                d += di[k] * di[k];
            }
            d = std::sqrt(d);
            f_tot[i] += di[component] / (d * d * d);
        }
        error += f_tot[i] - array[i];
    }

    return error;
}

void fill_array(double *arr, const unsigned int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = rand() / (double) RAND_MAX * 10;
    }
}

int checkCudaError(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error after " << message << ": " << cudaGetErrorString(error) << std::endl;
        //exit(-1); // Terminate the program on error

        return 1;
    }
    return 0;
}

void printMatrix(const std::string& name, const double* matrix, const int rows, const int cols) {
    std::cout << name << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Access the element at (i, j) in row-major order
            std::cout << matrix[i * cols + j] << "\t";
            //if ((j + 1) % BLOCK_SIZE == 0) std::cout << "  ";
        }
        std::cout << std::endl; // Move to the next row
        //if ((i + 1) % BLOCK_SIZE == 0) std::cout << std::endl;
    }
    std::cout << std::endl;
}

void printArray(const std::string& name, const double* arr, const int size) {
    std::cout << name << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < size; ++i) {
            std::cout << arr[i] << "\t";
    }
    std::cout << std::endl << std::endl;
}

void fill_donut_3D(double* pos, double* vel,
                    const int start, const int end,
                    const int axis,
                    const double r_in, const double r_d,
                    const double v, const unsigned int n_part) {

    for (unsigned int i = start; i < end && i < n_part; ++i) {
        const double theta = random_r() * 2 * M_PI;
        const double phi = random_r() * 2 * M_PI;
        const double psi = random_r() * 2 * M_PI;

        const double dx = r_d * std::sin(phi) * std::cos(psi);
        const double dy = r_d * std::sin(phi) * std::sin(psi);
        const double dz = r_d * std::cos(phi);

        const double x = r_in * std::cos(theta);
        const double y = r_in * std::sin(theta);

        const double vx = - std::sin(theta) * v;
        const double vy = std::cos(theta) * v;

        pos[i + n_part * ((axis + 0) % 3)] = x + dx;
        pos[i + n_part * ((axis + 1) % 3)] = y + dy;
        pos[i + n_part * ((axis + 2) % 3)] = dz;

        vel[i + n_part * ((axis + 0) % 3)] = vx;
        vel[i + n_part * ((axis + 1) % 3)] = vy;

    }
}

void fill_spiral_3D(double* pos, double* vel,
                    const int start, const int end,
                    const int axis, const int n_arms,
                    const double r_in, const double r_d,
                    const double v, const unsigned int n_part) {

    for (unsigned int i = start; i < end && i < n_part; ++i) {
        const double theta = 2 * M_PI * i / static_cast<double>(n_part) + (2 * M_PI / n_arms) * (i % n_arms);

        const double phi = random_r() * 2 * M_PI;
        const double psi = random_r() * 2 * M_PI;
        const double dx = r_d * std::sin(phi) * std::cos(psi);
        const double dy = r_d * std::sin(phi) * std::sin(psi);
        const double dz = r_d * std::cos(phi);

        const double rho = r_in * i / static_cast<double>(n_part);

        const double x = rho * std::cos(theta);
        const double y = rho * std::sin(theta);

        const double vx = - std::sin(theta) * v * std::sqrt(rho / 2);
        const double vy = std::cos(theta) * v * std::sqrt(rho / 2);

        pos[i + n_part * ((axis + 0) % 3)] = x  + dx ;
        pos[i + n_part * ((axis + 1) % 3)] = y  + dy ;
        pos[i + n_part * ((axis + 2) % 3)] = dz;

        vel[i + n_part * ((axis + 0) % 3)] = vx;
        vel[i + n_part * ((axis + 1) % 3)] = vy;

    }
}


