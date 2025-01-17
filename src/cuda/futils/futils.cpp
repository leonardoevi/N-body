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


