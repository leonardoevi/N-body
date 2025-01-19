#ifndef FUTILS_H
#define FUTILS_H

#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <cmath>
#include "../include/defines.h"
#include "cuda_runtime.h"

int checkCudaError(const char* message);

void printMatrix(const std::string& name, const double* matrix, int rows, int cols);

void fill_array(double* arr, unsigned int size);

void printArray(const std::string& name, const double* arr, int size);

double check_array_force_component(const double* array, const double* pos, const unsigned int component, const unsigned int size);

void fill_donut_3D(double* pos, double* vel, int start, int end, int axis, double r_in, double r_d, double v, unsigned int n_part);

inline double random_r() {
    return static_cast<double>(std::rand()) / RAND_MAX;
}


#endif //FUTILS_H
