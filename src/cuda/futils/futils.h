#ifndef FUTILS_H
#define FUTILS_H

#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <cmath>
#include "../include/defines.h"
#include "cuda_runtime.h"

void checkCudaError(const char* message);

void printMatrix(const std::string& name, const double* matrix, int rows, int cols);

void fill_array(double* arr, unsigned int size);

void printArray(const std::string& name, const double* arr, int size);

double check_array_force_component(const double* array, const double* pos, const unsigned int component, const unsigned int size);

#endif //FUTILS_H
