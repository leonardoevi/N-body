#ifndef FUTILS_H
#define FUTILS_H

#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <cmath>
#include "../include/defines.h"
#include "cuda_runtime.h"

/**
 * Prints a message on std::err in case of any errors signaled by the device
 * @param message message to output in case of errors
 * @return 0 if no errors occurred, 1 otherwise
 */
int checkCudaError(const char* message);

void printMatrix(const std::string& name, const double* matrix, int rows, int cols);

/**
 * Fills the array with random doubles in [0, 1]
 * @param arr pointer to the array
 * @param size size of the array
 */
void fill_array(double* arr, unsigned int size);

void printArray(const std::string& name, const double* arr, int size);

double check_array_force_component(const double* array, const double* pos, const unsigned int component, const unsigned int size);

/**
 *
 * @param pos position matrix
 * @param vel velocity matric
 * @param start index of the first particle to include in the ring
 * @param end index of the first particle not to include in the ring (start <= end)
 * @param axis normal axis to the ring (x = 0, y = 1, z = 2)
 * @param r_in internal radius of the ring
 * @param r_d radius of a section of the ring
 * @param v velocity of rotation
 * @param n_part number of columns un the pos and velocity matrix
 */
void fill_donut_3D(double* pos, double* vel, int start, int end, int axis, double r_in, double r_d, double v, unsigned int n_part);

/**
 *
 * @param pos position matrix
 * @param vel velocity matric
 * @param start index of the first particle to include in the ring
 * @param end index of the first particle not to include in the ring (start <= end)
 * @param axis normal axis to the ring (x = 0, y = 1, z = 2)
 * @param n_arms number of arms in the spiral
 * @param n_rounds number of rotation each arm makes around the center
 * @param r_in external radius of the spiral
 * @param r_d radius of a section of an arm
 * @param v velocity of rotation
 * @param n_part number of columns un the pos and velocity matrix
 */
void fill_spiral_3D(double* pos, double* vel, int start, int end, int axis, int n_arms, double n_rounds, double r_in, double r_d, double v, unsigned int n_part);

/**
 * @return a random double in [0, 1]
 */
inline double random_r() {
    return static_cast<double>(std::rand()) / RAND_MAX;
}

/**
 * @return a number picked from a custom probability distribution
 */
inline double my_dist(const double factor) {
    return -factor * (1.0 / (random_r() - 1.0 + 0.2) - 1.0);
}


#endif //FUTILS_H
