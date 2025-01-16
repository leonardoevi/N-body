#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <iomanip>
#include <vector>

#define N_PARTICLES 10000  // current max is circa 31500
#define BLOCK_SIZE 32       // blocks will be 2D: BLOCK_SIZE*BLOCK_SIZE
#define DIM 2

#define COMPONENT 1     // x = 0, y = 1, z = 2. Must be < DIM

/**
 * @param index
 * @param n size of the 2D matrix
 * @return the pair of indexes (i, j) that correspond to the index-th element
 * in a 2D square matrix of size n, numbered from 0, row major, skipping elements
 * under the main diagonal. In a 3x3 the numbering would be:
 * [0 1 2]
 * [- 3 4]
 * [- - 5]
 */
__host__ __device__ void mapIndexTo2D(unsigned int index, unsigned int n, unsigned int &i, unsigned int &j);

/**
 * This kernel computes one component of all the forces between the particles and writes it in the matrix.
 * @param pos a dim x n_particles matrix, each column contains the coordinates of a particle: (x, y, z) in the case dim=3
 * @param component value between 0 and dim-1 specifies the component of the force needed (x, y, or z)
 * @param matrix [i][j] will contain the value of the force along the specified axis on particle i caused by particle j.
 * It has to be an n_particles x n_particles matrix.
 */
__global__ void calculate_pairwise_force_component(const double* pos, const unsigned int component, double* matrix, const unsigned int n_particles, const unsigned int dim, const unsigned int blocks_per_row) {
    unsigned int i, j; mapIndexTo2D(blockIdx.x, blocks_per_row, i, j);
    i = i * blockDim.x + threadIdx.x;
    j = j * blockDim.y + threadIdx.y;

    if (i < n_particles && j < n_particles && i < j)
        if (j > i) {
            //double* di =(double*) malloc(dim * sizeof(double));
            double di[DIM];
            for (unsigned int k = 0; k < dim; ++k)
                di[k] = pos[k * n_particles + j] - pos[k * n_particles + i];

            double d = 0;
            for (unsigned int k = 0; k < dim; ++k)
                d += di[k] * di[k];
            d = std::sqrt(d);

            if (d > 0) {
                matrix[i * n_particles + j] = di[component] / (d * d * d);
                matrix[j * n_particles + i] = -matrix[i * n_particles + j];
            }
            free(di);
        }
}

__global__ void sum_over_rows(const double* mat, double* arr, const unsigned int matrix_edge_size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < matrix_edge_size) {
        arr[i] = 0;
        for (unsigned int j = 0; j < matrix_edge_size; j++)
            arr[i] += mat[i * matrix_edge_size + j];
    }
}

void checkCudaError(const char* message);
void printMatrix(const std::string& name, const double* matrix, int rows, int cols);
void fill_array(double* arr, unsigned int size);
void printArray(const std::string& name, const double* arr, int size);
double check_array_force_component(const double* array, const double* pos, const unsigned int component, const unsigned int size);

int main() {
    std::cout << N_PARTICLES << " particles." << std::endl;

    // calculate the number of blocks needed
    constexpr int blocks_per_row = N_PARTICLES % BLOCK_SIZE == 0 ?
                                       N_PARTICLES / BLOCK_SIZE : N_PARTICLES / BLOCK_SIZE + 1;
    constexpr int n_blocks = blocks_per_row * (blocks_per_row + 1) / 2;

    // allocate matrix and array
    auto *matrix = static_cast<double *>(malloc(N_PARTICLES * N_PARTICLES * sizeof(double)));
    auto *pos = static_cast<double *>(malloc(N_PARTICLES * DIM * sizeof(double)));

    // allocate matrix and array on device
    double *device_matrix, *device_pos;
    cudaMalloc(&device_matrix, N_PARTICLES*N_PARTICLES*sizeof(double)); checkCudaError("cudaMalloc1");
    cudaMalloc(&device_pos, N_PARTICLES * DIM * sizeof(double)); checkCudaError("cudaMalloc2");

    //pos[0] = 1.0; pos[1] = 3.0; pos[2] = 1.0; pos[3] = 3.0;
    //pos[4] = 1.0; pos[5] = 1.0; pos[6] = 3.0; pos[7] = 3.0;
    fill_array(pos, N_PARTICLES * DIM);
    cudaMemcpy(device_pos, pos, N_PARTICLES * DIM * sizeof(double), cudaMemcpyHostToDevice); checkCudaError("cudaMalloc4");

    // launch kernel
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim(n_blocks);
    calculate_pairwise_force_component<<<grid_dim, block_dim>>>(device_pos, COMPONENT, device_matrix, N_PARTICLES, DIM, blocks_per_row);
    checkCudaError("kernel 1 launch");
    cudaDeviceSynchronize();

    double *device_f_tot_x;
    cudaMalloc(&device_f_tot_x, N_PARTICLES * sizeof(double)); checkCudaError("cudaMalloc8");
    sum_over_rows<<<n_blocks, BLOCK_SIZE>>>(device_matrix, device_f_tot_x, N_PARTICLES);
    checkCudaError("kernel 2 launch");
    cudaDeviceSynchronize();

    auto *f_tot = static_cast<double *>(malloc(N_PARTICLES * sizeof(double)));
    cudaMemcpy(f_tot, device_f_tot_x, N_PARTICLES * sizeof(double), cudaMemcpyDeviceToHost);

    // copy back matrix data
    //cudaMemcpy(matrix, device_matrix, N_PARTICLES * N_PARTICLES * sizeof(double), cudaMemcpyDeviceToHost);

    //printMatrix("matrix", matrix, N_PARTICLES, N_PARTICLES);
    //printArray("f_tot", f_tot, N_PARTICLES);

    std::cout << "Kernel is done! Checking result:" << std::endl;
    std::flush(std::cout);

    // check correctness of result
    const double error = check_array_force_component(f_tot, pos, COMPONENT, N_PARTICLES);
    std::cout << "Total result error is: " << error << std::endl;

    // free space on Host and device
    free(matrix);               free(pos);
    cudaFree(device_matrix);    cudaFree(device_pos);

    free(f_tot);
    cudaFree(device_f_tot_x);

    return EXIT_SUCCESS;
}

__host__ __device__ void
mapIndexTo2D(unsigned int index, const unsigned int n, unsigned int &i, unsigned int &j) {
    // Find row and column
    int row = 0;
    while (index >= n - row) {
        index -= (n - row);
        row++;
    }
    unsigned int col = row + index;

    i = row;
    j = col;
}

// DEBUG FUNCTIONS //
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
        arr[i] = rand() / (double) RAND_MAX;
    }
}

void checkCudaError(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error after " << message << ": " << cudaGetErrorString(error) << std::endl;
        exit(-1); // Terminate the program on error
    }
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