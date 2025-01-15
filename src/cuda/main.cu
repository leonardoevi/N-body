#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <valarray>
#include <iomanip>

#define N_PARTICLES 4  // current max is circa 44000l
#define BLOCK_SIZE 2       // blocks will be 2D: BLOCK_SIZE*BLOCK_SIZE
#define DIM 2

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
 * @param arr pointer to the input array on the device
 * @param matrix pointer to the output matrix (passed as a single array)
 * @param matrix_edge_size size of the square matrix, must be equal to the length of the input array
 * @param blocks_per_row number of blocks needed to cover a row of the matrix.
 * For a matrix of size 20 and a block of size 8 this parameter should be 3.
 */
__global__ void calculate_pairwise_difference(const double* arr, double* matrix, const unsigned int matrix_edge_size, const unsigned int blocks_per_row) {
    unsigned int i, j; mapIndexTo2D(blockIdx.x, blocks_per_row, i, j);
    i = i * blockDim.x + threadIdx.x;
    j = j * blockDim.y + threadIdx.y;

    __shared__ double shared_array[BLOCK_SIZE];
    if (threadIdx.y < blockDim.y && threadIdx.x == 0)
        shared_array[threadIdx.y] = arr[j];

    __syncthreads();

    if (i < matrix_edge_size && j < matrix_edge_size && i < j)
        if (j > i) {
            matrix[i * matrix_edge_size + j] = shared_array[threadIdx.y] - arr[i];
            matrix[j * matrix_edge_size + i] = -matrix[i * matrix_edge_size + j];
        }
}

__global__ void calculate_pairwise_force(const double* x_pos, const double* y_pos, double* matrix, const unsigned int matrix_edge_size, const unsigned int blocks_per_row) {
    unsigned int i, j; mapIndexTo2D(blockIdx.x, blocks_per_row, i, j);
    i = i * blockDim.x + threadIdx.x;
    j = j * blockDim.y + threadIdx.y;

    if (i < matrix_edge_size && j < matrix_edge_size && i < j)
        if (j > i) {
            const double dx = x_pos[j] - x_pos[i];
            const double dy = y_pos[j] - y_pos[i];
            const double d = std::sqrt(dx * dx + dy * dy);
            matrix[i * matrix_edge_size + j] = dx / (d * d * d);
            matrix[j * matrix_edge_size + i] = -matrix[i * matrix_edge_size + j];
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
void printDeviceProperties(int deviceId);
void printMatrix(const double* matrix, int rows, int cols);
void fill_array(double* arr, unsigned int size);
bool check_matrix(const double* matrix, const double* array, unsigned int size);
double check_matrix_force(const double* matrix, const double* x_pos, const double* y_pos, const unsigned int size);

void cpf_test(const double* x_pos, const double* y_pos, double* matrix, const unsigned int matrix_edge_size) {
    for (int i = 0; i < matrix_edge_size; ++i) {
        for (int j = 0; j < matrix_edge_size; ++j) {
            if (j == i) continue;
            const double dx = x_pos[j] - x_pos[i];
            const double dy = y_pos[j] - y_pos[i];
            const double d = std::sqrt(dx * dx + dy * dy);
            matrix[i * matrix_edge_size + j] = dx / (d * d * d);
        }
    }
}

int main() {
    //printDeviceProperties(0);

    std::cout << N_PARTICLES << " particles." << std::endl;

    // calculate the number of blocks needed
    constexpr int blocks_per_row = N_PARTICLES % BLOCK_SIZE == 0 ?
                                       N_PARTICLES / BLOCK_SIZE : N_PARTICLES / BLOCK_SIZE + 1;
    constexpr int n_blocks = blocks_per_row * (blocks_per_row + 1) / 2;

    // allocate matrix and array
    auto *matrix = static_cast<double *>(malloc(N_PARTICLES * N_PARTICLES * sizeof(double))); // this matrix is not actually needed on the host
    auto *matrix_reference = static_cast<double *>(malloc(N_PARTICLES * N_PARTICLES * sizeof(double)));
    auto *x_pos = static_cast<double *>(malloc(N_PARTICLES * sizeof(double)));
    auto *y_pos = static_cast<double *>(malloc(N_PARTICLES * sizeof(double)));

    // allocate matrix and array on device
    double *device_matrix, *device_x_pos, *device_y_pos;
    cudaMalloc(&device_matrix, N_PARTICLES*N_PARTICLES*sizeof(double)); checkCudaError("cudaMalloc");
    cudaMalloc(&device_x_pos, N_PARTICLES*sizeof(double)); checkCudaError("cudaMalloc");
    cudaMalloc(&device_y_pos, N_PARTICLES*sizeof(double)); checkCudaError("cudaMalloc");

    // fill array on host and copy to device
    x_pos[0] = 1.0;x_pos[1] = 3.0;x_pos[2] = 1.0;x_pos[3] = 3.0;
    y_pos[0] = 1.0;y_pos[1] = 1.0;y_pos[2] = 3.0;y_pos[3] = 3.0;
    //fill_array(x_pos, N_PARTICLES);
    //fill_array(y_pos, N_PARTICLES);
    cudaMemcpy(device_x_pos, x_pos, N_PARTICLES * sizeof(double), cudaMemcpyHostToDevice); checkCudaError("cudaMalloc");
    cudaMemcpy(device_y_pos, y_pos, N_PARTICLES * sizeof(double), cudaMemcpyHostToDevice); checkCudaError("cudaMalloc");

    // launch kernel
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim(n_blocks);
    calculate_pairwise_force<<<grid_dim, block_dim>>>(device_x_pos, device_y_pos, device_matrix, N_PARTICLES, blocks_per_row);
    checkCudaError("kernel launch");
    cudaDeviceSynchronize();

    double *device_f_tot_x;
    cudaMalloc(&device_f_tot_x, N_PARTICLES * sizeof(double)); checkCudaError("cudaMalloc");
    sum_over_rows<<<n_blocks, BLOCK_SIZE>>>(device_matrix, device_f_tot_x, N_PARTICLES);

    auto *f_tot = static_cast<double *>(malloc(N_PARTICLES * sizeof(double)));
    cudaMemcpy(f_tot, device_f_tot_x, N_PARTICLES * sizeof(double), cudaMemcpyDeviceToHost);

    // copy back data
    cudaMemcpy(matrix, device_matrix, N_PARTICLES * N_PARTICLES * sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "x_pos: " << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < N_PARTICLES; i++)
        std::cout << x_pos[i] << " ";
    std::cout << std::endl;
    std::cout << "y_pos: " << std::endl;
    for (int i = 0; i < N_PARTICLES; i++)
        std::cout << y_pos[i] << " ";
    std::cout << std::endl << std::endl;

    std::cout << "matrix: " << std::endl;
    printMatrix(matrix, N_PARTICLES, N_PARTICLES);
    std::cout << std::endl;

    std::cout << "matrix_reference: " << std::endl;
    cpf_test(x_pos, y_pos, matrix_reference, N_PARTICLES);
    printMatrix(matrix_reference, N_PARTICLES, N_PARTICLES);
    std::cout << std::endl;

    std::cout << "f_tot_x: " << std::endl;
    for (int i = 0; i < N_PARTICLES; i++)
        std::cout << f_tot[i] << " ";
    std::cout << std::endl << std::endl;

    std::cout << "Kernel is done! Checking result:" << std::endl;

    // check correctness of result
    double error = check_matrix_force(matrix, x_pos, y_pos, N_PARTICLES);
    std::cout << "Avg result error is: " << error << std::endl;

    // free space on Host and device
    free(matrix);               free(x_pos); free(y_pos);
    cudaFree(device_matrix);    cudaFree(device_x_pos);

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

bool check_matrix(const double* matrix, const double* array, const unsigned int size) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            if (array[j] - array[i] != matrix[i * size + j])
                return false;

    return true;
}

double check_matrix_force(const double* matrix, const double* x_pos, const double* y_pos, const unsigned int size) {
    double delta = 0.0f;
    double epsilon = 0.0001f;
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            if (i != j) {
                const double dx = x_pos[j] - x_pos[i];
                const double dy = y_pos[j] - y_pos[i];
                const double d = std::sqrt(dx * dx + dy * dy);
                const double tmp = dx / (d * d * d);
                if (std::abs(matrix[i * size + j] - tmp) > epsilon)
                        std::cerr << "[" << i << "," << j << "]" << matrix[i * size + j] - tmp << std::endl;
                delta += matrix[i * size + j] - tmp;
            }


    return delta;
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

void printDeviceProperties(int deviceId) {
    cudaDeviceProp deviceProp;
    cudaError_t err = cudaGetDeviceProperties(&deviceProp, deviceId);

    if (err != cudaSuccess) {
        std::cerr << "Error retrieving device properties: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    std::cout << "Device " << deviceId << ": " << deviceProp.name << std::endl;
    std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "  Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Shared memory per block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "  Registers per block: " << deviceProp.regsPerBlock << std::endl;
    std::cout << "  Warp size: " << deviceProp.warpSize << std::endl;
    std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "  Max threads dimensions: ("
              << deviceProp.maxThreadsDim[0] << ", "
              << deviceProp.maxThreadsDim[1] << ", "
              << deviceProp.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "  Max grid dimensions: ("
              << deviceProp.maxGridSize[0] << ", "
              << deviceProp.maxGridSize[1] << ", "
              << deviceProp.maxGridSize[2] << ")" << std::endl;
    std::cout << "  Clock rate: " << deviceProp.clockRate / 1000 << " MHz" << std::endl;
    std::cout << "  Memory clock rate: " << deviceProp.memoryClockRate / 1000 << " MHz" << std::endl;
    std::cout << "  Memory bus width: " << deviceProp.memoryBusWidth << " bits" << std::endl;
    std::cout << "  L2 cache size: " << deviceProp.l2CacheSize / 1024 << " KB" << std::endl;
    std::cout << "  Multiprocessor count: " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "  Concurrent kernels: " << (deviceProp.concurrentKernels ? "Yes" : "No") << std::endl;
    std::cout << "  ECC support: " << (deviceProp.ECCEnabled ? "Yes" : "No") << std::endl;
    std::cout << std::endl;
}

void printMatrix(const double* matrix, const int rows, const int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Access the element at (i, j) in row-major order
            std::cout << matrix[i * cols + j] << "\t";
            //if ((j + 1) % BLOCK_SIZE == 0) std::cout << "  ";
        }
        std::cout << std::endl; // Move to the next row
        //if ((i + 1) % BLOCK_SIZE == 0) std::cout << std::endl;
    }
}