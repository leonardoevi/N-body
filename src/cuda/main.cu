#include <iostream>
#include <string>
#include <cuda_runtime.h>

#define N_PARTICLES 45000l  // current max is circa 45.000
#define BLOCK_SIZE 32       // blocks will be 2D: BLOCK_SIZE*BLOCK_SIZE

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
__global__ void calculate_pairwise_difference(const float* arr, float* matrix, const unsigned int matrix_edge_size, const unsigned int blocks_per_row) {
    unsigned int i, j; mapIndexTo2D(blockIdx.x, blocks_per_row, i, j);
    i = i * blockDim.x + threadIdx.x;
    j = j * blockDim.y + threadIdx.y;

    if (i < matrix_edge_size && j < matrix_edge_size && i < j)
        matrix[i * matrix_edge_size + j] = arr[i] - arr[j];
}

void checkCudaError(const char* message);
void printDeviceProperties(int deviceId);
void printMatrix(const float* matrix, int rows, int cols);
void fill_array(float* arr, unsigned int size);
bool check_matrix(const float* matrix, const float* array, unsigned int size);

int main() {
    printDeviceProperties(0);

    std::cout << N_PARTICLES << std::endl;

    // calculate the number of blocks needed
    constexpr int blocks_per_row = N_PARTICLES % BLOCK_SIZE == 0 ?
                                       N_PARTICLES / BLOCK_SIZE : N_PARTICLES / BLOCK_SIZE + 1;
    constexpr int n_blocks = blocks_per_row * (blocks_per_row + 1) / 2;

    // allocate matrix and array
    auto *matrix = static_cast<float *>(malloc(N_PARTICLES * N_PARTICLES * sizeof(float)));
    auto *array = static_cast<float *>(malloc(N_PARTICLES * sizeof(float)));

    // allocate matrix and array on device
    float *device_matrix, *device_array;
    cudaMalloc(&device_matrix, N_PARTICLES*N_PARTICLES*sizeof(float)); checkCudaError("cudaMalloc");
    cudaMalloc(&device_array, N_PARTICLES*sizeof(float)); checkCudaError("cudaMalloc");

    // fill array on host and copy to device
    fill_array(array, N_PARTICLES);
    cudaMemcpy(device_array, array, N_PARTICLES * sizeof(float), cudaMemcpyHostToDevice); checkCudaError("cudaMalloc");

    // launch kernel
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim(n_blocks);
    calculate_pairwise_difference<<<grid_dim, block_dim>>>(device_array, device_matrix, N_PARTICLES, blocks_per_row);
    checkCudaError("kernel launch");
    cudaDeviceSynchronize();

    std::cout << "Kernel is done! Checking result." << std::endl;

    // copy back data
    cudaMemcpy(matrix, device_matrix, N_PARTICLES * N_PARTICLES * sizeof(float), cudaMemcpyDeviceToHost);

    // check correctness of result
    const bool correct = check_matrix(matrix, array, N_PARTICLES);
    const std::string s = correct ? " " : " not ";
    std::cout << "Result is" << s << "correct" << std::endl;

    // free space on Host and device
    free(matrix);               free(array);
    cudaFree(device_matrix);    cudaFree(device_array);

    if (correct)
        return EXIT_SUCCESS;
    return EXIT_FAILURE;
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

bool check_matrix(const float* matrix, const float* array, const unsigned int size) {
    for (int i = 0; i < size; i++)
        for (int j = i; j < size; j++)
            if (array[i] - array[j] != matrix[i * size + j])
                return false;

    return true;
}

void fill_array(float *arr, const unsigned int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = rand() / (float) RAND_MAX;
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

void printMatrix(const float* matrix, const int rows, const int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Access the element at (i, j) in row-major order
            std::cout << matrix[i * cols + j] << "  ";
            if ((j + 1) % BLOCK_SIZE == 0) std::cout << "  ";
        }
        std::cout << std::endl; // Move to the next row
        if ((i + 1) % BLOCK_SIZE == 0) std::cout << std::endl;
    }
}