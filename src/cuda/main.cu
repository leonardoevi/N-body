#include <iostream>
#include <string>
#include <cuda_runtime.h>

#define N_PARTICLES 44720l  // current max is circa 44720
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

__global__ void vectorAdd(const float* A, const float* B, float* C, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void calculate_pairwise_difference(const float* arr, float* matrix, const unsigned int matrix_edge_size, const unsigned int blocks_per_row) {
    unsigned int i, j; mapIndexTo2D(blockIdx.x, blocks_per_row, i, j);
    i = i * blockDim.x + threadIdx.x;
    j = j * blockDim.y + threadIdx.y;

    if (i < matrix_edge_size && j < matrix_edge_size && i < j)
        matrix[i * matrix_edge_size + j] = arr[i] - arr[j];
}

void checkCudaError(const char* message);
void printDeviceProperties(int deviceId);
void test_kernel();
void printMatrix(const float* matrix, int rows, int cols);
void fill_array(float* arr, unsigned int size);
bool check_matrix(const float* matrix, const float* array, unsigned int size);

int main() {
/*
    // print device properties
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Error retrieving device count: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    std::cout << "Number of CUDA-capable devices: " << deviceCount << std::endl;
    for (int i = 0; i < deviceCount; ++i) {
        printDeviceProperties(i);
    }

    long long available_floats = 8105l * 1000l * 1000l / sizeof(float);
    std::cout << available_floats << std::endl;

    // test kernel execution
    test_kernel();
    */

    std::cout << N_PARTICLES << std::endl;

    // calculate the number of blocks needed
    constexpr int blocks_per_row = N_PARTICLES % BLOCK_SIZE == 0 ?
                                       N_PARTICLES / BLOCK_SIZE : N_PARTICLES / BLOCK_SIZE + 1;
    constexpr int n_blocks = blocks_per_row * (blocks_per_row + 1) / 2;

    // allocate matrix and array
    float *matrix = static_cast<float *>(malloc(N_PARTICLES * N_PARTICLES * sizeof(float)));
    float *array = static_cast<float *>(malloc(N_PARTICLES * sizeof(float)));

    float *device_matrix, *device_array;
    cudaMalloc(&device_matrix, N_PARTICLES*N_PARTICLES*sizeof(float)); checkCudaError("cudaMalloc");
    cudaMalloc(&device_array, N_PARTICLES*sizeof(float)); checkCudaError("cudaMalloc");

    // fill array on host and copy to device
    fill_array(array, N_PARTICLES);

/*
    std::cout << "Array:" << std::endl;
    for (int i = 0; i < N_PARTICLES; i++) {
        std::cout << array[i] << "  ";
        if ((i + 1) % BLOCK_SIZE == 0) std::cout << "  ";
    }
    std::cout << std::endl << std::endl;
    */

    cudaMemcpy(device_array, array, N_PARTICLES * sizeof(float), cudaMemcpyHostToDevice); checkCudaError("cudaMalloc");

    //dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    //dim3 grid_dim(blocks_per_row, blocks_per_row);
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim(n_blocks);

    calculate_pairwise_difference<<<grid_dim, block_dim>>>(device_array, device_matrix, N_PARTICLES, blocks_per_row);
    checkCudaError("kernel launch");

    cudaDeviceSynchronize();

    std::cout << "Kernel is done! Checking result." << std::endl;

    // copy back data
    cudaMemcpy(matrix, device_matrix, N_PARTICLES * N_PARTICLES * sizeof(float), cudaMemcpyDeviceToHost);

    //std::cout << "Matrix:" << std::endl;
    //printMatrix(matrix, N_PARTICLES, N_PARTICLES);

    bool correct = check_matrix(matrix, array, N_PARTICLES);

    const std::string s = correct ? " " : " not ";
    std::cout << "Result is" << s << "correct" << std::endl;

    free(matrix);
    free(array);

    cudaFree(device_matrix);
    cudaFree(device_array);

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

void test_kernel() {
    const int N = 1000; // Size of vectors
    const int size = N * sizeof(float);

    // Host vectors
    float *h_A, *h_B, *h_C;
    h_A = new float[N];
    h_B = new float[N];
    h_C = new float[N];

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // Device vectors
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify the result
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (abs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            success = false;
            std::cerr << "Error at index " << i << ": Expected "
                      << (h_A[i] + h_B[i]) << " but got " << h_C[i] << "\n";
            break;
        }
    }

    if (success) {
        std::cout << "CUDA kernel executed successfully and verified!\n";
    }

    // Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
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