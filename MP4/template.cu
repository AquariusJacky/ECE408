#include <wb.h>

#define wbCheck(stmt)                                                     \
do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
        wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
        wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
        return -1;                                                          \
    }                                                                     \
} while (0)

//@@ Define any useful program-wide constants here
#define TILE_WIDTH 8
#define MASK_WIDTH 3

//@@ Define constant memory for device kernel here
__constant__ float deviceMask[MASK_WIDTH * MASK_WIDTH * MASK_WIDTH];

__global__ void conv3d(float *input, float *output,
                       const int z_size, const int y_size, const int x_size) {
    
    //@@ Insert kernel code here
    __shared__ float inputTile[TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1];

    int radius = MASK_WIDTH / 2;

    int tz = threadIdx.z;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int z_o = blockIdx.z * TILE_WIDTH + tz;
    int y_o = blockIdx.y * TILE_WIDTH + ty;
    int x_o = blockIdx.x * TILE_WIDTH + tx;

    int z_i = z_o - radius;
    int y_i = y_o - radius;
    int x_i = x_o - radius;

    if (z_i >= 0 && z_i < z_size &&
        y_i >= 0 && y_i < y_size &&
        x_i >= 0 && x_i < x_size)
        inputTile[tz][ty][tx] = input[z_i * (y_size * x_size) + y_i * (x_size) + x_i];
    else inputTile[tz][ty][tx] = 0;

    __syncthreads();

    float Pvalue = 0;
    if (tz < TILE_WIDTH && ty < TILE_WIDTH && tx < TILE_WIDTH) {
        for (int i = 0; i < MASK_WIDTH; i++) {
            for (int j = 0; j < MASK_WIDTH; j++) {
                for (int k = 0; k < MASK_WIDTH; k++) {
                    Pvalue += deviceMask[i * MASK_WIDTH * MASK_WIDTH + j * MASK_WIDTH + k] * inputTile[tz + i][ty + j][tx + k];
                }
            }
        }

        if (z_o < z_size && y_o < y_size && x_o < x_size) {
            output[z_o * y_size * x_size + y_o * x_size + x_o] = Pvalue;
        }
    }

    __syncthreads();
}

int main(int argc, char* argv[]) {
    wbArg_t args;
    int z_size;
    int y_size;
    int x_size;
    int inputLength, kernelLength;
    float* hostInput;
    float* hostKernel;
    float* hostOutput;
    float* deviceInput;
    float* deviceOutput;

    args = wbArg_read(argc, argv);

    // Import data
    hostInput = (float*)wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostKernel =
        (float*)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
    hostOutput = (float*)malloc(inputLength * sizeof(float));

    // First three elements are the input dimensions
    z_size = hostInput[0];
    y_size = hostInput[1];
    x_size = hostInput[2];

    int input_size = z_size * y_size * x_size * sizeof(float);
    int output_size = input_size;
    int kernel_size = kernelLength * sizeof(float);

    wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
    assert(z_size * y_size * x_size == inputLength - 3);
    assert(kernelLength == 27);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    //@@ Allocate GPU memory here
    // Recall that inputLength is 3 elements longer than the input data
    // because the first  three elements were the dimensions
    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void**)&deviceInput, input_size);
    cudaMalloc((void**)&deviceOutput, output_size);
    wbTime_stop(GPU, "Doing GPU memory allocation");

    //@@ Copy input and kernel to GPU here
    // Recall that the first three elements of hostInput are dimensions and
    // do not need to be copied to the gpu
    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInput, hostInput + 3, input_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceMask, hostKernel, kernel_size, 0, cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");

    //@@ Initialize grid and block dimensions here
    dim3 dimGrid(ceil(x_size / (TILE_WIDTH * 1.0)), ceil(y_size / (TILE_WIDTH * 1.0)), ceil(z_size / (TILE_WIDTH * 1.0)));
    dim3 dimBlock(TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1);

    //@@ Launch the GPU kernel here
    wbTime_start(Compute, "Doing the computation on the GPU");
    conv3d <<<dimGrid, dimBlock>>> (deviceInput, deviceOutput, z_size, y_size, x_size);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Doing the computation on the GPU");

    //@@ Copy the device memory back to the host here
    // Recall that the first three elements of the output are the dimensions
    // and should not be set here (they are set below)
    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutput + 3, deviceOutput, output_size, cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    // Set the output dimensions for correctness checking
    hostOutput[0] = z_size;
    hostOutput[1] = y_size;
    hostOutput[2] = x_size;
    wbSolution(args, hostOutput, inputLength);
    
    // Free device memory
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    
    // Free host memory
    free(hostInput);
    free(hostOutput);
    return 0;
}
