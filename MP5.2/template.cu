// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
    do {                                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
            return -1;                                                          \
        }                                                                     \
    } while (0)

__global__ void scan(float *input, float *output, int InputSize) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from the host

    __shared__ float scanShared[BLOCK_SIZE * 2];

    unsigned int t = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + t * 2;

    scanShared[2 * t] = (idx < InputSize) ? input[idx] : 0;
    scanShared[2 * t + 1] = (idx + 1 < InputSize) ? input[idx + 1] : 0;

    // Reduction Step
    for (int stride = 1; stride < (BLOCK_SIZE * 2); stride *= 2) {
        __syncthreads();
        int index = (t + 1) * stride * 2 - 1;
        if (index < (BLOCK_SIZE * 2) && (index - stride) >= 0)
            scanShared[index] += scanShared[index - stride];
    }

    // Post Scan Step
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (t + 1) * stride * 2 - 1;
        if ((index + stride) < (BLOCK_SIZE * 2))
            scanShared[index + stride] += scanShared[index];
    }

    __syncthreads();

    if (idx < InputSize)
        input[idx] = scanShared[t * 2];
    if (idx + 1 < InputSize)
        input[idx + 1] = scanShared[t * 2 + 1];

    // if scan is only partially done
    if ((InputSize > (BLOCK_SIZE * 2)) && (t == 0))
        output[blockIdx.x] = scanShared[(BLOCK_SIZE * 2) - 1];
}

__global__ void scanFinalAdd(float* input, float* output, float* helpArray, int InputSize) {

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if ((blockIdx.x > 0) && (InputSize > (BLOCK_SIZE * 2)) && (idx < InputSize))
        output[idx] = input[idx] + helpArray[blockIdx.x - 1];
    else if (blockIdx.x == 0)
        output[idx] = input[idx];
}

int main(int argc, char **argv) {
    wbArg_t args;
    float* hostInput;  // The input 1D list
    float* hostOutput; // The output list
    float* deviceInput;
    float* deviceOutput;
    float* deviceHelper;
    int numElements; // number of elements in the list
    
    args = wbArg_read(argc, argv);
    
    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float *)malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");
    
    wbLog(TRACE, "The number of input elements in the input is ", numElements);
    
    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
    wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceHelper, ceil(numElements / (BLOCK_SIZE * 2.0)) * sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");
    
    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
    wbCheck(cudaMemset(deviceHelper, 0, ceil(numElements / (BLOCK_SIZE * 2.0)) * sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");
    
    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
    dim3 dimGrid(ceil(numElements / (BLOCK_SIZE * 2.0)));
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGridScanBlock(1);
    dim3 dimBlockAdd(BLOCK_SIZE * 2.0);
    
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
    wbTime_start(Compute, "Performing CUDA computation");
    scan <<<dimGrid, dimBlock >>> (deviceInput, deviceHelper, numElements);
    scan <<<dimGridScanBlock, dimBlock >>> (deviceHelper, NULL, ceil(numElements / (BLOCK_SIZE * 2.0)));
    scanFinalAdd <<<dimGrid, addDimBlock >>> (deviceInput, deviceOutput, deviceHelper, numElements);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");
    
    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");
    
    wbSolution(args, hostOutput, numElements);
    
    free(hostInput);
    free(hostOutput);
    
    return 0;
}
