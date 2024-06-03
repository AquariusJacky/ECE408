
#include <wb.h>

#define BLOCK_WIDTH 8.0

#define wbCheck(stmt)                                                     \
do {                                                                      \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
        wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
        wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
        return -1;                                                        \
    }                                                                     \
} while (0)

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C,
                               int numCRows, int numCColumns, int middleWidth) {

    //@@ Insert code to implement matrix multiplication here
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < numCRows && j < numCColumns) {
        float sum = 0;
        for (int k = 0; k < middleWidth; k++)
            sum += A[i * middleWidth + k] * B[k * numCColumns + j];
        C[i * numCColumns + j] = sum;
    }
}

int main(int argc, char **argv) {
    wbArg_t args;
    float *hostA; // The A matrix
    float *hostB; // The B matrix
    float *hostC; // The output C matrix
    float *deviceA;
    float *deviceB;
    float *deviceC;
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows;    // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)
    
    args = wbArg_read(argc, argv);
    
    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                              &numAColumns);
    hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                              &numBColumns);
    
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    
    size_t sizeA = numARows * numAColumns * sizeof(float);
    size_t sizeB = numBRows * numBColumns * sizeof(float);
    size_t sizeC = numCRows * numCColumns * sizeof(float);
    
    //@@ Allocate the hostC matrix
    hostC = (float*)malloc(sizeC);
    wbTime_stop(Generic, "Importing data and creating memory on host");
    
    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
    
    //@@ Allocate GPU memory here
    wbTime_start(GPU, "Allocating GPU memory.");
    cudaMalloc((void**)&deviceA, sizeA);
    cudaMalloc((void**)&deviceB, sizeB);
    cudaMalloc((void**)&deviceC, sizeC);
    wbTime_stop(GPU, "Allocating GPU memory.");
    
    //@@ Copy memory to the GPU here
    wbTime_start(GPU, "Copying input memory to the GPU.");
    cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);
    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
    dim3 dimGrid(ceil(numCRows / BLOCK_WIDTH * 1.0), ceil(numCColumns / BLOCK_WIDTH * 1.0));
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
    
    //@@ Launch the GPU Kernel here
    wbTime_start(Compute, "Performing CUDA computation");

    // numAColumns = numBRows = missing middle width.
    matrixMultiply <<<dimGrid, dimBlock>>> (deviceA, deviceB, deviceC, numCRows, numCColumns, numAColumns);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    //@@ Copy the GPU memory back to the CPU here
    wbTime_start(Copy, "Copying output memory to the CPU");
    cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying output memory to the CPU");
    
    //@@ Free the GPU memory here
    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    wbTime_stop(GPU, "Freeing GPU Memory");
    
    wbSolution(args, hostC, numCRows, numCColumns);
    
    free(hostA);
    free(hostB);
    free(hostC);
    
    return 0;
}
