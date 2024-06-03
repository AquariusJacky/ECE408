
#include <wb.h>

#define TILE_WIDTH 32

#define wbCheck(stmt)                                                           \
    do {                                                                        \
        cudaError_t err = stmt;                                                 \
        if (err != cudaSuccess) {                                               \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
            wbLog(ERROR, "Got CUDA error ... ", cudaGetErrorString(err));       \
            return -1;                                                          \
        }                                                                       \
    } while (0)

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C,
                               int numCRows, int numCColumns,
                               int middleWidth) {

    //@@ Insert code to implement matrix multiplication here
    __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = blockIdx.x * blockDim.x + tx;
    int Col = blockIdx.y * blockDim.y + ty;

    float sum = 0;
    
    for (int tileIdx = 0; tileIdx < ceil((float)middleWidth / TILE_WIDTH); tileIdx++) {

        // Fill in 0 if out of A's range
        if (Row < numCRows && (tileIdx * TILE_WIDTH + ty) < middleWidth) {
            subTileA[tx][ty] = A[Row * middleWidth + (tileIdx * TILE_WIDTH + ty)];
        } else subTileA[tx][ty] = 0;

        // Fill in 0 if out of B's range
        if ((tileIdx * TILE_WIDTH + tx) < middleWidth && Col < numCColumns) {
            subTileB[tx][ty] = B[(tileIdx * TILE_WIDTH + tx) * numCColumns + Col];
        } else subTileB[tx][ty] = 0;

        __syncthreads();

        // Even calculate when out of range
        // Avoid control branch divergence
        for (int k = 0; k < TILE_WIDTH; k++)
            sum += subTileA[tx][k] * subTileB[k][ty];

        __syncthreads();
    }

    if (Row < numCRows && Col < numCColumns)
        C[Row * numCColumns + Col] = sum;
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
    int numCColumns; // number of columns in the matrix C (you have to set
                     // this)
    
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
    dim3 dimGrid(ceil(numCRows / (TILE_WIDTH * 0.1)), ceil(numCColumns / (TILE_WIDTH * 0.1)));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    
    //@@ Launch the GPU Kernel here
    wbTime_start(Compute, "Performing CUDA computation");
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
