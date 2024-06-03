#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "cuda_fp16.h"

#define MAX_CHANNEL 4
#define MAX_MAP_OUT 16
#define MAX_K 7

#define TILE_WIDTH 20
#define MASK_WIDTH 7

// Optimization 1: Weight matrix (kernel values) in constant memory
__constant__ float const_device_mask[MAX_MAP_OUT * MAX_CHANNEL * MAX_K * MAX_K];

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask,
                                    const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K, const int W_tile) {
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // (void) Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void) Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(b, m, h, w) output[(b) * (Map_out * Height_out * Width_out) + (m) * (Height_out * Width_out) + (h) * (Width_out) + w]
    #define in_4d(b, c, h, w) input[(b) * (Channel * Height * Width) + (c) * (Height * Width) + (h) * (Width) + w]
    #define const_mask_4d(m, c, p, q) const_device_mask[(m) * (Channel * K * K) + (c) * (K * K) + (p) * (K) + q]

    // Insert your GPU convolution kernel code here

    // Tiled shared memory convolution
    // __shared__ float shared_input[MAX_CHANNEL][TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1];
    // __shared__ float shared_output[MAX_CHANNEL][TILE_WIDTH * TILE_WIDTH];

    // FP16 on Tiled shared memory convolution
    // __shared__ __half shared_input[MAX_CHANNEL][TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1];
    // __shared__ __half shared_output[MAX_CHANNEL][TILE_WIDTH * TILE_WIDTH];

    int b = blockIdx.x;
    int m = blockIdx.y;

    // int c  = threadIdx.x;
    // int ty = threadIdx.y;
    // int tx = threadIdx.z;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int h = (blockIdx.z / W_tile) * TILE_WIDTH + ty;
    int w = (blockIdx.z % W_tile) * TILE_WIDTH + tx;
    
    // // Tiled shared memory convolution
    // for (int c = 0; c < MAX_CHANNEL; c++) {
    //     if (c < Channel && h < Height && w < Width) {
    //         // shared_input[c][ty][tx] = in_4d(b, c, h, w);
    //         shared_input[c][ty][tx] = __float2half(in_4d(b, c, h, w));
    //     } else {
    //         shared_input[c][ty][tx] = 0;
    //     }
    // }
    // __syncthreads();

    float acc = 0.0f;
    // __half acc = 0.0f;
    if (ty < TILE_WIDTH && tx < TILE_WIDTH && h < Height_out && w < Width_out) {
        for (int c = 0; c < Channel; c++) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {

                    // Constant memory
                    acc += in_4d(b, c, h + p, w + q) * const_mask_4d(m, c, p, q);
                    
                    // Tiled shared memory convolution
                    // acc += shared_input[c][ty + p][tx + q] * const_mask_4d(m, c, p, q);

                    // FP16
                    // __half halfMask = __float2half(const_mask_4d(m, c, p, q));

                    // acc = __hadd(acc, __hmul(in_4d(b, c, h + p, w + q), halfMask));
                    // acc = __hadd(acc, __hmul(shared_input[c][ty + p][tx + q], halfMask));
                }
            }
        }
        out_4d(b, m, h, w) = acc;
        // out_4d(b, m, h, w) = __half2float(acc);
    }

    /////////////////////////////
    //                         //
    // Reduction on K * K loop //
    // Did not succeed at last //
    //                         //
    /////////////////////////////
    // if (ty < TILE_WIDTH && tx < TILE_WIDTH) {
    //     shared_output[c][ty * (TILE_WIDTH) + tx] = 0;
    // }
    // __syncthreads();
    
    // int tIdx = ty * (TILE_WIDTH + MAX_K - 1) + tx;
    // int newTIdx = tIdx % (TILE_WIDTH * TILE_WIDTH);
    // int newTy = newTIdx / TILE_WIDTH;
    // int newTx = newTIdx % TILE_WIDTH;
    // int newTIdxSub = tIdx / (TILE_WIDTH * TILE_WIDTH);

    // int newH = (blockIdx.z / W_tile) * TILE_WIDTH + newTy;
    // int newW = (blockIdx.z % W_tile) * TILE_WIDTH + newTx;
    
    // // shared_input[c][newTy][newTx] = shared_input[c][newTy][newTx] * const_mask_4d(m, c, 0, 0);
    // for (int pq = 0; pq < (K * K) / 1; pq++) {
    //     int newPQ = (pq * 1) + newTIdxSub;
    //     if (c < Channel && tIdx < (TILE_WIDTH * TILE_WIDTH) * 1 && newH < Height_out && newW < Width_out && newPQ < 49) {
    //         int yy = newPQ / K;
    //         int xx = newPQ % K;
            
    //         // atomicAdd(&out_4d(b, m, newH, newW), shared_input[c][newTy + yy][newTx + xx] * const_mask_4d(m, c, yy, xx));
    //         shared_output[c][tIdx] += shared_input[c][newTy + yy][newTx + xx] * const_mask_4d(m, c, yy, xx);
    //         // shared_input[c][newTy][newTy] += shared_input[c][newTy + yy][newTx + xx] * const_mask_4d(m, c, yy, xx);
    //     }
    // }
    // if (c < Channel && tIdx < (TILE_WIDTH * TILE_WIDTH) * 1 && newH < Height_out && newW < Width_out) {
    //     out_4d(b, m, newH, newW) = shared_output[c][tIdx];
    // }
    // if (c > 1) {
    //     if (tIdx < 64) {
    //         atomicAdd(&out_4d(b, m, newH, newW), shared_input[c][newTy][newTx]);
    //     }
    //     // atomicAdd(&shared_output[c][newTIdx], shared_output[c][newTIdx + newTIdxSub]);
    // }
    // __syncthreads();
    ///////////////////////////////////////////////////////////////////////////////////////////

    // Optimization 3, 4: Input Channel reduction
    // for (int p = 0; p < K; p++) {
    //     for (int q = 0; q < K; q++) {
    //         if (c < Channel && ty < TILE_WIDTH && tx < TILE_WIDTH && h < Height_out && w < Width_out) {
    //             // atomicAdd(&out_4d(b, m, h, w), shared_input[c][ty + p][tx + q] * const_mask_4d(m, c, p, q));
    //             shared_output[c][ty * (TILE_WIDTH) + tx] += shared_input[c][ty + p][tx + q] * const_mask_4d(m, c, p, q);
    //         }
    //     }
    // }
    // __syncthreads();
    // 
    // if (c < Channel && ty < TILE_WIDTH && tx < TILE_WIDTH && h < Height_out && w < Width_out) {
    //     if (Channel == 1) {
    //         out_4d(b, m, h, w) = shared_output[c][ty * (TILE_WIDTH) + tx];
    //     } else {
    // 
    //         // // Tree reduction
    //         // for (unsigned int stride = (MAX_CHANNEL / 2); stride >= 1; stride /= 2) {
    //         //     if (c < stride) {
    //         //         shared_output[c][ty * (TILE_WIDTH) + tx] += shared_output[c + stride][ty * (TILE_WIDTH) + tx];
    //         //     }
    //         //     __syncthreads();
    //         // }
    //         // out_4d(b, m, h, w) = shared_output[0][ty * (TILE_WIDTH) + tx];
    //       
    //
    //         // Atomic reduction
    //         if (c > 0) {
    //             atomicAdd(&shared_output[0][ty * (TILE_WIDTH) + tx], shared_output[c][ty * (TILE_WIDTH) + tx]);
    //         }
    //         __syncthreads();
    //         out_4d(b, m, h, w) = shared_output[0][ty * (TILE_WIDTH) + tx];
    //     }
    // }

    #undef out_4d
    #undef in_4d
    #undef const_mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask,
                                                    float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr,
                                                    const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

    const int H_out = Height - K + 1;
    const int W_out = Width - K + 1;

    const int input_size = Batch * Channel * Height * Width;
    const int output_size = Batch * Map_out * H_out * W_out;
    const int mask_size = Channel * Map_out * K * K;

    std::cout << "Batch: " << Batch << std::endl;
    std::cout << "Map_out: " << Map_out << std::endl;
    std::cout << "Channel: " << Channel << std::endl;
    std::cout << "Height: " << Height << std::endl;
    std::cout << "Width: " << Width << std::endl;
    std::cout << "K: " << K << std::endl;

    cudaMalloc((void**)device_input_ptr, input_size * sizeof(float));
    cudaMalloc((void**)device_output_ptr, output_size * sizeof(float));
    cudaMalloc((void**)device_mask_ptr, mask_size * sizeof(float));

    cudaMemset(*device_output_ptr, 0, output_size * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(*device_mask_ptr, host_mask, mask_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(const_device_mask, host_mask, mask_size * sizeof(float), 0, cudaMemcpyHostToDevice);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask,
                                             const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {
    // Set the kernel dimensions and call the kernel

    int H_out = Height - K + 1;
    int W_out = Width - K + 1;

    const int H_tile = ceil(H_out * 1.0 / TILE_WIDTH); // number of vertical tiles per output map
    const int W_tile = ceil(W_out * 1.0 / TILE_WIDTH); // number of horizontal tiles per output map

    dim3 dimGrid(Batch, Map_out, H_tile * W_tile);

    // Untiled
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    
    // Tiled shared memory convolution
    // dim3 dimBlock(TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1);

    // Reduce
    // dim3 dimBlock(MAX_CHANNEL, TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1);

    conv_forward_kernel<<<dimGrid, dimBlock>>> (device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K, W_tile);

    cudaDeviceSynchronize();
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask,
                                                    const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {

    const int H_out = Height - K + 1;
    const int W_out = Width - K + 1;

    const int output_size = Batch * Map_out * H_out * W_out;

    // Copy the output back to host
    cudaMemcpy(host_output, device_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
