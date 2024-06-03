// Histogram Equalization

#include <wb.h>
#include <iostream>

using namespace std;

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 512

//@@ insert code here

__global__ void kernel_getGrayImage(float* input, unsigned char* outputChar, unsigned char* outputGray, int Width, int Height) {
    
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = y * Width + x;

    if (x < Width && y < Height) {
        
        outputChar[idx * 3] = (unsigned char)(input[idx * 3] * 255.0);
        outputChar[idx * 3 + 1] = (unsigned char)(input[idx * 3 + 1] * 255.0);
        outputChar[idx * 3 + 2] = (unsigned char)(input[idx * 3 + 2] * 255.0);
        
        unsigned char r = outputChar[idx * 3];
        unsigned char g = outputChar[idx * 3 + 1];
        unsigned char b = outputChar[idx * 3 + 2];

        outputGray[idx] = (unsigned char)(0.21 * r + 0.71 * g + 0.07 * b);
    }
}

__global__ void kernel_getHistogram(unsigned char* input, unsigned int* output, int Size) {

    __shared__ unsigned int privHistogram[256];

    if (threadIdx.x < 256) {
        privHistogram[threadIdx.x] = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    while (idx < Size) {
        // Atomic version of output[input[idx]]++
        atomicAdd(&(privHistogram[(unsigned int)input[idx]]), 1);
        idx += stride;
    }
    __syncthreads();

    if (threadIdx.x < 256) {
        atomicAdd(&(output[threadIdx.x]), privHistogram[threadIdx.x]);
    }

}

__global__ void kernel_scan(unsigned int* input, float* cdf, unsigned int Size) {

    __shared__ float scanShared[256];

    unsigned int t = threadIdx.x;
    unsigned int idx = t * 2;

    scanShared[idx] = input[idx];
    scanShared[idx + 1] = input[idx + 1];

    // Reduction Step
    for (int stride = 1; stride < 256; stride *= 2) {
        __syncthreads();
        int index = (t + 1) * stride * 2 - 1;
        if (index < 256 && (index - stride) >= 0)
            scanShared[index] += scanShared[index - stride];
    }

    // Post Scan Step
    for (int stride = 64; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (t + 1) * stride * 2 - 1;
        if ((index + stride) < 256)
            scanShared[index + stride] += scanShared[index];
    }

    __syncthreads();

    cdf[idx] = scanShared[idx] / (Size * 1.0);
    cdf[idx + 1] = scanShared[idx + 1] / (Size * 1.0);
}

__device__ unsigned char kernel_clamp(float x, float start, float end) {
    return (unsigned char)min(max(x, start), end);
}

__global__ void kernel_histogramEqualization(unsigned char* input, float* output, float* cdf, float cdfMin, int Width, int Height) {
    
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = y * Width + x;

    if (x < Width && y < Height) {
        
        output[idx * 3] = (float)(kernel_clamp(255 * (cdf[input[idx * 3]] - cdfMin) / (1.0f - cdfMin), 0, 255.0f) / 225.0);
        output[idx * 3 + 1] = (float)(kernel_clamp(255 * (cdf[input[idx * 3 + 1]] - cdfMin) / (1.0f - cdfMin), 0, 255.0f) / 225.0);
        output[idx * 3 + 2] = (float)(kernel_clamp(255 * (cdf[input[idx * 3 + 2]] - cdfMin) / (1.0f - cdfMin), 0, 255.0f) / 225.0);
    }
}

int main(int argc, char **argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float* host_inputImageData;
    float* host_outputImageData;
    const char* inputImageFile;
    
    //@@ Insert more code here
    
    args = wbArg_read(argc, argv); /* parse the input arguments */
    
    inputImageFile = wbArg_getInputFile(args, 0);
    
    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    host_inputImageData = wbImage_getData(inputImage);
    host_outputImageData = wbImage_getData(outputImage);
    wbTime_stop(Generic, "Importing data and creating memory on host");
    
    //@@ insert code here
    int imageSize = imageWidth * imageHeight;

    //////////////////////////////////////////////////////
    // Step 1:                                          //
    // Cast the image from `float` to "unsigned char"   //
    //                                                  //
    // Step 2:                                          //
    // Convert the image from RGB to GrayScale          //
    //////////////////////////////////////////////////////
    float* device_inputImageData;
    unsigned char* device_inputImageChar;
    unsigned char* device_inputImageGray;
    unsigned char* host_inputImageChar = new unsigned char[imageSize * imageChannels]();
    unsigned char* host_inputImageGray = new unsigned char[imageSize]();

    cudaMalloc((void**)&device_inputImageData, imageSize * 3 * sizeof(float));
    cudaMalloc((void**)&device_inputImageChar, imageSize * 3 * sizeof(unsigned char));
    cudaMalloc((void**)&device_inputImageGray, imageSize * sizeof(unsigned char));

    cudaMemset(device_inputImageData, 0, imageSize * 3 * sizeof(float));
    cudaMemset(device_inputImageChar, 0, imageSize * 3 * sizeof(unsigned char));
    cudaMemset(device_inputImageGray, 0, imageSize * sizeof(unsigned char));

    cudaMemcpy(device_inputImageData, host_inputImageData, imageSize * 3 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGridGray(ceil(imageWidth / 32.0), ceil(imageHeight / 32.0));
    dim3 dimBlockGray(32, 32);
    
    kernel_getGrayImage <<<dimGridGray, dimBlockGray>>> (device_inputImageData, device_inputImageChar, device_inputImageGray, imageWidth, imageHeight);
    cudaDeviceSynchronize();

    cudaMemcpy(host_inputImageChar, device_inputImageChar, imageSize * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_inputImageGray, device_inputImageGray, imageSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    //////////////////////////////////////////
    // Step 3:                              //
    // Compute the histogram of "grayImage" //
    //////////////////////////////////////////
    unsigned char* device_input;
    unsigned int* device_histogram;
    unsigned int* host_histogram = new unsigned int[256]();

    cudaMalloc((void**)&device_input, imageSize * sizeof(unsigned char));
    cudaMalloc((void**)&device_histogram, 256 * sizeof(unsigned int));

    cudaMemset(device_input, 0, imageSize * sizeof(unsigned char));
    cudaMemset(device_histogram, 0, 256 * sizeof(unsigned int));
    
    cudaMemcpy(device_input, host_inputImageGray, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    dim3 dimGridHis(ceil(imageSize / (BLOCK_SIZE * 8.0f)));
    dim3 dimBlockHis(BLOCK_SIZE);
    
    kernel_getHistogram <<<dimGridHis, dimBlockHis>>> (device_input, device_histogram, imageSize);
    cudaDeviceSynchronize();

    cudaMemcpy(host_histogram, device_histogram, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    
    //////////////////////////////////////////////////////////////////
    // Step 4:                                                      //
    // Compute the Cumulative Distribution Function of "histogram"  //
    //////////////////////////////////////////////////////////////////
    float* device_cdf;
    float* host_cdf = new float[256]; // Cumulative Distribution Function
    float cdfMin;
    
    cudaMalloc((void**)&device_cdf, 256 * sizeof(float));

    cudaMemset(device_cdf, 0, 256 * sizeof(unsigned char));

    dim3 dimGridScan(1);
    dim3 dimBlockScan(128);

    kernel_scan <<<dimGridScan, dimBlockScan>>> (device_histogram, device_cdf, imageSize);

    cudaMemcpy(host_cdf, device_cdf, 256 * sizeof(float), cudaMemcpyDeviceToHost);
    
    cdfMin = host_cdf[0];

    //////////////////////////////////////////////////
    // Step 5:                                      //
    // Define the histogram equalization function   //
    //                                              //
    // Step 6:                                      //
    // Apply the histogram equalization function    //
    //////////////////////////////////////////////////
    float* device_outputImageData;

    cudaMalloc((void**)&device_outputImageData, imageSize * 3 * sizeof(float));

    cudaMemset(device_outputImageData, 0, imageSize * 3 * sizeof(float));

    dim3 dimGridHE(ceil(imageWidth / 32.0), ceil(imageHeight / 32.0));
    dim3 dimBlockHE(32, 32);
    
    kernel_histogramEqualization <<<dimGridHE, dimBlockHE>>> (device_inputImageChar, device_outputImageData, device_cdf, cdfMin, imageWidth, imageHeight);
    cudaDeviceSynchronize();

    cudaMemcpy(host_outputImageData, device_outputImageData, imageSize * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    wbSolution(args, outputImage);
    
    //@@ insert code here

    cudaFree(device_inputImageChar);
    cudaFree(device_inputImageGray);
    cudaFree(device_input);
    cudaFree(device_histogram);
    cudaFree(device_cdf);

    free(host_inputImageData);
    free(host_outputImageData);
    free(host_inputImageChar);
    free(host_inputImageGray);
    free(host_histogram);
    free(host_cdf);

    return 0;
}
