#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cub/cub.cuh>

#include "utils.h"
#include "zOrder2d.h"


#define BLOCK_SIZE 1024

#define BIT_PASS 4 // number of bit each time

#define BUCK_SIZE 256
#define BIT_MASK 0xF
#define CUDA_MAX_BLOCK 65535

#define CUDA_RT_CALL( call )                                                                       \
{                                                                                                  \
    cudaError_t cudaStatus = call;                                                                 \
    if ( cudaSuccess != cudaStatus )                                                               \
        fprintf(stderr, "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s (%d).\n", \
                        #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus);    \
}

__host__ __device__ inline ulli calcuateZorder2d(ulli input, ulli input2, int xGroupInterval, int yGroupInterval)  {

    int Xgroup = input/xGroupInterval;
    int Ygroup = input2/yGroupInterval;

    static const ulli MASKS[] = { 0x00000000FFFFFFFF, 0x0000FFFF0000FFFF, 0x00FF00FF00FF00FF, 0x0F0F0F0F0F0F0F0F, 0x3333333333333333, 0x5555555555555555 };

    ulli x = Xgroup;  // Interleave lower 16 bits of x and y, so the bits of x
    ulli y = Ygroup;  // are in the even positions and bits from y in the odd;

    x = (x | (x << 32)) & MASKS[0];
    x = (x | (x << 16)) & MASKS[1];
    x = (x | (x << 8)) & MASKS[2];
    x = (x | (x << 4)) & MASKS[3];
    x = (x | (x << 2)) & MASKS[4];
    x = (x | (x << 1)) & MASKS[5];

    y = (y | (y << 32)) & MASKS[0];
    y = (y | (y << 16)) & MASKS[1];
    y = (y | (y << 8)) & MASKS[2];
    y = (y | (y << 4)) & MASKS[3];
    y = (y | (y << 2)) & MASKS[4];
    y = (y | (y << 1)) & MASKS[5];

    ulli result = x | (y << 1);

    return result;
}

__global__
void calcuateZorderKernel2d(ulli *dev_input, ulli *dev_input2, ulli *dev_output, ulli inputSize, ulli intervalX, ulli intervalY) {
    ulli h_pos = threadIdx.x + blockIdx.x * blockDim.x;
    while( h_pos < inputSize ) {
        dev_output[h_pos] = calcuateZorder2d(dev_input[h_pos],dev_input2[h_pos],intervalX,intervalY);
        h_pos += gridDim.x * blockDim.x;
    }
}

void genGrouping2dZ(ulli *input, ulli* input2, ulli* &output, ulli inputSize) {

    ulli *host_output = new ulli[inputSize];
    output = host_output;
    // init cuda
    ulli *dev_input;
    ulli *dev_input2;
    ulli *dev_output;

    cudaEvent_t m_start, m_stop;
    float m_time;

    cudaSetDevice(0);

    cudaMalloc( (void**) &(dev_input), sizeof(ulli)*inputSize);
    cudaMalloc( (void**) &(dev_input2), sizeof(ulli)*inputSize);
    cudaMalloc( (void**) &(dev_output), sizeof(ulli)*inputSize);
    cudaMemcpy( dev_input, input, sizeof(ulli)*inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_input2, input2, sizeof(ulli)*inputSize, cudaMemcpyHostToDevice);
    cudaMemset( dev_output, 0, sizeof(ulli)*inputSize );


    cudaEventCreate( &m_start );
    cudaEventCreate( &m_stop );
    cudaEventRecord( m_start, 0 );

    ulli *maxValueA;
    ulli *maxValueB;

    {
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cudaMallocManaged( (void**) &(maxValueA), sizeof(ulli));
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, dev_input, maxValueA, inputSize);
        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // Run max-reduction
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, dev_input, maxValueA, inputSize);

        cudaDeviceSynchronize();

        cudaFree(d_temp_storage);

    }

    {
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cudaMallocManaged( (void**) &(maxValueB), sizeof(ulli));
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, dev_input2, maxValueB, inputSize);
        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // Run max-reduction
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, dev_input2, maxValueB, inputSize);

        CUDA_RT_CALL(cudaDeviceSynchronize());

        cudaFree(d_temp_storage);
    }

    int MaxBitA = 0;
    int MaxBitB = 0;
    ulli OriginMaxA = *maxValueA;
    ulli OriginMaxB = *maxValueB;
    int intervalX = ceil((float)OriginMaxA / (float)pow(2,32));
    int intervalY = ceil((float)OriginMaxB / (float)pow(2,32));

    while(*maxValueA) {
        *maxValueA = (*maxValueA)>>1;
        MaxBitA++;
    }
    while(*maxValueB) {
        *maxValueB = (*maxValueB)>>1;
        MaxBitB++;
    }

    int StartPassA = 16-ceil((float)MaxBitA/(float)BIT_PASS);
    int StartPassB = 16-ceil((float)MaxBitB/(float)BIT_PASS);



    // std::cout<<OriginMaxA<<","<<OriginMaxB<<std::endl;
    // std::cout<<intervalX<<","<<intervalY<<std::endl;

    ulli m_blocks = (ceil((float)inputSize /((float)BLOCK_SIZE)))>CUDA_MAX_BLOCK ? CUDA_MAX_BLOCK:ceil((float)inputSize /((float)BLOCK_SIZE));
    dim3 grid(m_blocks, 1, 1);
    dim3 threads(BLOCK_SIZE, 1, 1);

    calcuateZorderKernel2d<<<grid,threads>>>(dev_input, dev_input2, dev_output, inputSize, intervalX, intervalY);

    CUDA_RT_CALL(cudaDeviceSynchronize());
    cudaEventRecord( m_stop, 0 );
    cudaEventSynchronize( m_stop );
    cudaEventElapsedTime( &m_time, m_start, m_stop);
    cudaEventDestroy( m_start);
    cudaEventDestroy( m_stop);

    cudaMemcpy( host_output, dev_output, sizeof(ulli)*inputSize, cudaMemcpyDeviceToHost);

    cudaFree(dev_input);
    cudaFree(dev_input2);
    cudaFree(dev_output);

    std::cout<<m_time/1000<<std::endl;

    return;
}
