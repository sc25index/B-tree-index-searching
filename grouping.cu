#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cub/cub.cuh>

#include "utils.h"
#include "grouping.h"


#define BLOCK_SIZE 768

#define BIT_PASS 8 // number of bit each time
#define BUCK_SIZE 256
#define BIT_MASK 0xFF
#define CUDA_MAX_BLOCK 65535


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);


__host__ __device__ inline int bucketIdentify(ulli input, int nPass)  {

    int maxPass = (int)((sizeof(input)*8)/BIT_PASS) + ((((sizeof(input)*8)%BIT_PASS)>0)?1:0);
    return (input>>(BIT_PASS*(maxPass-1-nPass))) & BIT_MASK;
}

__global__ void kernelPrefixSum( ulli *hist, ulli * prefixSum) {
    ulli h_pos = threadIdx.x + blockIdx.x * blockDim.x;

    while(h_pos < BUCK_SIZE) {
        ulli tmpAns = 0;
        for(ulli i=0;i<h_pos;i++){
            tmpAns = tmpAns + hist[i];
        }
        prefixSum[h_pos] = tmpAns;
        //printf("%lld %lld\n",h_pos,tmpAns);
        h_pos += blockDim.x * gridDim.x;
    }
    __syncthreads();

}


__global__ void kernelRadixPass( const ulli *dev_input_, ulli *dev_Hist, int nPass, ulli inputSize) {
    __shared__ int SHist[BUCK_SIZE];
    const ulli2* __restrict dev_input = (ulli2*)dev_input_;
    ulli h_pos = threadIdx.x;
    while(h_pos < BUCK_SIZE) {
        SHist[h_pos]=0;
        h_pos += blockDim.x;
    }
    __syncthreads();

    h_pos = threadIdx.x + blockIdx.x * blockDim.x;
#if __CUDA_ARCH__  >= 750
    int twid = threadIdx.x%32;
#endif
    while( h_pos < inputSize/2 ) {
        ulli2 currentPtr = dev_input[h_pos];

#if __CUDA_ARCH__  >= 750
        if(((h_pos/32)*32)+32  < inputSize/2) {
            int bucketAns = bucketIdentify(currentPtr.a,nPass);
            int bitmap = __match_any_sync(0xffffffff,bucketAns);
            if(__ffs(bitmap)-1 == twid) {
                atomicAdd(&SHist[bucketAns],__popc(bitmap));
            }
            bucketAns = bucketIdentify(currentPtr.b,nPass);
            bitmap = __match_any_sync(0xffffffff,bucketAns);
            if(__ffs(bitmap)-1 == twid) {
                atomicAdd(&SHist[bucketAns],__popc(bitmap));
            }
        }
        else{
            int bucketAns = bucketIdentify(currentPtr.a,nPass);
            atomicAdd(&SHist[bucketAns],1);
            bucketAns = bucketIdentify(currentPtr.b,nPass);
            atomicAdd(&SHist[bucketAns],1);
        }
#else
        int bucketAns = bucketIdentify(currentPtr.a,nPass);
        atomicAdd(&SHist[bucketAns],1);
        bucketAns = bucketIdentify(currentPtr.b,nPass);
        atomicAdd(&SHist[bucketAns],1);
#endif

        h_pos += gridDim.x * blockDim.x;
    }

    {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            int remainder = inputSize%2;
            if(remainder != 0) {
                while(remainder) {
                    int idx = inputSize - remainder--;
                    int bucketAns = bucketIdentify(dev_input_[idx],nPass);
                    atomicAdd(&SHist[bucketAns],1);
                }
            }
        }
    }

    __syncthreads();
    h_pos = threadIdx.x;
    while( h_pos < BUCK_SIZE ) {
        atomicAdd(&dev_Hist[h_pos],SHist[h_pos]);
        h_pos += blockDim.x;
    }
}

__global__
void kernelRadixPassRelocateWithTwoData( const ulli *dev_input_, const ulli *dev_input_data_,
                                         const ulli *dev_input_data2_,
                                         ulli *dev_output, ulli *dev_output_data,
                                         ulli *dev_output_data2,
                                         ulli* dev_histogram, ulli *dev_Prefix,
                                         int nPass, ulli inputSize) {

    __shared__ ulli SHLocate[BUCK_SIZE];
    __shared__ int SHist[BUCK_SIZE];
    __shared__ ulli SBuffer[BLOCK_SIZE];
    __shared__ ulli SBuffer_Data[BLOCK_SIZE];
    __shared__ ulli SBuffer_Data2[BLOCK_SIZE];
    __shared__ int SInLoopPrefix[BUCK_SIZE];
    __shared__ int SInLoopBKPrefix[BUCK_SIZE];

    const ulli* __restrict dev_input = (ulli*)dev_input_;
    const ulli* __restrict dev_input_data = (ulli*)dev_input_data_;
    const ulli* __restrict dev_input_data2 = (ulli*)dev_input_data2_;
    ulli h_pos = threadIdx.x;
    while(h_pos < BUCK_SIZE) {
        SHist[h_pos]=0;
        h_pos += blockDim.x;
    }

    __syncthreads();
    h_pos = threadIdx.x + blockIdx.x * blockDim.x;
    while( h_pos < inputSize ) {

        ulli currentPtr = dev_input[h_pos];
        int bucketAns = bucketIdentify(currentPtr,nPass);
        atomicAdd(&SHist[bucketAns],1);
        h_pos += gridDim.x * blockDim.x;
    }

    __syncthreads();

    h_pos = threadIdx.x;
    while(h_pos < BUCK_SIZE) {
        SHLocate[h_pos] = atomicAdd(&dev_Prefix[h_pos],SHist[h_pos]);
        h_pos += blockDim.x;
    }
    __syncthreads();

    // reuse histogram for int-loop histogram
    // assume threadid.x always greater than BUCK_SIZE !!!
    // Fixme enable vector 2
    // Fixme when the inputSize isn't diviable by blocksize (on the 4 round)
    h_pos = threadIdx.x + blockIdx.x * blockDim.x;
    while( h_pos < inputSize ) {


        //reset inloop histogram
        ulli h_pos2 = threadIdx.x;
        while(h_pos2 < BUCK_SIZE) {
            SHist[h_pos2]=0;
            if( inputSize%BLOCK_SIZE != 0 && (h_pos-threadIdx.x+BLOCK_SIZE) > inputSize)
                h_pos2 += (inputSize%BLOCK_SIZE);
            else
                h_pos2 += blockDim.x;
        }
        SBuffer[threadIdx.x] = 0;
        SBuffer_Data[threadIdx.x] = 0;
        SBuffer_Data2[threadIdx.x] = 0;

        __syncthreads();

        ulli currentPtr = dev_input[h_pos];
        ulli currentPtr_data = dev_input_data[h_pos];
        ulli currentPtr_data2 = dev_input_data2[h_pos];
        int bucketAns = bucketIdentify(currentPtr,nPass);
        atomicAdd(&SHist[bucketAns],1);

        __syncthreads();

        //calcuate in loop prefix
        if(threadIdx.x == 0) {
            SInLoopPrefix[0] = 0;
            for(int i=1;i<BUCK_SIZE;i++) {
                SInLoopPrefix[i] = SInLoopPrefix[i-1] + SHist[i-1];
            }
        }

        __syncthreads();

        h_pos2 = threadIdx.x;
        while(h_pos2 < BUCK_SIZE) {
            SInLoopBKPrefix[h_pos2] = SInLoopPrefix[h_pos2];
            if( inputSize%BLOCK_SIZE != 0 && (h_pos-threadIdx.x+BLOCK_SIZE) > inputSize)
                h_pos2 += (inputSize%BLOCK_SIZE);
            else
                h_pos2 += blockDim.x;
        }

        __syncthreads();

        ulli offset = atomicAdd(&SInLoopPrefix[bucketAns],1);
        SBuffer[offset] = currentPtr;
        SBuffer_Data[offset] = currentPtr_data;
        SBuffer_Data2[offset] = currentPtr_data2;

        __syncthreads();

        {
            currentPtr = SBuffer[threadIdx.x];
            currentPtr_data = SBuffer_Data[threadIdx.x];
            currentPtr_data2 = SBuffer_Data2[threadIdx.x];
            bucketAns = bucketIdentify(currentPtr,nPass);
            offset = SHLocate[bucketAns]+ threadIdx.x - SInLoopBKPrefix[bucketAns];
            dev_output[offset] = currentPtr;
            dev_output_data[offset] = currentPtr_data;
            dev_output_data2[offset] = currentPtr_data2;

        }

        __syncthreads();

        h_pos2 = threadIdx.x;
        while(h_pos2 < BUCK_SIZE) {
            SHLocate[h_pos2] += SHist[h_pos2];
            if( inputSize%BLOCK_SIZE != 0 && (h_pos-threadIdx.x+BLOCK_SIZE) > inputSize)
                h_pos2 += (inputSize%BLOCK_SIZE);
            else
                h_pos2 += blockDim.x;
        }

        h_pos += (gridDim.x * blockDim.x);
    }

}

__global__
void kernelRadixPassRelocateWithOneData( const ulli *dev_input_, const ulli *dev_input_data_,
                                         ulli *dev_output, ulli *dev_output_data,
                                         ulli* dev_histogram, ulli *dev_Prefix,
                                         int nPass, ulli inputSize) {

    __shared__ ulli SHLocate[BUCK_SIZE];
    __shared__ int SHist[BUCK_SIZE];
    __shared__ ulli SBuffer[BLOCK_SIZE];
    __shared__ ulli SBuffer_Data[BLOCK_SIZE];
    __shared__ int SInLoopPrefix[BUCK_SIZE];
    __shared__ int SInLoopBKPrefix[BUCK_SIZE];

    const ulli* __restrict dev_input = (ulli*)dev_input_;
    const ulli* __restrict dev_input_data = (ulli*)dev_input_data_;
    ulli h_pos = threadIdx.x;
    while(h_pos < BUCK_SIZE) {
        SHist[h_pos]=0;
        h_pos += blockDim.x;
    }

    __syncthreads();
    h_pos = threadIdx.x + blockIdx.x * blockDim.x;
    while( h_pos < inputSize ) {

        ulli currentPtr = dev_input[h_pos];
        int bucketAns = bucketIdentify(currentPtr,nPass);
        atomicAdd(&SHist[bucketAns],1);
        h_pos += gridDim.x * blockDim.x;
    }

    __syncthreads();

    h_pos = threadIdx.x;
    while(h_pos < BUCK_SIZE) {
        SHLocate[h_pos] = atomicAdd(&dev_Prefix[h_pos],SHist[h_pos]);
        h_pos += blockDim.x;
    }
    __syncthreads();

    // reuse histogram for int-loop histogram
    // assume threadid.x always greater than BUCK_SIZE !!!
    // Fixme enable vector 2
    // Fixme when the inputSize isn't diviable by blocksize (on the 4 round)
    h_pos = threadIdx.x + blockIdx.x * blockDim.x;
    while( h_pos < inputSize ) {


        //reset inloop histogram
        ulli h_pos2 = threadIdx.x;
        while(h_pos2 < BUCK_SIZE) {
            SHist[h_pos2]=0;
            if( inputSize%BLOCK_SIZE != 0 && (h_pos-threadIdx.x+BLOCK_SIZE) > inputSize)
                h_pos2 += (inputSize%BLOCK_SIZE);
            else
                h_pos2 += blockDim.x;
        }
        SBuffer[threadIdx.x] = 0;
        SBuffer_Data[threadIdx.x] = 0;

        __syncthreads();

        ulli currentPtr = dev_input[h_pos];
        ulli currentPtr_data = dev_input_data[h_pos];
        int bucketAns = bucketIdentify(currentPtr,nPass);
        atomicAdd(&SHist[bucketAns],1);

        __syncthreads();

        //calcuate in loop prefix
        if(threadIdx.x == 0) {
            SInLoopPrefix[0] = 0;
            for(int i=1;i<BUCK_SIZE;i++) {
                SInLoopPrefix[i] = SInLoopPrefix[i-1] + SHist[i-1];
            }
        }

        __syncthreads();

        h_pos2 = threadIdx.x;
        while(h_pos2 < BUCK_SIZE) {
            SInLoopBKPrefix[h_pos2] = SInLoopPrefix[h_pos2];
            if( inputSize%BLOCK_SIZE != 0 && (h_pos-threadIdx.x+BLOCK_SIZE) > inputSize)
                h_pos2 += (inputSize%BLOCK_SIZE);
            else
                h_pos2 += blockDim.x;
        }

        __syncthreads();

        ulli offset = atomicAdd(&SInLoopPrefix[bucketAns],1);
        SBuffer[offset] = currentPtr;
        SBuffer_Data[offset] = currentPtr_data;

        __syncthreads();

        {
            currentPtr = SBuffer[threadIdx.x];
            currentPtr_data = SBuffer_Data[threadIdx.x];
            bucketAns = bucketIdentify(currentPtr,nPass);
            offset = SHLocate[bucketAns]+ threadIdx.x - SInLoopBKPrefix[bucketAns];
            dev_output[offset] = currentPtr;
            dev_output_data[offset] = currentPtr_data;

        }

        __syncthreads();

        h_pos2 = threadIdx.x;
        while(h_pos2 < BUCK_SIZE) {
            SHLocate[h_pos2] += SHist[h_pos2];
            if( inputSize%BLOCK_SIZE != 0 && (h_pos-threadIdx.x+BLOCK_SIZE) > inputSize)
                h_pos2 += (inputSize%BLOCK_SIZE);
            else
                h_pos2 += blockDim.x;
        }

        h_pos += (gridDim.x * blockDim.x);
    }

}

__global__ void kernelRadixPassRelocate( const ulli *dev_input_, ulli *dev_output, ulli* dev_histogram,
                                         ulli *dev_Prefix, int nPass, ulli inputSize) {

    __shared__ ulli SHLocate[BUCK_SIZE];
    __shared__ int SHist[BUCK_SIZE];
    __shared__ ulli SBuffer[BLOCK_SIZE];
    __shared__ int SInLoopPrefix[BUCK_SIZE];
    __shared__ int SInLoopBKPrefix[BUCK_SIZE];

    const ulli* __restrict dev_input = (ulli*)dev_input_;
    ulli h_pos = threadIdx.x;
    while(h_pos < BUCK_SIZE) {
        SHist[h_pos]=0;
        h_pos += blockDim.x;
    }

    __syncthreads();
    h_pos = threadIdx.x + blockIdx.x * blockDim.x;
    while( h_pos < inputSize ) {

        ulli currentPtr = dev_input[h_pos];
        int bucketAns = bucketIdentify(currentPtr,nPass);
        atomicAdd(&SHist[bucketAns],1);
        h_pos += gridDim.x * blockDim.x;
    }

    __syncthreads();

    h_pos = threadIdx.x;
    while(h_pos < BUCK_SIZE) {
        SHLocate[h_pos] = atomicAdd(&dev_Prefix[h_pos],SHist[h_pos]);
        h_pos += blockDim.x;
    }
    __syncthreads();

    // reuse histogram for int-loop histogram
    // assume threadid.x always greater than BUCK_SIZE !!!
    // Fixme enable vector 2
    // Fixme when the inputSize isn't diviable by blocksize (on the 4 round)
    h_pos = threadIdx.x + blockIdx.x * blockDim.x;
    while( h_pos < inputSize ) {


        //reset inloop histogram
        ulli h_pos2 = threadIdx.x;
        while(h_pos2 < BUCK_SIZE) {
            SHist[h_pos2]=0;
            if( inputSize%BLOCK_SIZE != 0 && (h_pos-threadIdx.x+BLOCK_SIZE) > inputSize)
                h_pos2 += (inputSize%BLOCK_SIZE);
            else
                h_pos2 += blockDim.x;
        }
        SBuffer[threadIdx.x] = 0;

        __syncthreads();

        ulli currentPtr = dev_input[h_pos];
        int bucketAns = bucketIdentify(currentPtr,nPass);
        atomicAdd(&SHist[bucketAns],1);

        __syncthreads();

        //calcuate in loop prefix
        if(threadIdx.x == 0) {
            SInLoopPrefix[0] = 0;
            for(int i=1;i<BUCK_SIZE;i++) {
                SInLoopPrefix[i] = SInLoopPrefix[i-1] + SHist[i-1];
            }
        }

        __syncthreads();

        h_pos2 = threadIdx.x;
        while(h_pos2 < BUCK_SIZE) {
            SInLoopBKPrefix[h_pos2] = SInLoopPrefix[h_pos2];
            if( inputSize%BLOCK_SIZE != 0 && (h_pos-threadIdx.x+BLOCK_SIZE) > inputSize)
                h_pos2 += (inputSize%BLOCK_SIZE);
            else
                h_pos2 += blockDim.x;
        }

        __syncthreads();

        ulli offset = atomicAdd(&SInLoopPrefix[bucketAns],1);
        SBuffer[offset] = currentPtr;

        __syncthreads();

        {
            currentPtr = SBuffer[threadIdx.x];
            bucketAns = bucketIdentify(currentPtr,nPass);
            offset = SHLocate[bucketAns]+ threadIdx.x - SInLoopBKPrefix[bucketAns];
            dev_output[offset] = currentPtr;

        }

        __syncthreads();

        h_pos2 = threadIdx.x;
        while(h_pos2 < BUCK_SIZE) {
            SHLocate[h_pos2] += SHist[h_pos2];
            if( inputSize%BLOCK_SIZE != 0 && (h_pos-threadIdx.x+BLOCK_SIZE) > inputSize)
                h_pos2 += (inputSize%BLOCK_SIZE);
            else
                h_pos2 += blockDim.x;
        }

        h_pos += (gridDim.x * blockDim.x);
    }

}

__global__ void kernelLocalSort( ulli *dev_input, ulli *dev_output, ulli inputSize) {

    extern __shared__ ulli SHCache[];
    ulli *SHCacheOut = &SHCache[inputSize];

    ulli h_pos = threadIdx.x;

    while( h_pos < inputSize ) {
        SHCache[h_pos] = dev_input[h_pos];
        h_pos += blockDim.x;
    }
    __syncthreads();

    h_pos = threadIdx.x;
    while(h_pos < inputSize){
        ulli currentPtr = SHCache[h_pos];
        int offset=0;
        for(int i=0;i<inputSize;i++) {
            if(SHCache[i]<currentPtr)
                offset++;
        }
        SHCacheOut[offset] = currentPtr;
        h_pos += blockDim.x;
    }
    __syncthreads();

    h_pos = threadIdx.x;
    while( h_pos < inputSize ) {
        dev_input[h_pos] = SHCacheOut[h_pos];
        h_pos += blockDim.x;
    }
}

__global__
void kernelRadixPassSingleKernelWithTwoData( const ulli *dev_input_, const ulli *dev_input_data_,
                                             const ulli *dev_input_data2_,
                                             ulli *dev_prefix, ulli *dev_group_cout,
                                             int nPass, ulli *dev_output, ulli *dev_output_data,
                                             ulli *dev_output_data2 ,ulli group_num,
                                             ulli *dev_groupNum_out, ulli Local_Group_Size) {
    __shared__ int SHist[BUCK_SIZE];
    __shared__ int SPrefix[BUCK_SIZE];

    ulli currentGroup = blockIdx.x;
    ulli h_pos;

    while(currentGroup < group_num){
        const ulli* __restrict dev_input = &dev_input_[dev_prefix[currentGroup]];
        const ulli* __restrict dev_input_data = &dev_input_data_[dev_prefix[currentGroup]];
        const ulli* __restrict dev_input_data2 = &dev_input_data2_[dev_prefix[currentGroup]];
        if(dev_group_cout[currentGroup] > Local_Group_Size) {
            h_pos = threadIdx.x;
            while(h_pos < BUCK_SIZE) {
                SHist[h_pos]=0;
                SPrefix[h_pos]=0;
                h_pos += blockDim.x;
            }
            __syncthreads();

            h_pos = threadIdx.x;

            // calcuate histogram
            while( h_pos < dev_group_cout[currentGroup] ) {

                ulli currentPtr = dev_input[h_pos];
                int bucketAns = bucketIdentify(currentPtr,nPass);
                atomicAdd(&SHist[bucketAns],1);
                h_pos += blockDim.x;
            }

            __syncthreads();

            h_pos = threadIdx.x;
            ulli *currentGroupList = &dev_groupNum_out[BUCK_SIZE*currentGroup];
            while(h_pos < BUCK_SIZE) {
                currentGroupList[h_pos] = SHist[h_pos];
                h_pos += blockDim.x;
            }

            // calcuate prefix sum
            if( threadIdx.x == 0 ) {
                SPrefix[0]=0;
                for(int i=1;i<BUCK_SIZE;i++){
                    SPrefix[i]=SPrefix[i-1] + SHist[i-1];
                }
            }

            __syncthreads();

            h_pos = threadIdx.x;
            while( h_pos < dev_group_cout[currentGroup] ) {

                ulli currentPtr = dev_input[h_pos];
                ulli currentPtr_data = dev_input_data[h_pos];
                ulli currentPtr_data2 = dev_input_data2[h_pos];
                int bucketAns = bucketIdentify(currentPtr,nPass);
                int offset = atomicAdd(&SPrefix[bucketAns],1);
                dev_output[dev_prefix[currentGroup]+offset] = currentPtr;
                dev_output_data[dev_prefix[currentGroup]+offset] = currentPtr_data;
                dev_output_data2[dev_prefix[currentGroup]+offset] = currentPtr_data2;
                h_pos += (blockDim.x);
            }

            __syncthreads();
        }
        else {
            if( threadIdx.x == 0 ) {
                ulli *currentGroupList = &dev_groupNum_out[BUCK_SIZE*currentGroup];
                currentGroupList[0] = dev_group_cout[currentGroup];
            }
            h_pos = threadIdx.x;
            while( h_pos < dev_group_cout[currentGroup] ) {
                dev_output[dev_prefix[currentGroup]+h_pos] = dev_input[h_pos];
                dev_output_data[dev_prefix[currentGroup]+h_pos] = dev_input_data[h_pos];
                dev_output_data2[dev_prefix[currentGroup]+h_pos] = dev_input_data2[h_pos];
                h_pos += (blockDim.x);
            }

            __syncthreads();
        }
        currentGroup += gridDim.x;
    }

}

__global__
void kernelRadixPassSingleKernelWithOneData( const ulli *dev_input_, const ulli *dev_input_data_,
                                             ulli *dev_prefix, ulli *dev_group_cout,
                                             int nPass, ulli *dev_output, ulli *dev_output_data,
                                             ulli group_num, ulli *dev_groupNum_out,
                                             ulli Local_Group_Size) {
    __shared__ int SHist[BUCK_SIZE];
    __shared__ int SPrefix[BUCK_SIZE];

    ulli currentGroup = blockIdx.x;
    ulli h_pos;

    while(currentGroup < group_num){
        const ulli* __restrict dev_input = &dev_input_[dev_prefix[currentGroup]];
        const ulli* __restrict dev_input_data = &dev_input_data_[dev_prefix[currentGroup]];
        if(dev_group_cout[currentGroup] > Local_Group_Size) {
            h_pos = threadIdx.x;
            while(h_pos < BUCK_SIZE) {
                SHist[h_pos]=0;
                SPrefix[h_pos]=0;
                h_pos += blockDim.x;
            }
            __syncthreads();

            h_pos = threadIdx.x;

            // calcuate histogram
            while( h_pos < dev_group_cout[currentGroup] ) {

                ulli currentPtr = dev_input[h_pos];
                int bucketAns = bucketIdentify(currentPtr,nPass);
                atomicAdd(&SHist[bucketAns],1);
                h_pos += blockDim.x;
            }

            __syncthreads();

            h_pos = threadIdx.x;
            ulli *currentGroupList = &dev_groupNum_out[BUCK_SIZE*currentGroup];
            while(h_pos < BUCK_SIZE) {
                currentGroupList[h_pos] = SHist[h_pos];
                h_pos += blockDim.x;
            }

            // calcuate prefix sum
            if( threadIdx.x == 0 ) {
                SPrefix[0]=0;
                for(int i=1;i<BUCK_SIZE;i++){
                    SPrefix[i]=SPrefix[i-1] + SHist[i-1];
                }
            }

            __syncthreads();

            h_pos = threadIdx.x;
            while( h_pos < dev_group_cout[currentGroup] ) {

                ulli currentPtr = dev_input[h_pos];
                ulli currentPtr_data = dev_input_data[h_pos];
                int bucketAns = bucketIdentify(currentPtr,nPass);
                int offset = atomicAdd(&SPrefix[bucketAns],1);
                dev_output[dev_prefix[currentGroup]+offset] = currentPtr;
                dev_output_data[dev_prefix[currentGroup]+offset] = currentPtr_data;
                h_pos += (blockDim.x);
            }

            __syncthreads();
        }
        else {
            if( threadIdx.x == 0 ) {
                ulli *currentGroupList = &dev_groupNum_out[BUCK_SIZE*currentGroup];
                currentGroupList[0] = dev_group_cout[currentGroup];
            }
            h_pos = threadIdx.x;
            while( h_pos < dev_group_cout[currentGroup] ) {
                dev_output[dev_prefix[currentGroup]+h_pos] = dev_input[h_pos];
                dev_output_data[dev_prefix[currentGroup]+h_pos] = dev_input_data[h_pos];
                h_pos += (blockDim.x);
            }

            __syncthreads();
        }
        currentGroup += gridDim.x;
    }

}

__global__ void kernelRadixPassSingleKernel( const ulli *dev_input_, ulli *dev_prefix, ulli *dev_group_cout,
                                             int nPass, ulli *dev_output, ulli group_num, ulli *dev_groupNum_out,
                                             ulli Local_Group_Size) {
    __shared__ int SHist[BUCK_SIZE];
    __shared__ int SPrefix[BUCK_SIZE];

    ulli currentGroup = blockIdx.x;
    ulli h_pos;

    while(currentGroup < group_num){
        const ulli* __restrict dev_input = &dev_input_[dev_prefix[currentGroup]];
        if(dev_group_cout[currentGroup] > Local_Group_Size) {
            h_pos = threadIdx.x;
            while(h_pos < BUCK_SIZE) {
                SHist[h_pos]=0;
                SPrefix[h_pos]=0;
                h_pos += blockDim.x;
            }
            __syncthreads();

            h_pos = threadIdx.x;

            // calcuate histogram
            while( h_pos < dev_group_cout[currentGroup] ) {

                ulli currentPtr = dev_input[h_pos];
                int bucketAns = bucketIdentify(currentPtr,nPass);
                atomicAdd(&SHist[bucketAns],1);
                h_pos += blockDim.x;
            }

            __syncthreads();

            h_pos = threadIdx.x;
            ulli *currentGroupList = &dev_groupNum_out[BUCK_SIZE*currentGroup];
            while(h_pos < BUCK_SIZE) {
                currentGroupList[h_pos] = SHist[h_pos];
                h_pos += blockDim.x;
            }

            // calcuate prefix sum
            if( threadIdx.x == 0 ) {
                SPrefix[0]=0;
                for(int i=1;i<BUCK_SIZE;i++){
                    SPrefix[i]=SPrefix[i-1] + SHist[i-1];
                }
            }

            __syncthreads();

            h_pos = threadIdx.x;
            while( h_pos < dev_group_cout[currentGroup] ) {

                ulli currentPtr = dev_input[h_pos];
                int bucketAns = bucketIdentify(currentPtr,nPass);
                int offset = atomicAdd(&SPrefix[bucketAns],1);
                dev_output[dev_prefix[currentGroup]+offset] = currentPtr;
                h_pos += (blockDim.x);
            }

            __syncthreads();
        }
        else {
            if( threadIdx.x == 0 ) {
                ulli *currentGroupList = &dev_groupNum_out[BUCK_SIZE*currentGroup];
                currentGroupList[0] = dev_group_cout[currentGroup];
            }
            h_pos = threadIdx.x;
            while( h_pos < dev_group_cout[currentGroup] ) {
                dev_output[dev_prefix[currentGroup]+h_pos] = dev_input[h_pos];
                h_pos += (blockDim.x);
            }

            __syncthreads();
        }
        currentGroup += gridDim.x;
    }

}


// groupNum is number of groupt
// groupList is size of each group
// groupPrefix is prefix of each group in dev_input
void radixGroupGPU(ulli *dev_input, ulli *dev_output, std::vector<ulli> &groupList,
                    std::vector<ulli> &groupPrefix, int pass, ulli Local_Group_Size) {

    std::vector<ulli> currentList;
    std::vector<ulli> currentPrefix;

    for(int i=0;i<groupList.size();i++) {
        if(groupList[i] == 0) {
            continue;
        }
        else if(groupList[i] < Local_Group_Size) {
            currentList.push_back(groupList[i]);
            currentPrefix.push_back(groupPrefix[i]);
            cudaMemcpyAsync( dev_output, dev_input, sizeof(ulli)*groupList[i], cudaMemcpyDeviceToDevice);
            continue;
        }

        ulli *dev_histogram;
        ulli *devPrefix;
        ulli *host_Prefix = new ulli[BUCK_SIZE];
        ulli *host_histogram = new ulli[BUCK_SIZE];

        cudaMalloc( (void**) &(dev_histogram), sizeof(ulli)*BUCK_SIZE);
        cudaMalloc( (void**) &devPrefix, sizeof(ulli)*BUCK_SIZE);

        cudaMemset( devPrefix, 0, sizeof(ulli)*BUCK_SIZE );
        cudaMemset( dev_histogram, 0, sizeof(ulli)*BUCK_SIZE );

        //std::cout<<"Number of input "<<groupList[i]<<std::endl;

        ulli m_blocks = (groupList[i] / (BLOCK_SIZE))>CUDA_MAX_BLOCK ? CUDA_MAX_BLOCK:groupList[i] /(BLOCK_SIZE);
        dim3 grid(m_blocks, 1, 1);
        dim3 threads(BLOCK_SIZE, 1, 1);

        //std::cout<<"Number of Block "<<m_blocks<<std::endl;
        //std::cout<<"Number of pass "<<pass<<std::endl;


        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        kernelRadixPass<<<grid,threads>>>(&dev_input[groupPrefix[i]],dev_histogram,pass, groupList[i]);


        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );


        cudaMemcpyAsync( host_histogram, dev_histogram, sizeof(ulli)*BUCK_SIZE, cudaMemcpyDeviceToHost);

        dim3 grid2(1, 1, 1);
        dim3 threads2(512, 1, 1);
        kernelPrefixSum<<<grid2,threads2>>>(dev_histogram,devPrefix);


        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        int validBucket=0;
        for(int j=0;j<BUCK_SIZE;j++) {
            if(host_histogram[j] != 0) {
                validBucket++;
            }
        }

        if(validBucket == 1 && groupList.size() == 1) {
            currentList.push_back(groupList[i]);
            currentPrefix.push_back(groupPrefix[i]);
        }
        else {
            //note must copy prefix before call relocate!!!
            cudaMemcpy( host_Prefix, devPrefix, sizeof(ulli)*BUCK_SIZE, cudaMemcpyDeviceToHost);
            kernelRadixPassRelocate<<<grid,threads>>>(&dev_input[groupPrefix[i]],&dev_output[groupPrefix[i]],dev_histogram,
                                                        devPrefix, pass, groupList[i]);

            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaStreamSynchronize(0) );


            for(int j=0;j<BUCK_SIZE;j++) {
                if(host_histogram[j] != 0 ) {
                    ulli k=0;
                    ulli currentCount = host_histogram[j];
                    while(currentCount < Local_Group_Size && (j+k+1) < BUCK_SIZE) {
                        if(currentCount+host_histogram[j+k+1]<Local_Group_Size) {
                            currentCount = currentCount + host_histogram[j+k+1];
                            k++;
                        }
                        else{
                            break;
                        }
                    }
                    currentList.push_back(currentCount);
                    currentPrefix.push_back(groupPrefix[i]+host_Prefix[j]);
                    j += k;
                }
            }
        }

        cudaFree(dev_histogram);
        cudaFree(devPrefix);

        delete host_Prefix;
        delete host_histogram;
    }

    groupList = currentList;
    groupPrefix = currentPrefix;
}

// groupNum is number of groupt
// groupList is size of each group
// groupPrefix is prefix of each group in dev_input
void radixGroupGPUWithOneData(ulli *dev_input, ulli *dev_input_data, ulli *dev_output,
                              ulli *dev_output_data, std::vector<ulli> &groupList,
                              std::vector<ulli> &groupPrefix, int pass,
                              ulli Local_Group_Size) {

    std::vector<ulli> currentList;
    std::vector<ulli> currentPrefix;

    for(int i=0;i<groupList.size();i++) {
        if(groupList[i] == 0) {
            continue;
        }
        else if(groupList[i] < Local_Group_Size) {
            currentList.push_back(groupList[i]);
            currentPrefix.push_back(groupPrefix[i]);
            cudaMemcpyAsync( dev_output, dev_input, sizeof(ulli)*groupList[i], cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync( dev_output_data, dev_input_data, sizeof(ulli)*groupList[i], cudaMemcpyDeviceToDevice);
            continue;
        }

        ulli *dev_histogram;
        ulli *devPrefix;
        ulli *host_Prefix = new ulli[BUCK_SIZE];
        ulli *host_histogram = new ulli[BUCK_SIZE];

        cudaMalloc( (void**) &(dev_histogram), sizeof(ulli)*BUCK_SIZE);
        cudaMalloc( (void**) &devPrefix, sizeof(ulli)*BUCK_SIZE);

        cudaMemset( devPrefix, 0, sizeof(ulli)*BUCK_SIZE );
        cudaMemset( dev_histogram, 0, sizeof(ulli)*BUCK_SIZE );

        //std::cout<<"Number of input "<<groupList[i]<<std::endl;

        ulli m_blocks = (groupList[i] / (BLOCK_SIZE))>CUDA_MAX_BLOCK ? CUDA_MAX_BLOCK:groupList[i] /(BLOCK_SIZE);
        dim3 grid(m_blocks, 1, 1);
        dim3 threads(BLOCK_SIZE, 1, 1);

        //std::cout<<"Number of Block "<<m_blocks<<std::endl;
        //std::cout<<"Number of pass "<<pass<<std::endl;


        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        kernelRadixPass<<<grid,threads>>>(&dev_input[groupPrefix[i]],dev_histogram,pass, groupList[i]);


        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );


        cudaMemcpyAsync( host_histogram, dev_histogram, sizeof(ulli)*BUCK_SIZE, cudaMemcpyDeviceToHost);

        dim3 grid2(1, 1, 1);
        dim3 threads2(512, 1, 1);
        kernelPrefixSum<<<grid2,threads2>>>(dev_histogram,devPrefix);


        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        int validBucket=0;
        for(int j=0;j<BUCK_SIZE;j++) {
            if(host_histogram[j] != 0) {
                validBucket++;
            }
        }

        if(validBucket == 1 && groupList.size() == 1) {
            currentList.push_back(groupList[i]);
            currentPrefix.push_back(groupPrefix[i]);
        }
        else {
            //note must copy prefix before call relocate!!!
            cudaMemcpy( host_Prefix, devPrefix, sizeof(ulli)*BUCK_SIZE, cudaMemcpyDeviceToHost);
            kernelRadixPassRelocateWithOneData<<<grid,threads>>>(&dev_input[groupPrefix[i]],&dev_input_data[groupPrefix[i]],
                                                                 &dev_output[groupPrefix[i]],&dev_output_data[groupPrefix[i]],
                                                                 dev_histogram, devPrefix, pass, groupList[i]);

            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaStreamSynchronize(0) );


            for(int j=0;j<BUCK_SIZE;j++) {
                if(host_histogram[j] != 0 ) {
                    ulli k=0;
                    ulli currentCount = host_histogram[j];
                    while(currentCount < Local_Group_Size && (j+k+1) < BUCK_SIZE) {
                        if(currentCount+host_histogram[j+k+1]<Local_Group_Size) {
                            currentCount = currentCount + host_histogram[j+k+1];
                            k++;
                        }
                        else{
                            break;
                        }
                    }
                    currentList.push_back(currentCount);
                    currentPrefix.push_back(groupPrefix[i]+host_Prefix[j]);
                    j += k;
                }
            }
        }

        cudaFree(dev_histogram);
        cudaFree(devPrefix);

        delete host_Prefix;
        delete host_histogram;
    }

    groupList = currentList;
    groupPrefix = currentPrefix;
}

// groupNum is number of groupt
// groupList is size of each group
// groupPrefix is prefix of each group in dev_input
void radixGroupGPUWithTwoData(ulli *dev_input, ulli *dev_input_data, ulli *dev_input_data2,
                              ulli *dev_output, ulli *dev_output_data, ulli *dev_output_data2,
                              std::vector<ulli> &groupList, std::vector<ulli> &groupPrefix,
                              int pass, ulli Local_Group_Size) {

    std::vector<ulli> currentList;
    std::vector<ulli> currentPrefix;

    for(int i=0;i<groupList.size();i++) {
        if(groupList[i] == 0) {
            continue;
        }
        else if(groupList[i] < Local_Group_Size) {
            currentList.push_back(groupList[i]);
            currentPrefix.push_back(groupPrefix[i]);
            cudaMemcpyAsync( dev_output, dev_input, sizeof(ulli)*groupList[i], cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync( dev_output_data, dev_input_data, sizeof(ulli)*groupList[i], cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync( dev_output_data2, dev_input_data2, sizeof(ulli)*groupList[i], cudaMemcpyDeviceToDevice);
            continue;
        }

        ulli *dev_histogram;
        ulli *devPrefix;
        ulli *host_Prefix = new ulli[BUCK_SIZE];
        ulli *host_histogram = new ulli[BUCK_SIZE];

        cudaMalloc( (void**) &(dev_histogram), sizeof(ulli)*BUCK_SIZE);
        cudaMalloc( (void**) &devPrefix, sizeof(ulli)*BUCK_SIZE);

        cudaMemset( devPrefix, 0, sizeof(ulli)*BUCK_SIZE );
        cudaMemset( dev_histogram, 0, sizeof(ulli)*BUCK_SIZE );

        //std::cout<<"Number of input "<<groupList[i]<<std::endl;

        ulli m_blocks = (groupList[i] / (BLOCK_SIZE))>CUDA_MAX_BLOCK ? CUDA_MAX_BLOCK:groupList[i] /(BLOCK_SIZE);
        dim3 grid(m_blocks, 1, 1);
        dim3 threads(BLOCK_SIZE, 1, 1);

        //std::cout<<"Number of Block "<<m_blocks<<std::endl;
        //std::cout<<"Number of pass "<<pass<<std::endl;


        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        kernelRadixPass<<<grid,threads>>>(&dev_input[groupPrefix[i]],dev_histogram,pass, groupList[i]);


        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );


        cudaMemcpyAsync( host_histogram, dev_histogram, sizeof(ulli)*BUCK_SIZE, cudaMemcpyDeviceToHost);

        dim3 grid2(1, 1, 1);
        dim3 threads2(512, 1, 1);
        kernelPrefixSum<<<grid2,threads2>>>(dev_histogram,devPrefix);


        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        int validBucket=0;
        for(int j=0;j<BUCK_SIZE;j++) {
            if(host_histogram[j] != 0) {
                validBucket++;
            }
        }

        if(validBucket == 1 && groupList.size() == 1) {
            currentList.push_back(groupList[i]);
            currentPrefix.push_back(groupPrefix[i]);
        }
        else {
            //note must copy prefix before call relocate!!!
            cudaMemcpy( host_Prefix, devPrefix, sizeof(ulli)*BUCK_SIZE, cudaMemcpyDeviceToHost);
            kernelRadixPassRelocateWithTwoData<<<grid,threads>>>(&dev_input[groupPrefix[i]],&dev_input_data[groupPrefix[i]],
                                                                 &dev_input_data2[groupPrefix[i]], &dev_output[groupPrefix[i]],
                                                                 &dev_output_data[groupPrefix[i]], &dev_output_data2[groupPrefix[i]],
                                                                 dev_histogram, devPrefix, pass, groupList[i]);

            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaStreamSynchronize(0) );


            for(int j=0;j<BUCK_SIZE;j++) {
                if(host_histogram[j] != 0 ) {
                    ulli k=0;
                    ulli currentCount = host_histogram[j];
                    while(currentCount < Local_Group_Size && (j+k+1) < BUCK_SIZE) {
                        if(currentCount+host_histogram[j+k+1]<Local_Group_Size) {
                            currentCount = currentCount + host_histogram[j+k+1];
                            k++;
                        }
                        else{
                            break;
                        }
                    }
                    currentList.push_back(currentCount);
                    currentPrefix.push_back(groupPrefix[i]+host_Prefix[j]);
                    j += k;
                }
            }
        }

        cudaFree(dev_histogram);
        cudaFree(devPrefix);

        delete host_Prefix;
        delete host_histogram;
    }

    groupList = currentList;
    groupPrefix = currentPrefix;
}

void radixGroupSingleKernelGPUWithTwoData(ulli *dev_input, ulli *dev_input_data, ulli *dev_input_data2,
                                          ulli *dev_output, ulli *dev_output_data, ulli *dev_output_data2,
                                          std::vector<ulli> &groupList, std::vector<ulli> &groupPrefix,
                                          int pass, ulli Local_Group_Size) {

    std::vector<ulli> currentList;
    std::vector<ulli> currentPrefix;

    ulli *dev_group_cout, *host_group_count;
    ulli *dev_prefix, *host_prefix;
    ulli *dev_groupOutList, *host_groupOutList;

    host_group_count = new ulli[groupList.size()];
    host_prefix = new ulli[groupList.size()];
    host_groupOutList = new ulli[BUCK_SIZE*groupList.size()];
    //std::cout<<"Allocation Complete!!"<<std::endl;

    ulli tmpcheck = 0;
    for(ulli i=0;i<groupList.size();i++) {
        host_group_count[i] = groupList[i];
        host_prefix[i] = groupPrefix[i];
        tmpcheck += groupList[i];
    }
    //std::cout<<"tmpcheck = "<<tmpcheck<<std::endl;
    cudaMalloc( (void**) &(dev_group_cout), sizeof(ulli)*groupList.size());
    cudaMalloc( (void**) &(dev_prefix), sizeof(ulli)*groupList.size());
    cudaMalloc( (void**) &(dev_groupOutList), sizeof(ulli)*BUCK_SIZE*groupList.size());
    //std::cout<<"CUDA Allocation Complete!!"<<std::endl;

    cudaMemcpy( dev_group_cout, host_group_count, sizeof(ulli)*groupList.size(), cudaMemcpyHostToDevice);
    cudaMemcpy( dev_prefix, host_prefix, sizeof(ulli)*groupList.size(), cudaMemcpyHostToDevice);
    cudaMemset( dev_groupOutList, 0, sizeof(ulli)*BUCK_SIZE*groupList.size() );

    gpuErrchk( cudaPeekAtLastError() );

    ulli m_blocks = groupList.size()>CUDA_MAX_BLOCK ? CUDA_MAX_BLOCK:groupList.size();
    dim3 grid(m_blocks, 1, 1);
    dim3 threads(BLOCK_SIZE, 1, 1);
    //std::cout<<"Number of Group: "<<groupList.size()<<std::endl;
    //std::cout<<"Block size: "<<m_blocks<<std::endl;
    kernelRadixPassSingleKernelWithTwoData<<<grid,threads>>>( dev_input, dev_input_data, dev_input_data2,
                                                              dev_prefix, dev_group_cout, pass, dev_output,
                                                              dev_output_data, dev_output_data2,
                                                              groupList.size(), dev_groupOutList,
                                                              Local_Group_Size);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaStreamSynchronize(0) );

    cudaMemcpy( host_groupOutList, dev_groupOutList, sizeof(ulli)*BUCK_SIZE*groupList.size(), cudaMemcpyDeviceToHost);

    ulli prefixTmp=0;
    for(ulli i = 0;i<BUCK_SIZE*groupList.size();i++) {
        if(host_groupOutList[i] != 0) {
            ulli j=0;
            ulli currentCount = host_groupOutList[i];
            while(currentCount < Local_Group_Size && (i+j+1) < BUCK_SIZE*groupList.size()) {
                if(currentCount+host_groupOutList[i+j+1]<Local_Group_Size) {
                    currentCount = currentCount + host_groupOutList[i+j+1];
                    j++;
                }
                else{
                    break;
                }
            }
            currentList.push_back(currentCount);
            currentPrefix.push_back(prefixTmp);
            prefixTmp += currentCount;
            i = i+j;
        }
    }

    cudaFree(dev_group_cout);
    cudaFree(dev_groupOutList);
    cudaFree(dev_prefix);
    delete host_group_count;
    delete host_prefix;
    delete host_groupOutList;

    groupList = currentList;
    groupPrefix = currentPrefix;
}

void radixGroupSingleKernelGPUWithOneData(ulli *dev_input, ulli *dev_input_data,
                                          ulli *dev_output, ulli *dev_output_data,
                                          std::vector<ulli> &groupList,
                                          std::vector<ulli> &groupPrefix,
                                          int pass, ulli Local_Group_Size) {

    std::vector<ulli> currentList;
    std::vector<ulli> currentPrefix;

    ulli *dev_group_cout, *host_group_count;
    ulli *dev_prefix, *host_prefix;
    ulli *dev_groupOutList, *host_groupOutList;

    host_group_count = new ulli[groupList.size()];
    host_prefix = new ulli[groupList.size()];
    host_groupOutList = new ulli[BUCK_SIZE*groupList.size()];
    //std::cout<<"Allocation Complete!!"<<std::endl;

    ulli tmpcheck = 0;
    for(ulli i=0;i<groupList.size();i++) {
        host_group_count[i] = groupList[i];
        host_prefix[i] = groupPrefix[i];
        tmpcheck += groupList[i];
    }
    //std::cout<<"tmpcheck = "<<tmpcheck<<std::endl;
    cudaMalloc( (void**) &(dev_group_cout), sizeof(ulli)*groupList.size());
    cudaMalloc( (void**) &(dev_prefix), sizeof(ulli)*groupList.size());
    cudaMalloc( (void**) &(dev_groupOutList), sizeof(ulli)*BUCK_SIZE*groupList.size());
    //std::cout<<"CUDA Allocation Complete!!"<<std::endl;

    cudaMemcpy( dev_group_cout, host_group_count, sizeof(ulli)*groupList.size(), cudaMemcpyHostToDevice);
    cudaMemcpy( dev_prefix, host_prefix, sizeof(ulli)*groupList.size(), cudaMemcpyHostToDevice);
    cudaMemset( dev_groupOutList, 0, sizeof(ulli)*BUCK_SIZE*groupList.size() );

    gpuErrchk( cudaPeekAtLastError() );

    ulli m_blocks = groupList.size()>CUDA_MAX_BLOCK ? CUDA_MAX_BLOCK:groupList.size();
    dim3 grid(m_blocks, 1, 1);
    dim3 threads(BLOCK_SIZE, 1, 1);
    //std::cout<<"Number of Group: "<<groupList.size()<<std::endl;
    //std::cout<<"Block size: "<<m_blocks<<std::endl;
    kernelRadixPassSingleKernelWithOneData<<<grid,threads>>>( dev_input, dev_input_data, dev_prefix,
                                                              dev_group_cout, pass, dev_output,
                                                              dev_output_data, groupList.size(),
                                                              dev_groupOutList, Local_Group_Size);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaStreamSynchronize(0) );

    cudaMemcpy( host_groupOutList, dev_groupOutList, sizeof(ulli)*BUCK_SIZE*groupList.size(), cudaMemcpyDeviceToHost);

    ulli prefixTmp=0;
    for(ulli i = 0;i<BUCK_SIZE*groupList.size();i++) {
        if(host_groupOutList[i] != 0) {
            ulli j=0;
            ulli currentCount = host_groupOutList[i];
            while(currentCount < Local_Group_Size && (i+j+1) < BUCK_SIZE*groupList.size()) {
                if(currentCount+host_groupOutList[i+j+1]<Local_Group_Size) {
                    currentCount = currentCount + host_groupOutList[i+j+1];
                    j++;
                }
                else{
                    break;
                }
            }
            currentList.push_back(currentCount);
            currentPrefix.push_back(prefixTmp);
            prefixTmp += currentCount;
            i = i+j;
        }
    }

    cudaFree(dev_group_cout);
    cudaFree(dev_groupOutList);
    cudaFree(dev_prefix);
    delete host_group_count;
    delete host_prefix;
    delete host_groupOutList;

    groupList = currentList;
    groupPrefix = currentPrefix;
}

void radixGroupSingleKernelGPU(ulli *dev_input, ulli *dev_output, std::vector<ulli> &groupList,
                                std::vector<ulli> &groupPrefix, int pass, ulli Local_Group_Size) {

    std::vector<ulli> currentList;
    std::vector<ulli> currentPrefix;

    ulli *dev_group_cout, *host_group_count;
    ulli *dev_prefix, *host_prefix;
    ulli *dev_groupOutList, *host_groupOutList;

    host_group_count = new ulli[groupList.size()];
    host_prefix = new ulli[groupList.size()];
    host_groupOutList = new ulli[BUCK_SIZE*groupList.size()];
    //std::cout<<"Allocation Complete!!"<<std::endl;

    ulli tmpcheck = 0;
    for(ulli i=0;i<groupList.size();i++) {
        host_group_count[i] = groupList[i];
        host_prefix[i] = groupPrefix[i];
        tmpcheck += groupList[i];
    }
    //std::cout<<"tmpcheck = "<<tmpcheck<<std::endl;
    cudaMalloc( (void**) &(dev_group_cout), sizeof(ulli)*groupList.size());
    cudaMalloc( (void**) &(dev_prefix), sizeof(ulli)*groupList.size());
    cudaMalloc( (void**) &(dev_groupOutList), sizeof(ulli)*BUCK_SIZE*groupList.size());
    //std::cout<<"CUDA Allocation Complete!!"<<std::endl;

    cudaMemcpy( dev_group_cout, host_group_count, sizeof(ulli)*groupList.size(), cudaMemcpyHostToDevice);
    cudaMemcpy( dev_prefix, host_prefix, sizeof(ulli)*groupList.size(), cudaMemcpyHostToDevice);
    cudaMemset( dev_groupOutList, 0, sizeof(ulli)*BUCK_SIZE*groupList.size() );

    gpuErrchk( cudaPeekAtLastError() );

    ulli m_blocks = groupList.size()>CUDA_MAX_BLOCK ? CUDA_MAX_BLOCK:groupList.size();
    dim3 grid(m_blocks, 1, 1);
    dim3 threads(BLOCK_SIZE, 1, 1);
    //std::cout<<"Number of Group: "<<groupList.size()<<std::endl;
    //std::cout<<"Block size: "<<m_blocks<<std::endl;
    kernelRadixPassSingleKernel<<<grid,threads>>>( dev_input, dev_prefix, dev_group_cout, pass, dev_output, groupList.size(),dev_groupOutList, Local_Group_Size);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaStreamSynchronize(0) );

    cudaMemcpy( host_groupOutList, dev_groupOutList, sizeof(ulli)*BUCK_SIZE*groupList.size(), cudaMemcpyDeviceToHost);

    ulli prefixTmp=0;
    for(ulli i = 0;i<BUCK_SIZE*groupList.size();i++) {
        if(host_groupOutList[i] != 0) {
            ulli j=0;
            ulli currentCount = host_groupOutList[i];
            while(currentCount < Local_Group_Size && (i+j+1) < BUCK_SIZE*groupList.size()) {
                if(currentCount+host_groupOutList[i+j+1]<Local_Group_Size) {
                    currentCount = currentCount + host_groupOutList[i+j+1];
                    j++;
                }
                else{
                    break;
                }
            }
            currentList.push_back(currentCount);
            currentPrefix.push_back(prefixTmp);
            prefixTmp += currentCount;
            i = i+j;
        }
    }

    cudaFree(dev_group_cout);
    cudaFree(dev_groupOutList);
    cudaFree(dev_prefix);
    delete host_group_count;
    delete host_prefix;
    delete host_groupOutList;

    groupList = currentList;
    groupPrefix = currentPrefix;
}

void genGroupingWithTwodata(ulli *input, ulli *input_data, ulli *input_data2, ulli inputSize,
                            ulli *&output, ulli* &output_data, ulli *&output_data2,
                            std::vector<ulli> &groupList, std::vector<ulli> &groupPrefix,
                            ulli Local_Group_Size) {

    ulli *host_output = new ulli[inputSize];
    ulli *host_output_data = new ulli[inputSize];
    ulli *host_output_data2 = new ulli[inputSize];
    output = host_output;
    output_data = host_output_data;
    output_data2 = host_output_data2;
    groupList.push_back(inputSize);
    groupPrefix.push_back(0);

    // init cuda
    ulli *dev_input;
    ulli *dev_input_data;
    ulli *dev_input_data2;
    ulli *dev_output;
    ulli *dev_output_data;
    ulli *dev_output_data2;

    cudaEvent_t m_start, m_stop;
    float m_time;

    cudaSetDevice(0);

    cudaMalloc( (void**) &(dev_input), sizeof(ulli)*inputSize);
    cudaMalloc( (void**) &(dev_input_data), sizeof(ulli)*inputSize);
    cudaMalloc( (void**) &(dev_input_data2), sizeof(ulli)*inputSize);
    cudaMalloc( (void**) &(dev_output), sizeof(ulli)*inputSize);
    cudaMalloc( (void**) &(dev_output_data), sizeof(ulli)*inputSize);
    cudaMalloc( (void**) &(dev_output_data2), sizeof(ulli)*inputSize);
    cudaMemcpy( dev_input, input, sizeof(ulli)*inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_input_data, input_data, sizeof(ulli)*inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_input_data2, input_data2, sizeof(ulli)*inputSize, cudaMemcpyHostToDevice);
    cudaMemset( dev_output, 0, sizeof(ulli)*inputSize );
    cudaMemset( dev_output_data, 0, sizeof(ulli)*inputSize );
    cudaMemset( dev_output_data2, 0, sizeof(ulli)*inputSize );


    cudaEventCreate( &m_start );
    cudaEventCreate( &m_stop );
    cudaEventRecord( m_start, 0 );

    int pass=0;
    for(;pass<8;pass++){
        if(groupList.size() != 1){
            ulli *tmpDev = dev_input;
            dev_input = dev_output;
            dev_output = tmpDev;

            tmpDev = dev_input_data;
            dev_input_data = dev_output_data;
            dev_output_data = tmpDev;

            tmpDev = dev_input_data2;
            dev_input_data2 = dev_output_data2;
            dev_output_data2 = tmpDev;
        }

        if( groupList.size() < 10 )
            radixGroupGPUWithTwoData(dev_input,dev_input_data,dev_input_data2,
                                     dev_output, dev_output_data, dev_output_data2,
                                     groupList, groupPrefix, pass, Local_Group_Size);
        else
            radixGroupSingleKernelGPUWithTwoData(dev_input,dev_input_data, dev_input_data2,
                                                 dev_output, dev_output_data, dev_output_data2,
                                                 groupList, groupPrefix, pass, Local_Group_Size);

        double averageSize = 0;
        ulli MinSize=groupList[0];
        ulli MaxSize=groupList[0];
        for(int i=0;i<groupList.size();i++){
            averageSize += groupList[i];
            if(MinSize > groupList[i])
                MinSize = groupList[i];
            if(MaxSize < groupList[i])
                MaxSize = groupList[i];
        }

        averageSize = averageSize/groupList.size();
#if 1

        double standardDeviation = 0;
        for(int i=0;i<groupList.size();i++)
            standardDeviation += pow(groupList[i] - averageSize, 2);
        standardDeviation = sqrt(standardDeviation / 10);

        std::cout<<"Number of passed "<<pass<<" Total Group: "<<groupList.size()<<" averageSize: "<<averageSize<<std::endl;
        std::cout<<"StandardDeviation: "<<standardDeviation<<" Max: "<<MaxSize<<" Min: "<<MinSize <<std::endl;
#endif


        if(MaxSize <= Local_Group_Size) {
            break;
        }
    }

    cudaDeviceSynchronize();


    cudaDeviceSynchronize();
    cudaEventRecord( m_stop, 0 );
    cudaEventSynchronize( m_stop );
    cudaEventElapsedTime( &m_time, m_start, m_stop);
    cudaEventDestroy( m_start);
    cudaEventDestroy( m_stop);

    cudaMemcpy( host_output, dev_output, sizeof(ulli)*inputSize, cudaMemcpyDeviceToHost);
    cudaMemcpy( host_output_data, dev_output_data, sizeof(ulli)*inputSize, cudaMemcpyDeviceToHost);
    cudaMemcpy( host_output_data2, dev_output_data2, sizeof(ulli)*inputSize, cudaMemcpyDeviceToHost);

    cudaFree(dev_input);
    cudaFree(dev_input_data);
    cudaFree(dev_input_data2);
    cudaFree(dev_output);
    cudaFree(dev_output_data);
    cudaFree(dev_output_data2);

    //for(int i=0;i<groupList[0];i++){
    //    std::cout<<"Debug:"<<host_output[i]<<"\n";
    //}

#if 0
    for(int i=0;i<groupList[0];i++){
        std::cout<<"Debug:"<<host_output[i]<<"\n";
    }

    for(int i=0;i<groupList[1];i++){
        std::cout<<"Debug2:"<<host_output[groupList[0]+i]<<"\n";
    }
#endif

#if 0
    std::cout<<"Testing Result"<<std::endl;
    for(ulli i =0;i<groupList.size()-1;i++) {
        #pragma omp parallel for
        for(ulli j=groupPrefix[i];j<groupPrefix[i+1];j++) {
            for(ulli k=groupPrefix[i+1];k<inputSize;k++) {
                if(host_output[j]>host_output[k]) {
                    std::cout<<"Negitive: ";
                    std::cout<<host_output[j]<<" Belong to group "<<i << " at "<< j <<" greater than "<<host_output[k] << "At "<<k<<" End at:" <<groupPrefix[i+1]<<std::endl;
                    break;
                }
            }
        }
        if(i%100 ==0)
            std::cout<<"Group Pass:"<<i<<std::endl;
    }
#endif
    //std::cout<<"Number of group = "<<groupList.size()<<std::endl;
    //ulli checkNum=0;
    //for(int i=0;i<groupList.size();i++){
        //if(test[i]<LOCAL_BLOCK_SIZE/2)
        //    std::cout<<"Block Number "<<i<<" less than "<< LOCAL_BLOCK_SIZE/2 <<" which is "<< test[i] <<std::endl;
        //checkNum+=groupList[i];
    //}
    //std::cout<<"Total Output:"<<checkNum<<std::endl;

    //std::cout<<"Running time:"<<m_time/1000<<"Sec."<<std::endl;
    std::cout<<m_time/1000<<std::endl;

    return;
}

void genGroupingWithOnedata(ulli *input, ulli *input_data, ulli inputSize,
                            ulli *&output, ulli* &output_data,
                            std::vector<ulli> &groupList, std::vector<ulli> &groupPrefix,
                            ulli Local_Group_Size) {

    ulli *host_output = new ulli[inputSize];
    ulli *host_output_data = new ulli[inputSize];
    output = host_output;
    output_data = host_output_data;
    groupList.push_back(inputSize);
    groupPrefix.push_back(0);

    // init cuda
    ulli *dev_input;
    ulli *dev_input_data;
    ulli *dev_output;
    ulli *dev_output_data;

    cudaEvent_t m_start, m_stop;
    float m_time;

    cudaSetDevice(0);

    cudaMalloc( (void**) &(dev_input), sizeof(ulli)*inputSize);
    cudaMalloc( (void**) &(dev_input_data), sizeof(ulli)*inputSize);
    cudaMalloc( (void**) &(dev_output), sizeof(ulli)*inputSize);
    cudaMalloc( (void**) &(dev_output_data), sizeof(ulli)*inputSize);
    cudaMemcpy( dev_input, input, sizeof(ulli)*inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy( dev_input_data, input_data, sizeof(ulli)*inputSize, cudaMemcpyHostToDevice);
    cudaMemset( dev_output, 0, sizeof(ulli)*inputSize );
    cudaMemset( dev_output_data, 0, sizeof(ulli)*inputSize );


    cudaEventCreate( &m_start );
    cudaEventCreate( &m_stop );
    cudaEventRecord( m_start, 0 );

    int pass=0;
    for(;pass<8;pass++){
        if(groupList.size() != 1){
            ulli *tmpDev = dev_input;
            dev_input = dev_output;
            dev_output = tmpDev;

            tmpDev = dev_input_data;
            dev_input_data = dev_output_data;
            dev_output_data = tmpDev;
        }

        if( groupList.size() < 10 )
            radixGroupGPUWithOneData(dev_input,dev_input_data, dev_output, dev_output_data, groupList, groupPrefix, pass, Local_Group_Size);
        else
            radixGroupSingleKernelGPUWithOneData(dev_input,dev_input_data, dev_output, dev_output_data, groupList, groupPrefix, pass, Local_Group_Size);

        double averageSize = 0;
        ulli MinSize=groupList[0];
        ulli MaxSize=groupList[0];
        for(int i=0;i<groupList.size();i++){
            averageSize += groupList[i];
            if(MinSize > groupList[i])
                MinSize = groupList[i];
            if(MaxSize < groupList[i])
                MaxSize = groupList[i];
        }

        averageSize = averageSize/groupList.size();
#if 1

        double standardDeviation = 0;
        for(int i=0;i<groupList.size();i++)
            standardDeviation += pow(groupList[i] - averageSize, 2);
        standardDeviation = sqrt(standardDeviation / 10);

        std::cout<<"Number of passed "<<pass<<" Total Group: "<<groupList.size()<<" averageSize: "<<averageSize<<std::endl;
        std::cout<<"StandardDeviation: "<<standardDeviation<<" Max: "<<MaxSize<<" Min: "<<MinSize <<std::endl;
#endif


        if(MaxSize <= Local_Group_Size) {
            break;
        }
    }

    cudaDeviceSynchronize();


    cudaDeviceSynchronize();
    cudaEventRecord( m_stop, 0 );
    cudaEventSynchronize( m_stop );
    cudaEventElapsedTime( &m_time, m_start, m_stop);
    cudaEventDestroy( m_start);
    cudaEventDestroy( m_stop);

    cudaMemcpy( host_output, dev_output, sizeof(ulli)*inputSize, cudaMemcpyDeviceToHost);
    cudaMemcpy( host_output_data, dev_output_data, sizeof(ulli)*inputSize, cudaMemcpyDeviceToHost);
    cudaFree(dev_input);
    cudaFree(dev_input_data);
    cudaFree(dev_output);
    cudaFree(dev_output_data);

    //for(int i=0;i<groupList[0];i++){
    //    std::cout<<"Debug:"<<host_output[i]<<"\n";
    //}

#if 0
    for(int i=0;i<groupList[0];i++){
        std::cout<<"Debug:"<<host_output[i]<<"\n";
    }

    for(int i=0;i<groupList[1];i++){
        std::cout<<"Debug2:"<<host_output[groupList[0]+i]<<"\n";
    }
#endif

#if 0
    std::cout<<"Testing Result"<<std::endl;
    for(ulli i =0;i<groupList.size()-1;i++) {
        #pragma omp parallel for
        for(ulli j=groupPrefix[i];j<groupPrefix[i+1];j++) {
            for(ulli k=groupPrefix[i+1];k<inputSize;k++) {
                if(host_output[j]>host_output[k]) {
                    std::cout<<"Negitive: ";
                    std::cout<<host_output[j]<<" Belong to group "<<i << " at "<< j <<" greater than "<<host_output[k] << "At "<<k<<" End at:" <<groupPrefix[i+1]<<std::endl;
                    break;
                }
            }
        }
        if(i%100 ==0)
            std::cout<<"Group Pass:"<<i<<std::endl;
    }
#endif
    //std::cout<<"Number of group = "<<groupList.size()<<std::endl;
    //ulli checkNum=0;
    //for(int i=0;i<groupList.size();i++){
        //if(test[i]<LOCAL_BLOCK_SIZE/2)
        //    std::cout<<"Block Number "<<i<<" less than "<< LOCAL_BLOCK_SIZE/2 <<" which is "<< test[i] <<std::endl;
        //checkNum+=groupList[i];
    //}
    //std::cout<<"Total Output:"<<checkNum<<std::endl;

    //std::cout<<"Running time:"<<m_time/1000<<"Sec."<<std::endl;
    std::cout<<m_time/1000<<std::endl;

    return;
}

void* genGrouping(ulli *input, ulli inputSize, std::vector<ulli> &groupList, std::vector<ulli> &groupPrefix, ulli Local_Group_Size) {

    ulli *host_output = new ulli[inputSize];
    groupList.push_back(inputSize);
    groupPrefix.push_back(0);

    // init cuda
    ulli *dev_input;
    ulli *dev_output;

    cudaEvent_t m_start, m_stop;
    float m_time;

    cudaSetDevice(0);


    cudaMalloc( (void**) &(dev_input), sizeof(ulli)*inputSize);
    cudaMalloc( (void**) &(dev_output), sizeof(ulli)*inputSize);
    {
        cudaEvent_t m_startio, m_stopio;
        float m_timeio;

        cudaEventCreate( &m_startio );
        cudaEventCreate( &m_stopio );
        cudaEventRecord( m_startio, 0 );

        cudaMemcpy( dev_input, input, sizeof(ulli)*inputSize, cudaMemcpyHostToDevice);

        cudaEventRecord( m_stopio, 0 );
        cudaEventSynchronize( m_stopio );
        cudaEventElapsedTime( &m_timeio, m_startio, m_stopio);
        cudaEventDestroy( m_startio);
        cudaEventDestroy( m_stopio);

        std::cout<<"I/O in time:"<<m_timeio/1000<<" Sec."<<std::endl;
    }
    cudaMemset( dev_output, 0, sizeof(ulli)*inputSize );


    cudaEventCreate( &m_start );
    cudaEventCreate( &m_stop );
    cudaEventRecord( m_start, 0 );

    ulli *maxValueA;

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

    int MaxBitA = 0;
    while(*maxValueA) {
        *maxValueA = (*maxValueA)>>1;
        MaxBitA++;
    }

    int StartPassA = 8-ceil((float)MaxBitA/(float)BIT_PASS);

    int pass=StartPassA;
    for(;pass<8;pass++){
        if(groupList.size() != 1){
            ulli *tmpDev = dev_input;
            dev_input = dev_output;
            dev_output = tmpDev;
        }

        if( groupList.size() < 10 )
            radixGroupGPU(dev_input, dev_output, groupList, groupPrefix, pass, Local_Group_Size);
        else
            radixGroupSingleKernelGPU(dev_input, dev_output, groupList, groupPrefix, pass, Local_Group_Size);

        double averageSize = 0;
        ulli MinSize=groupList[0];
        ulli MaxSize=groupList[0];
        ulli groupDoneCount=0;
        for(int i=0;i<groupList.size();i++){
            if(Local_Group_Size>=groupList[i]) {
                groupDoneCount++;
            }
            averageSize += groupList[i];
            if(MinSize > groupList[i])
                MinSize = groupList[i];
            if(MaxSize < groupList[i])
                MaxSize = groupList[i];
        }

        averageSize = averageSize/groupList.size();
#if 1

        double standardDeviation = 0;
        for(int i=0;i<groupList.size();i++)
            standardDeviation += pow(groupList[i] - averageSize, 2);
        standardDeviation = sqrt(standardDeviation / 10);

        std::cout<<"Number of passed "<<pass<<" Total Group: "<<groupList.size()<<" averageSize: "<<averageSize<<std::endl;
        std::cout<<"StandardDeviation: "<<standardDeviation<<" Max: "<<MaxSize<<" Min: "<<MinSize <<std::endl;
        std::cout<<"Number of Group completed: "<<groupDoneCount<<""<<std::endl;
#endif


        if(MaxSize <= Local_Group_Size) {
            break;
        }
    }

    cudaDeviceSynchronize();


    cudaDeviceSynchronize();
    cudaEventRecord( m_stop, 0 );
    cudaEventSynchronize( m_stop );
    cudaEventElapsedTime( &m_time, m_start, m_stop);
    cudaEventDestroy( m_start);
    cudaEventDestroy( m_stop);

    cudaMemcpy( host_output, dev_output, sizeof(ulli)*inputSize, cudaMemcpyDeviceToHost);

    cudaFree(dev_input);

    //for(int i=0;i<groupList[0];i++){
    //    std::cout<<"Debug:"<<host_output[i]<<"\n";
    //}

#if 0
    for(int i=0;i<groupList[0];i++){
        std::cout<<"Debug:"<<host_output[i]<<"\n";
    }

    for(int i=0;i<groupList[1];i++){
        std::cout<<"Debug2:"<<host_output[groupList[0]+i]<<"\n";
    }
#endif

#if 0
    std::cout<<"Testing Result"<<std::endl;
    for(ulli i =0;i<groupList.size()-1;i++) {
        #pragma omp parallel for
        for(ulli j=groupPrefix[i];j<groupPrefix[i+1];j++) {
            for(ulli k=groupPrefix[i+1];k<inputSize;k++) {
                if(host_output[j]>host_output[k]) {
                    std::cout<<"Negitive: ";
                    std::cout<<host_output[j]<<" Belong to group "<<i << " at "<< j <<" greater than "<<host_output[k] << "At "<<k<<" End at:" <<groupPrefix[i+1]<<std::endl;
                    break;
                }
            }
        }
        if(i%100 ==0)
            std::cout<<"Group Pass:"<<i<<std::endl;
    }
#endif
    //std::cout<<"Number of group = "<<groupList.size()<<std::endl;
    //ulli checkNum=0;
    //for(int i=0;i<groupList.size();i++){
        //if(test[i]<LOCAL_BLOCK_SIZE/2)
        //    std::cout<<"Block Number "<<i<<" less than "<< LOCAL_BLOCK_SIZE/2 <<" which is "<< test[i] <<std::endl;
        //checkNum+=groupList[i];
    //}
    //std::cout<<"Total Output:"<<checkNum<<std::endl;

    //std::cout<<"Running time:"<<m_time/1000<<"Sec."<<std::endl;
    std::cout<<m_time/1000<<std::endl;

    return dev_output;
}




inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
