#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <cstring>

#include "utils.h"
#include "BPTree.h"
#include "grouping.h"

#define BLOCK_SIZE 1024

#define CUDA_MAX_BLOCK 65535

// #define STM_NUM 84 //70


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);


//Searching Only one step
//Naive
__global__
void kernelSearchingNaiveOneStep( const ulli *dev_Btree_root, BTPage *dev_globalPointer,
                                  const ulli *dev_search_key, ulli dev_search_key_size ,
                                  ulli *dev_output) {
    ulli h_pos;

    h_pos = blockIdx.x * blockDim.x + threadIdx.x;

    while( h_pos < dev_search_key_size ) {

        ulli currentPtr = dev_search_key[h_pos];
        BTPage *searchPage = &dev_globalPointer[*dev_Btree_root];
        bool stop = false;
        ulli searchingLeaf = 0;
        while(!stop){
            ulli foundAt = INVALID_DATA;
            int start = 0;
            int end = searchPage->size-1;
            while(start <= end) {
                int middle = start + (end-start)/2;
                if(currentPtr == searchPage->key[middle]) {
                    foundAt = middle;
                    break;
                }
                else if(currentPtr < searchPage->key[middle]) {
                    end = middle - 1;
                }
                else {
                    start = middle + 1;
                }
            }
            if(foundAt == INVALID_DATA)
                foundAt = start;
            if(foundAt == searchPage->size)
                foundAt = start-1;

            if(dev_globalPointer[searchPage->data[foundAt]].leafFlag == true) {
                searchingLeaf = searchPage->data[foundAt];
                stop = true;
            }
            else {
                searchPage = &dev_globalPointer[searchPage->data[foundAt]];
            }
        }

        //leaf search
        {
            if(searchingLeaf > PAGE_ALLOCATION){
                printf("Something Worng!!!\n");
                dev_output[h_pos] = INVALID_DATA;
            }
            else {
                BTPage *searchPageLeaf = &dev_globalPointer[searchingLeaf];
                ulli foundAt = INVALID_DATA;
                int start = 0;
                int end = searchPageLeaf->size-1;
                while(start <= end) {
                    int middle = start + (end-start)/2;
                    if(currentPtr == searchPageLeaf->key[middle]) {
                        foundAt = middle;
                        break;
                    }
                    else if(currentPtr < searchPageLeaf->key[middle]) {
                        end = middle - 1;
                    }
                    else {
                        start = middle + 1;
                    }
                }
                if(foundAt == INVALID_DATA) {
                    dev_output[h_pos] = INVALID_DATA;
                }
                else {
                    dev_output[h_pos] = searchPageLeaf->data[foundAt];
                }
            }
        }

        h_pos += gridDim.x * blockDim.x;
    }
}

//searching state 3
//Naive two states
__global__
void kernelSearchingKeyGroupStateThreeWithoutGrouping(  const ulli *dev_search_key, ulli dev_key_Size,
                                                        const ulli *dev_search_leaf /*output from step2*/,
                                                        ulli *dev_output, BTPage *dev_globalPointer) {
    ulli h_pos = blockIdx.x * blockDim.x + threadIdx.x;

    while(h_pos < dev_key_Size) {

        ulli currentPtr = dev_search_key[h_pos];
        ulli searchingLeaf = dev_search_leaf[h_pos];

        if(searchingLeaf > PAGE_ALLOCATION){
                printf("Something Worng!!!\n");
                dev_output[h_pos] = INVALID_DATA;
        }
        else {
            BTPage *searchPage = &dev_globalPointer[searchingLeaf];
            ulli foundAt = INVALID_DATA;
            int start = 0;
            int end = searchPage->size-1;
            while(start <= end) {
                int middle = start + (end-start)/2;
                if(currentPtr == searchPage->key[middle]) {
                    foundAt = middle;
                    break;
                }
                else if(currentPtr < searchPage->key[middle]) {
                    end = middle - 1;
                }
                else {
                    start = middle + 1;
                }
            }
            if(foundAt == INVALID_DATA) {
                dev_output[h_pos] = INVALID_DATA;
            }
            else {
                dev_output[h_pos] = searchPage->data[foundAt];
            }
        }

        h_pos += gridDim.x * blockDim.x;
    }
}

//Searching state 2
//Naive two states
__global__
void kernelSearchingKeyGroupStateTwoWithOutGrouping( const ulli *dev_Btree_root, BTPage *dev_globalPointer,
                                                     const ulli *dev_search_key, ulli dev_search_key_size ,
                                                     ulli *dev_output) {
    ulli h_pos;

    h_pos = blockIdx.x * blockDim.x + threadIdx.x;

    while( h_pos < dev_search_key_size ) {

        ulli currentPtr = dev_search_key[h_pos];
        BTPage *searchPage = &dev_globalPointer[*dev_Btree_root];
        bool stop = false;
        while(!stop){
            ulli foundAt = INVALID_DATA;
            int start = 0;
            int end = searchPage->size-1;
            while(start <= end) {
                int middle = start + (end-start)/2;
                if(currentPtr == searchPage->key[middle]) {
                    foundAt = middle;
                    break;
                }
                else if(currentPtr < searchPage->key[middle]) {
                    end = middle - 1;
                }
                else {
                    start = middle + 1;
                }
            }
            if(foundAt == INVALID_DATA)
                foundAt = start;
            if(foundAt == searchPage->size)
                foundAt = start-1;

            if(dev_globalPointer[searchPage->data[foundAt]].leafFlag == true) {
                dev_output[h_pos] = searchPage->data[foundAt];
                stop = true;
            }
            else {
                searchPage = &dev_globalPointer[searchPage->data[foundAt]];
            }
        }

        h_pos += gridDim.x * blockDim.x;
    }
}

// grouping with 2 step
//searching combine step 2 and 3 with grouping
__global__
void kernelSearchingKeyGroupTwoStep( const ulli *dev_Btree_root, BTPage *dev_globalPointer,
                                     const ulli *dev_search_key, ulli *dev_key_prefix,
                                     ulli *dev_key_group_cout, ulli group_key_num,
                                     ulli *dev_output) {
    ulli currentGroup = blockIdx.x;
    ulli h_pos;

    while(currentGroup < group_key_num){

        const ulli* __restrict dev_input_key = &dev_search_key[dev_key_prefix[currentGroup]];

        h_pos = threadIdx.x;

        while( h_pos < dev_key_group_cout[currentGroup] ) {

            ulli currentPtr = dev_input_key[h_pos];
            BTPage *searchPage = &dev_globalPointer[*dev_Btree_root];
            ulli leafNodeSearch = 0;
            bool stop = false;
            while(!stop){
                ulli foundAt = INVALID_DATA;
                int start = 0;
                int end = searchPage->size-1;
                while(start <= end) {
                    int middle = start + (end-start)/2;
                    if(currentPtr == searchPage->key[middle]) {
                        foundAt = middle;
                        break;
                    }
                    else if(currentPtr < searchPage->key[middle]) {
                        end = middle - 1;
                    }
                    else {
                        start = middle + 1;
                    }
                }
                if(foundAt == INVALID_DATA)
                    foundAt = start;
                if(foundAt == searchPage->size)
                    foundAt = start-1;

                if(dev_globalPointer[searchPage->data[foundAt]].leafFlag == true) {
                    leafNodeSearch = searchPage->data[foundAt];
                    stop = true;
                }
                else {
                    searchPage = &dev_globalPointer[searchPage->data[foundAt]];
                }
            }

            {
                if(leafNodeSearch > PAGE_ALLOCATION){
                    printf("Something Worng!!!\n");
                    dev_output[dev_key_prefix[currentGroup]+ h_pos] = INVALID_DATA;
                }
                else {
                    searchPage = &dev_globalPointer[leafNodeSearch];
                    ulli foundAt = INVALID_DATA;
                    int start = 0;
                    int end = searchPage->size-1;
                    while(start <= end) {
                        int middle = start + (end-start)/2;
                        if(currentPtr == searchPage->data[middle]) {
                            foundAt = middle;
                            break;
                        }
                        else if(currentPtr < searchPage->data[middle]) {
                            end = middle - 1;
                        }
                        else {
                            start = middle + 1;
                        }
                    }

                    if(foundAt == INVALID_DATA) {
                        dev_output[dev_key_prefix[currentGroup]+ h_pos] = INVALID_DATA;
                    }
                    else {
                        dev_output[dev_key_prefix[currentGroup]+ h_pos] = searchPage->data[foundAt];
                    }
                }
            }

            h_pos += blockDim.x;
        }
        currentGroup += gridDim.x;
    }
}

//searching state 3
__global__ void kernelSearchingKeyGroupStateThree(  const ulli *dev_search_key, ulli *dev_key_prefix,
                                                    ulli *dev_key_group_cout, ulli group_key_num,
                                                    const ulli *dev_search_leaf /*output from step2*/,
                                                    const ulli *dev_Btree_root, BTPage *dev_globalPointer,
                                                    ulli *dev_output) {
    ulli currentGroup = blockIdx.x;
    ulli h_pos;

    while(currentGroup < group_key_num){
        const ulli* __restrict dev_input = &dev_search_key[dev_key_prefix[currentGroup]];

        h_pos = threadIdx.x;

        while( h_pos < dev_key_group_cout[currentGroup] ) {

            ulli currentPtr = dev_input[h_pos];
            if(dev_search_leaf[dev_key_prefix[currentGroup]+h_pos] > PAGE_ALLOCATION){
                printf("Something Worng!!!\n");
                dev_output[dev_key_prefix[currentGroup]+ h_pos] = INVALID_DATA;
            }
            else {
                BTPage *searchPage = &dev_globalPointer[dev_search_leaf[dev_key_prefix[currentGroup]+h_pos]];
                ulli foundAt = INVALID_DATA;
                int start = 0;
                int end = searchPage->size-1;
                while(start <= end) {
                    int middle = start + (end-start)/2;
                    if(currentPtr == searchPage->key[middle]) {
                        foundAt = middle;
                        break;
                    }
                    else if(currentPtr < searchPage->key[middle]) {
                        end = middle - 1;
                    }
                    else {
                        start = middle + 1;
                    }
                }

                if(foundAt == INVALID_DATA) {
                    dev_output[dev_key_prefix[currentGroup]+ h_pos] = INVALID_DATA;
                }
                else {
                    dev_output[dev_key_prefix[currentGroup]+ h_pos] = searchPage->data[foundAt];
                }
            }

            h_pos += blockDim.x;
        }
        currentGroup += gridDim.x;
    }
}

//Searching state 2
__global__
void kernelSearchingKeyGroupStateTwo( const ulli *dev_Btree_root, BTPage *dev_globalPointer,
                                      const ulli *dev_search_key, ulli *dev_key_prefix,
                                      ulli *dev_key_group_cout, ulli group_key_num,
                                      ulli *dev_output) {
    ulli currentGroup = blockIdx.x;
    ulli h_pos;

    while(currentGroup < group_key_num){
        const ulli* __restrict dev_input_key = &dev_search_key[dev_key_prefix[currentGroup]];

        h_pos = threadIdx.x;

        while( h_pos < dev_key_group_cout[currentGroup] ) {

            ulli currentPtr = dev_input_key[h_pos];
            BTPage *searchPage = &dev_globalPointer[*dev_Btree_root];
            bool stop = false;
            while(!stop){
                ulli foundAt = INVALID_DATA;
                int start = 0;
                int end = searchPage->size-1;
                while(start <= end) {
                    int middle = start + (end-start)/2;
                    if(currentPtr == searchPage->key[middle]) {
                        foundAt = middle;
                        break;
                    }
                    else if(currentPtr < searchPage->key[middle]) {
                        end = middle - 1;
                    }
                    else {
                        start = middle + 1;
                    }
                }
                if(currentPtr < searchPage->key[start])
                    foundAt = start;

                if(dev_globalPointer[searchPage->data[foundAt]].leafFlag == true) {
                    dev_output[dev_key_prefix[currentGroup]+ h_pos] = searchPage->data[foundAt];
                    stop = true;
                }
                else {
                    searchPage = &dev_globalPointer[searchPage->data[foundAt]];
                }
            }

            h_pos += blockDim.x;
        }
        currentGroup += gridDim.x;
    }
}




//Searching Only one step
//Naive
void* SearchingBTreeGPUOnlyOneState(ulli *input_key, ulli inputSize,
                                    ulli *dev_Btree, BTPage *dev_globalPointer) {

    std::cout<<"SearchingBTreeGPU Naive One state Searching"<<std::endl;

    ulli *dev_input_key;
    ulli *dev_result, *host_result;

    {
        cudaEvent_t m_start, m_stop;
        float m_time;

        host_result = new ulli[inputSize];

        std::memset(host_result, 0, sizeof(ulli)*inputSize);

        cudaEventCreate( &m_start );
        cudaEventCreate( &m_stop );
        cudaEventRecord( m_start, 0 );

        cudaSetDevice(0);

        cudaMalloc( (void**) &(dev_result), sizeof(ulli)*inputSize);
        cudaMalloc( (void**) &(dev_input_key), sizeof(ulli)*inputSize );

        cudaMemset( dev_result, 0, sizeof(ulli)*inputSize );
        cudaMemcpy( dev_input_key, input_key, sizeof(ulli)*inputSize, cudaMemcpyHostToDevice);

        ulli m_blocks = floor(inputSize/BLOCK_SIZE)>CUDA_MAX_BLOCK ? CUDA_MAX_BLOCK:floor(inputSize/BLOCK_SIZE);
        dim3 grid(m_blocks, 1, 1);
        dim3 threads(BLOCK_SIZE, 1, 1);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );
        kernelSearchingNaiveOneStep<<<grid,threads>>>( dev_Btree, dev_globalPointer,
                                                       dev_input_key, inputSize,
                                                       dev_result);


        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        cudaDeviceSynchronize();

        cudaDeviceSynchronize();
        cudaEventRecord( m_stop, 0 );
        cudaEventSynchronize( m_stop );
        cudaEventElapsedTime( &m_time, m_start, m_stop);
        cudaEventDestroy( m_start);
        cudaEventDestroy( m_stop);

        std::cout<<"Naive Running time Searching ="<<m_time<<std::endl;

        cudaMemcpy( host_result, dev_result, sizeof(ulli)*inputSize, cudaMemcpyDeviceToHost);
        cudaFree(dev_result);
        cudaFree(dev_input_key);

        // Debug
        {
            ulli counter=0;
            for(ulli i=0;i<inputSize;i++) {
                if(host_result[i]!=input_key[i] && host_result[i]!= INVALID_DATA) {
                    std::cout<<"Something wrong with searching!!!!"<<std::endl;
                    std::cout<<host_result[i]<<"!="<<input_key[i]<<std::endl;
                    break;
                }
                if(host_result[i]== INVALID_DATA) {
                    counter++;
                }
            }
            std::cout<<"Search step checking complete"<<std::endl;
            std::cout<<"Found:"<<(1-((float)counter)/((float)inputSize))*100<<"% of key"<<std::endl;
        }
    }

    return 0;
}

void* SearchingBTreeGPUNaive(ulli *input_key, ulli inputSize, ulli *dev_Btree, BTPage *dev_globalPointer) {

    std::cout<<"SearchingBTreeGPU Naive"<<std::endl;

    //State2
    ulli *dev_input_key;
    ulli *dev_result, *host_result;

    {
        cudaEvent_t m_start, m_stop;
        float m_time;

        host_result = new ulli[inputSize];

        std::memset(host_result, 0, sizeof(ulli)*inputSize);

        cudaEventCreate( &m_start );
        cudaEventCreate( &m_stop );
        cudaEventRecord( m_start, 0 );

        cudaSetDevice(0);

        cudaMalloc( (void**) &(dev_result), sizeof(ulli)*inputSize);
        cudaMalloc( (void**) &(dev_input_key), sizeof(ulli)*inputSize );

        cudaMemset( dev_result, 0, sizeof(ulli)*inputSize );
        cudaMemcpy( dev_input_key, input_key, sizeof(ulli)*inputSize, cudaMemcpyHostToDevice);

        ulli m_blocks = floor(inputSize/BLOCK_SIZE)>CUDA_MAX_BLOCK ? CUDA_MAX_BLOCK:floor(inputSize/BLOCK_SIZE);
        dim3 grid(m_blocks, 1, 1);
        dim3 threads(BLOCK_SIZE, 1, 1);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );
        kernelSearchingKeyGroupStateTwoWithOutGrouping<<<grid,threads>>>( dev_Btree, dev_globalPointer, dev_input_key,
                                                                          inputSize,dev_result);


        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        cudaDeviceSynchronize();

        cudaDeviceSynchronize();
        cudaEventRecord( m_stop, 0 );
        cudaEventSynchronize( m_stop );
        cudaEventElapsedTime( &m_time, m_start, m_stop);
        cudaEventDestroy( m_start);
        cudaEventDestroy( m_stop);

        std::cout<<"Running time Searching State2= "<<m_time<<std::endl;
    }

    //State3
    ulli *dev_search_result;
    ulli *host_search_result;
    host_search_result = new ulli[inputSize];

    {
        cudaEvent_t m_start, m_stop;
        float m_time;

        cudaEventCreate( &m_start );
        cudaEventCreate( &m_stop );
        cudaEventRecord( m_start, 0 );

        cudaMalloc( (void**) &(dev_search_result), sizeof(ulli)*inputSize);
        cudaMemset( dev_search_result, 0, sizeof(ulli)*inputSize );

        ulli m_blocks = floor(inputSize/BLOCK_SIZE)>CUDA_MAX_BLOCK ? CUDA_MAX_BLOCK:floor(inputSize/BLOCK_SIZE);
        dim3 grid(m_blocks, 1, 1);
        dim3 threads(BLOCK_SIZE, 1, 1);
        kernelSearchingKeyGroupStateThreeWithoutGrouping<<<grid,threads>>>(  dev_input_key, inputSize,
                                                                             dev_result /*output from step2*/,
                                                                             dev_search_result,dev_globalPointer);


        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        cudaDeviceSynchronize();

        cudaEventRecord( m_stop, 0 );
        cudaEventSynchronize( m_stop );
        cudaEventElapsedTime( &m_time, m_start, m_stop);
        cudaEventDestroy( m_start);
        cudaEventDestroy( m_stop);
        std::cout<<"Running time Searching State3= "<<m_time <<std::endl;
    }

    cudaMemcpy( host_search_result, dev_search_result, sizeof(ulli)*inputSize, cudaMemcpyDeviceToHost);
    cudaFree(dev_search_result);
    cudaFree(dev_result);
    cudaFree(dev_input_key);

    // Debug
    {
        ulli counter=0;
        for(ulli i=0;i<inputSize;i++) {
            if(host_search_result[i]!=input_key[i] && host_search_result[i]!= INVALID_DATA) {
                std::cout<<"Something wrong with searching!!!!"<<std::endl;
                std::cout<<host_search_result[i]<<"!="<<input_key[i]<<std::endl;
                break;
            }
            if(host_search_result[i]== INVALID_DATA) {
                counter++;
            }
        }
        std::cout<<"Search step checking complete"<<std::endl;
        std::cout<<"Found:"<<(1-((float)counter)/((float)inputSize))*100<<"% of key"<<std::endl;
    }

    return host_search_result;
}

// grouping idea with 2 state
//combineStep
void* SearchingBTreeGPUCombineStep(ulli *input_key, ulli inputSize, ulli *dev_Btree,
                                   BTPage *dev_globalPointer) {

    std::cout<<"SearchingBTreeGPU Combine Step"<<std::endl;
    //Grouping
    std::vector<ulli> groupList;
    std::vector<ulli> groupPrefix;
    ulli *dev_output = (ulli*)genGrouping(input_key,inputSize,groupList,groupPrefix,GROUP_SEARCH_SIZE);

    gpuErrchk( cudaPeekAtLastError() );

    cudaMemcpy( input_key, dev_output, sizeof(ulli)*inputSize, cudaMemcpyDeviceToHost);

    gpuErrchk( cudaPeekAtLastError() );

    //State2

    ulli *dev_group_cout, *host_group_count;
    ulli *dev_prefix, *host_prefix;
    ulli *dev_result, *host_result;

    {
        cudaEvent_t m_start, m_stop;
        float m_time;

        host_group_count = new ulli[groupList.size()];
        host_prefix = new ulli[groupList.size()];

        host_result = new ulli[inputSize];

        std::memset(host_result, 0, sizeof(ulli)*inputSize);

        for(ulli i=0;i<groupList.size();i++) {
            host_group_count[i] = groupList[i];
            host_prefix[i] = groupPrefix[i];
        }

        cudaEventCreate( &m_start );
        cudaEventCreate( &m_stop );
        cudaEventRecord( m_start, 0 );

        cudaSetDevice(0);

        cudaMalloc( (void**) &(dev_group_cout), sizeof(ulli)*groupList.size());

        gpuErrchk( cudaPeekAtLastError() );

        cudaMalloc( (void**) &(dev_prefix), sizeof(ulli)*groupList.size());

        gpuErrchk( cudaPeekAtLastError() );

        cudaMalloc( (void**) &(dev_result), sizeof(ulli)*inputSize);

        gpuErrchk( cudaPeekAtLastError() );

        cudaMemcpy( dev_group_cout, host_group_count, sizeof(ulli)*groupList.size(), cudaMemcpyHostToDevice);

        gpuErrchk( cudaPeekAtLastError() );

        cudaMemcpy( dev_prefix, host_prefix, sizeof(ulli)*groupList.size(), cudaMemcpyHostToDevice);

        gpuErrchk( cudaPeekAtLastError() );

        cudaMemset( dev_result, 0, sizeof(ulli)*inputSize );

        gpuErrchk( cudaPeekAtLastError() );
  
        ulli m_blocks = groupList.size()>CUDA_MAX_BLOCK ? CUDA_MAX_BLOCK:groupList.size();
        dim3 grid(m_blocks, 1, 1);
        dim3 threads(BLOCK_SIZE, 1, 1);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );
        kernelSearchingKeyGroupTwoStep<<<grid,threads>>>( dev_Btree, dev_globalPointer,
                                                          dev_output, dev_prefix,
                                                          dev_group_cout, groupList.size(),
                                                          dev_result);
                                                
        // kernelSearchingNaiveOneStep<<<grid,threads>>>( dev_Btree, dev_globalPointer,
        // dev_output, inputSize,
        // dev_result);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        cudaDeviceSynchronize();

        cudaDeviceSynchronize();
        cudaEventRecord( m_stop, 0 );
        cudaEventSynchronize( m_stop );
        cudaEventElapsedTime( &m_time, m_start, m_stop);
        cudaEventDestroy( m_start);
        cudaEventDestroy( m_stop);

        std::cout<<"combi Running time Searching State2:"<<m_time<<std::endl;

        cudaMemcpy( host_result, dev_result, sizeof(ulli)*inputSize, cudaMemcpyDeviceToHost);
        cudaFree(dev_result);
        cudaFree(dev_output);
        cudaFree(dev_group_cout);
        cudaFree(dev_prefix);

        // Debug
        {
            ulli counter = 0;
            for(ulli i=0;i<inputSize;i++) {
                if(host_result[i]!=input_key[i] && host_result[i]!= INVALID_DATA) {
                    std::cout<<host_result[i]<<"!="<<input_key[i]<<std::endl;
                    std::cout<<"Something wrong with searching at i ="<<i<<"!!!!"<<std::endl;
                    break;
                }
                if(host_result[i]== INVALID_DATA) {
                    counter++;
                }
            }
            std::cout<<"Search step checking complete"<<std::endl;
            std::cout<<"Found:"<<(1-((float)counter)/((float)inputSize))*100<<"% of key"<<std::endl;
        }

    }

    return host_result;
}

// void* SearchingBTreeGPU(ulli *input_key, ulli inputSize, ulli *dev_Btree, BTPage *dev_globalPointer) {

//     std::cout<<"SearchingBTreeGPU"<<std::endl;
//     //Grouping
//     std::vector<ulli> groupList;
//     std::vector<ulli> groupPrefix;
//     ulli *dev_output = (ulli*)genGrouping(input_key,inputSize,groupList,groupPrefix,GROUP_SEARCH_SIZE);

//     cudaMemcpy( input_key, dev_output, sizeof(ulli)*inputSize, cudaMemcpyDeviceToHost);

//     //State2

//     ulli *dev_group_cout, *host_group_count;
//     ulli *dev_prefix, *host_prefix;
//     ulli *dev_result, *host_result;

//     {
//         cudaEvent_t m_start, m_stop;
//         float m_time;

//         host_group_count = new ulli[groupList.size()];
//         host_prefix = new ulli[groupList.size()];

//         host_result = new ulli[inputSize];

//         std::memset(host_result, 0, sizeof(ulli)*inputSize);

//         for(ulli i=0;i<groupList.size();i++) {
//             host_group_count[i] = groupList[i];
//             host_prefix[i] = groupPrefix[i];
//         }

//         cudaEventCreate( &m_start );
//         cudaEventCreate( &m_stop );
//         cudaEventRecord( m_start, 0 );

//         cudaSetDevice(0);

//         cudaMalloc( (void**) &(dev_group_cout), sizeof(ulli)*groupList.size());
//         cudaMalloc( (void**) &(dev_prefix), sizeof(ulli)*groupList.size());

//         cudaMalloc( (void**) &(dev_result), sizeof(ulli)*inputSize);



//         cudaMemcpy( dev_group_cout, host_group_count, sizeof(ulli)*groupList.size(), cudaMemcpyHostToDevice);
//         cudaMemcpy( dev_prefix, host_prefix, sizeof(ulli)*groupList.size(), cudaMemcpyHostToDevice);

//         cudaMemset( dev_result, 0, sizeof(ulli)*inputSize );
//         ulli m_blocks = STM_NUM;//groupList.size()>CUDA_MAX_BLOCK ? CUDA_MAX_BLOCK:groupList.size();
//         // ulli m_blocks = groupList.size()>CUDA_MAX_BLOCK ? CUDA_MAX_BLOCK:groupList.size();
//         dim3 grid(m_blocks, 1, 1);
//         dim3 threads(BLOCK_SIZE, 1, 1);
//         gpuErrchk( cudaPeekAtLastError() );
//         gpuErrchk( cudaStreamSynchronize(0) );
//         kernelSearchingKeyGroupStateTwo<<<grid,threads>>>( dev_Btree, dev_globalPointer, dev_output,
//                                                            dev_prefix, dev_group_cout, groupList.size(),dev_result);


//         gpuErrchk( cudaPeekAtLastError() );
//         gpuErrchk( cudaStreamSynchronize(0) );

//         cudaDeviceSynchronize();

//         cudaDeviceSynchronize();
//         cudaEventRecord( m_stop, 0 );
//         cudaEventSynchronize( m_stop );
//         cudaEventElapsedTime( &m_time, m_start, m_stop);
//         cudaEventDestroy( m_start);
//         cudaEventDestroy( m_stop);

//         std::cout<<"Running time Searching State2:"<<m_time/1000<<"Sec."<<std::endl;

//     }

//     //State3
//     ulli *dev_search_result;
//     ulli *host_search_result;
//     host_search_result = new ulli[inputSize];

//     {
//         cudaEvent_t m_start, m_stop;
//         float m_time;

//         cudaEventCreate( &m_start );
//         cudaEventCreate( &m_stop );
//         cudaEventRecord( m_start, 0 );

//         cudaMalloc( (void**) &(dev_search_result), sizeof(ulli)*inputSize);
//         cudaMemset( dev_search_result, 0, sizeof(ulli)*inputSize );
//         ulli m_blocks = STM_NUM;//groupList.size()>CUDA_MAX_BLOCK ? CUDA_MAX_BLOCK:groupList.size();
//         // ulli m_blocks = groupList.size()>CUDA_MAX_BLOCK ? CUDA_MAX_BLOCK:groupList.size();
//         dim3 grid(m_blocks, 1, 1);
//         dim3 threads(BLOCK_SIZE, 1, 1);
//         kernelSearchingKeyGroupStateThree<<<grid,threads>>>(  dev_output, dev_prefix,
//                                                             dev_group_cout, groupList.size(),
//                                                             dev_result /*output from step2*/,
//                                                             dev_Btree, dev_globalPointer,
//                                                             dev_search_result);

//         gpuErrchk( cudaPeekAtLastError() );
//         gpuErrchk( cudaStreamSynchronize(0) );

//         cudaDeviceSynchronize();

//         cudaEventRecord( m_stop, 0 );
//         cudaEventSynchronize( m_stop );
//         cudaEventElapsedTime( &m_time, m_start, m_stop);
//         cudaEventDestroy( m_start);
//         cudaEventDestroy( m_stop);
//         std::cout<<"Running time Searching State3:"<<m_time/1000<<"Sec."<<std::endl;
//     }

//     cudaMemcpy( host_search_result, dev_search_result, sizeof(ulli)*inputSize, cudaMemcpyDeviceToHost);
//     cudaFree(dev_search_result);
//     cudaFree(dev_result);
//     cudaFree(dev_output);

//     // Debug
//     {
//         ulli counter=0;
//         for(ulli i=0;i<inputSize;i++) {
//             if(host_search_result[i]!=input_key[i] && host_search_result[i]!= INVALID_DATA) {
//                 std::cout<<host_search_result[i]<<"!="<<input_key[i]<<std::endl;
//                 std::cout<<"Something wrong with searching at i ="<<i<<"!!!!"<<std::endl;
//                 break;
//             }
//             if(host_search_result[i]== INVALID_DATA) {
//                 counter++;
//             }
//         }
//         std::cout<<"Search step checking complete"<<std::endl;
//         std::cout<<"Found:"<<(1-((float)counter)/((float)inputSize))*100<<"% of key"<<std::endl;
//     }

//     return host_search_result;
// }


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
