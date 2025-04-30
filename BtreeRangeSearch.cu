#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <cstring>

#include "utils.h"
#include "BPTree.h"
#include "grouping.h"
#include "grouping2d.h"
#include "zOrder2d.h"

#define BLOCK_SIZE 1024

#define CUDA_MAX_BLOCK 65535

#define CUDA_RT_CALL( call )                                                                       \
{                                                                                                  \
    cudaError_t cudaStatus = call;                                                                 \
    if ( cudaSuccess != cudaStatus )                                                               \
        fprintf(stderr, "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s (%d).\n", \
                        #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus);    \
}                                                                                                  \


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);

// Grouping
//Searching range state 2
__global__
void kernelSearchingKeyRangeGroupStateTwoGrouping( const ulli *dev_Btree_root, BTPage *dev_globalPointer,
                                                   const ulli *dev_search_key, ulli dev_search_key_size ,
                                                   ulli *dev_output, ulli group_key_num, ulli *dev_key_prefix,
                                                   ulli *dev_key_group_cout) {
    ulli currentGroup = blockIdx.x;
    ulli h_pos;

    while(currentGroup < group_key_num) {
        h_pos = threadIdx.x;

        const ulli* __restrict dev_input_key = &dev_search_key[dev_key_prefix[currentGroup]];

        while( h_pos < dev_key_group_cout[currentGroup] ) {
            ulli currentPtr = dev_input_key[h_pos];
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
                    if(foundAt == INVALID_DATA)
                        foundAt = start;
                    if(foundAt == searchPage->size)
                        foundAt = start-1;

                    foundAt = foundAt << 32;
                    dev_output[dev_key_prefix[currentGroup]+ h_pos] = foundAt | searchingLeaf;
                }
            }

            h_pos += blockDim.x;
        }
        currentGroup += gridDim.x;
    }
}

//Grouping
//searching state 3
__global__ void kernelGenerateSearchResultGrouping(  const ulli *dev_search_key_start,
                                                     const ulli *dev_search_key_end, ulli dev_key_Size,
                                                     const ulli *dev_search_leaf_start /*output from step2*/,
                                                     const ulli *dev_search_leaf_end /*output from step2*/,
                                                     ulli *uni_result, ulli *output_offset,
                                                     BTPage *dev_globalPointer, ulli *dev_key_prefix,
                                                     ulli *dev_key_group_cout, ulli group_key_num) {
    ulli currentGroup = blockIdx.x;
    ulli h_pos;

    while(currentGroup < group_key_num){
        const ulli* __restrict dev_input_key_start = &dev_search_key_start[dev_key_prefix[currentGroup]];
        const ulli* __restrict dev_input_key_end = &dev_search_key_end[dev_key_prefix[currentGroup]];
        const ulli* __restrict dev_input_leaf_start = &dev_search_leaf_start[dev_key_prefix[currentGroup]];
        const ulli* __restrict dev_input_leaf_end = &dev_search_leaf_end[dev_key_prefix[currentGroup]];

        h_pos = threadIdx.x;
        while( h_pos < dev_key_group_cout[currentGroup] ) {

            unsigned long long int offsetAddress = DOB_PAGE_SIZE*(dev_key_prefix[currentGroup]+ h_pos);
            ulli countOutput=0;

            ulli currentPtr_start = dev_input_key_start[h_pos];
            ulli currentPtr_end = dev_input_key_end[h_pos];
            ulli searchingLeaf_start = dev_input_leaf_start[h_pos];
            ulli searchingLeaf_startPosition = searchingLeaf_start >> 32;
            searchingLeaf_start = searchingLeaf_start & 0xffffffff;
            ulli searchingLeaf_end = dev_input_leaf_end[h_pos];
            ulli searchingLeaf_endPosition = searchingLeaf_end >> 32;
            searchingLeaf_end = searchingLeaf_end & 0xffffffff;

            if(searchingLeaf_start >= PAGE_ALLOCATION ||
                searchingLeaf_end >= PAGE_ALLOCATION){
                    printf("Something Worng!!!: %d %llu %llu\n",__LINE__,searchingLeaf_start,searchingLeaf_end);
                    uni_result[offsetAddress] = INVALID_DATA;
                    uni_result[offsetAddress+1] = INVALID_DATA;
            }
            else {
                if(searchingLeaf_start != searchingLeaf_end) {

                    {
                        BTPage *searchPage = &dev_globalPointer[searchingLeaf_start];
                        for(int i=searchingLeaf_startPosition;i<searchPage->size-1 ;i++) {
                            if(searchPage->data[i] <= currentPtr_end && searchPage->data[i]>=currentPtr_start) {
                                countOutput++;
                            }
                        }
                    }
                    ///
                    //leaf in the middle
                    ulli currentLeaf = dev_globalPointer[searchingLeaf_start].next;
                    while(currentLeaf != searchingLeaf_end) {
                        BTPage *searchPage = &dev_globalPointer[currentLeaf];
                        for(int i=0;i<searchPage->size;i++) {
                            if(searchPage->data[i] <= currentPtr_end && searchPage->data[i]>=currentPtr_start) {
                                countOutput++;
                            }
                        }
                        currentLeaf = dev_globalPointer[currentLeaf].next;
                    }
                    ///
                    {
                        BTPage *searchPage = &dev_globalPointer[searchingLeaf_end];
                        for(int i=0;i<searchingLeaf_endPosition && i<searchPage->size-1 ;i++) {
                            if(searchPage->data[i] <= currentPtr_end && searchPage->data[i]>=currentPtr_start) {
                                countOutput++;
                            }
                        }
                    }
                }
                else {
                    BTPage *searchPage = &dev_globalPointer[searchingLeaf_end];
                    for(int i=searchingLeaf_startPosition;i<=searchingLeaf_endPosition;i++) {
                        if( searchPage->data[i] >= currentPtr_start && searchPage->data[i]<= currentPtr_end) {
                            countOutput++;
                        }
                    }
                }

                //uni_result[offsetAddress] = currentPtr_start;
                //uni_result[offsetAddress+1] = currentPtr_end;
                uni_result[offsetAddress] = countOutput;
            }
            h_pos += blockDim.x;
        }

        currentGroup += gridDim.x;
    }
}

// Naive
//Searching range state 2
__global__ void kernelSearchingKeyRangeGroupStateTwo( const ulli *dev_Btree_root, BTPage *dev_globalPointer, const ulli *dev_search_key,
                                                      ulli dev_search_key_size , ulli *dev_output) {
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
                if(foundAt == INVALID_DATA)
                    foundAt = start;
                if(foundAt == searchPage->size)
                    foundAt = start-1;

                foundAt = foundAt << 32;
                dev_output[h_pos] = foundAt | searchingLeaf;
            }
        }

        h_pos += gridDim.x * blockDim.x;
    }
}

//searching state 3
__global__ void kernelGenerateSearchResult(  const ulli *dev_search_key_start,
                                             const ulli *dev_search_key_end, ulli dev_key_Size,
                                             const ulli *dev_search_leaf_start /*output from step2*/,
                                             const ulli *dev_search_leaf_end /*output from step2*/,
                                             ulli *uni_result, ulli *output_offset,
                                             BTPage *dev_globalPointer) {
    ulli h_pos = blockIdx.x * blockDim.x + threadIdx.x;
    while(h_pos < dev_key_Size){
        unsigned long long int offsetAddress = DOB_PAGE_SIZE*h_pos;
        ulli countOutput=0;

        ulli currentPtr_start = dev_search_key_start[h_pos];
        ulli currentPtr_end = dev_search_key_end[h_pos];
        ulli searchingLeaf_start = dev_search_leaf_start[h_pos];
        ulli searchingLeaf_startPosition = searchingLeaf_start >> 32;
        searchingLeaf_start = searchingLeaf_start & 0xffffffff;
        ulli searchingLeaf_end = dev_search_leaf_end[h_pos];
        ulli searchingLeaf_endPosition = searchingLeaf_end >> 32;
        searchingLeaf_end = searchingLeaf_end & 0xffffffff;


        if(searchingLeaf_start >= PAGE_ALLOCATION ||
            searchingLeaf_end >= PAGE_ALLOCATION){
                printf("Something Worng!!!: %d %llu %llu\n",__LINE__,searchingLeaf_start,searchingLeaf_end);
                uni_result[offsetAddress] = INVALID_DATA;
                uni_result[offsetAddress+1] = INVALID_DATA;
        }
        else {
            if(searchingLeaf_start != searchingLeaf_end) {

                {
                    BTPage *searchPage = &dev_globalPointer[searchingLeaf_start];
                    for(int i=searchingLeaf_startPosition;i<searchPage->size-1 ;i++) {
                        if(searchPage->key[i] <= currentPtr_end && searchPage->key[i]>=currentPtr_start) {
                            countOutput++;
                        }
                    }
                }
                ///
                //leaf in the middle
                ulli currentLeaf = dev_globalPointer[searchingLeaf_start].next;
                while(currentLeaf != searchingLeaf_end) {
                    BTPage *searchPage = &dev_globalPointer[currentLeaf];
                    for(int i=0;i<searchPage->size;i++) {
                        if(searchPage->key[i] <= currentPtr_end && searchPage->key[i]>=currentPtr_start) {
                            countOutput++;
                        }
                    }
                    currentLeaf = dev_globalPointer[currentLeaf].next;
                }
                ///
                {
                    BTPage *searchPage = &dev_globalPointer[searchingLeaf_end];
                    for(int i=0;i<searchingLeaf_endPosition && i<searchPage->size-1 ;i++) {
                        if(searchPage->key[i] <= currentPtr_end && searchPage->key[i]>=currentPtr_start) {
                            countOutput++;
                        }
                    }
                }
            }
            else {
                BTPage *searchPage = &dev_globalPointer[searchingLeaf_end];
                for(int i=searchingLeaf_startPosition;i<=searchingLeaf_endPosition;i++) {
                    if( searchPage->key[i] >= currentPtr_start && searchPage->key[i]<= currentPtr_end) {
                        countOutput++;
                    }
                }
            }

            //uni_result[offsetAddress] = currentPtr_start;
            //uni_result[offsetAddress+1] = currentPtr_end;
            uni_result[offsetAddress] = countOutput;

        }

        h_pos += gridDim.x * blockDim.x;
    }
}

//grouping
void* SearchingBTreeGPURangeWithGrouping(ulli *input_key_start, ulli *input_key_end,
                                         ulli inputSize, ulli *dev_Btree, BTPage *dev_globalPointer) {

    std::cout<<"SearchingBTreeGPU Range With Grouping"<<std::endl;

    std::vector<ulli> groupList;
    std::vector<ulli> groupPrefix;

    //state1
    {
        //Grouping
        ulli *host_output;
        ulli *host_output2;
        genGrouping2d(input_key_start,input_key_end,inputSize,groupList,
                      host_output,host_output2,groupPrefix,GROUP_SEARCH_SIZE);

        input_key_start=host_output;
        input_key_end=host_output2;
    }

    //fixme: didn't run new function yet


    //State2
    ulli *dev_group_cout, *host_group_count;
    ulli *dev_prefix, *host_prefix;

    ulli *dev_input_key_start, *dev_input_key_end;
    ulli *dev_result_start, *dev_result_end, *uni_result;
    cudaEvent_t m_start, m_stop;
    float m_time;

    {

        host_group_count = new ulli[groupList.size()];
        host_prefix = new ulli[groupList.size()];

        for(ulli i=0;i<groupList.size();i++) {
            host_group_count[i] = groupList[i];
            host_prefix[i] = groupPrefix[i];
        }

        cudaSetDevice(0);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );


        cudaMalloc( (void**) &(dev_group_cout), sizeof(ulli)*groupList.size());
        cudaMalloc( (void**) &(dev_prefix), sizeof(ulli)*groupList.size());

        cudaMemcpy( dev_group_cout, host_group_count, sizeof(ulli)*groupList.size(), cudaMemcpyHostToDevice);
        cudaMemcpy( dev_prefix, host_prefix, sizeof(ulli)*groupList.size(), cudaMemcpyHostToDevice);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        cudaMalloc( (void**) &(dev_result_start), sizeof(ulli)*inputSize);
        cudaMalloc( (void**) &(dev_input_key_start), sizeof(ulli)*inputSize );

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        cudaMemset( dev_result_start, 0, sizeof(ulli)*inputSize );
        cudaMemcpy( dev_input_key_start, input_key_start, sizeof(ulli)*inputSize,
                    cudaMemcpyHostToDevice);

        cudaMalloc( (void**) &(dev_result_end), sizeof(ulli)*inputSize);
        cudaMalloc( (void**) &(dev_input_key_end), sizeof(ulli)*inputSize );


        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        cudaMemset( dev_result_end, 0, sizeof(ulli)*inputSize );
        cudaMemcpy( dev_input_key_end, input_key_end, sizeof(ulli)*inputSize,
                    cudaMemcpyHostToDevice);


        ulli m_blocks = groupList.size()>CUDA_MAX_BLOCK ? CUDA_MAX_BLOCK:groupList.size();
        dim3 grid(m_blocks, 1, 1);
        dim3 threads(BLOCK_SIZE, 1, 1);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        cudaEventCreate( &m_start );
        cudaEventCreate( &m_stop );
        cudaEventRecord( m_start, 0 );

        kernelSearchingKeyRangeGroupStateTwoGrouping<<<grid,threads>>>( dev_Btree, dev_globalPointer,
                                                                dev_input_key_start, inputSize,dev_result_start,
                                                                groupList.size(), dev_prefix, dev_group_cout);


    }

    {

        ulli m_blocks = groupList.size()>CUDA_MAX_BLOCK ? CUDA_MAX_BLOCK:groupList.size();
        dim3 grid(m_blocks, 1, 1);
        dim3 threads(BLOCK_SIZE, 1, 1);

        kernelSearchingKeyRangeGroupStateTwoGrouping<<<grid,threads>>>( dev_Btree, dev_globalPointer,
                                                                dev_input_key_end, inputSize,dev_result_end,
                                                                groupList.size(),dev_prefix, dev_group_cout);


        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        cudaDeviceSynchronize();

        cudaDeviceSynchronize();
        cudaEventRecord( m_stop, 0 );
        cudaEventSynchronize( m_stop );
        cudaEventElapsedTime( &m_time, m_start, m_stop);
        cudaEventDestroy( m_start);
        cudaEventDestroy( m_stop);

        std::cout<<"Running time Searching State2 Start-end: "<<m_time<<std::endl;
    }

    //State3
    {
        cudaEvent_t m_start, m_stop;
        float m_time;

        /* use unified memory */
        cudaMalloc( (void**) &(uni_result), sizeof(ulli)*inputSize*DOB_PAGE_SIZE*1);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        cudaMemset( uni_result, 0, sizeof(ulli)*inputSize*DOB_PAGE_SIZE*1 );


        //std::memset(uni_result, 0, sizeof(ulli)*inputSize*DOB_PAGE_SIZE*1.5);

        ulli *output_offset;

        cudaMalloc( (void**) &(output_offset), sizeof(ulli)*inputSize);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        ulli m_blocks = groupList.size()>CUDA_MAX_BLOCK ? CUDA_MAX_BLOCK:groupList.size();
        dim3 grid(m_blocks, 1, 1);
        dim3 threads(BLOCK_SIZE, 1, 1);

        cudaMemset( output_offset, m_blocks*BLOCK_SIZE, sizeof(ulli) );

        cudaEventCreate( &m_start );
        cudaEventCreate( &m_stop );
        cudaEventRecord( m_start, 0 );


        kernelGenerateSearchResultGrouping<<<grid,threads>>>(  dev_input_key_start,
                                                               dev_input_key_end, inputSize,
                                                               dev_result_start /*output from step2*/,
                                                               dev_result_end /*output from step2*/,
                                                               uni_result, output_offset,
                                                               dev_globalPointer, dev_prefix,
                                                               dev_group_cout, groupList.size());


        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        cudaDeviceSynchronize();

        cudaEventRecord( m_stop, 0 );
        cudaEventSynchronize( m_stop );
        cudaEventElapsedTime( &m_time, m_start, m_stop);
        cudaEventDestroy( m_start);
        cudaEventDestroy( m_stop);
        std::cout<<"Running time Searching State3: "<<m_time<<std::endl;

        // for(int i=0;i<inputSize*3;i=i+3) {
        //     if(uni_result[i+1]-uni_result[i]+1 < uni_result[i+2]) {
        //         std::cout<<uni_result[i]<<" "<<uni_result[i+1]<<" "<<uni_result[i+2]<<" Diff:"<<uni_result[i+1]-uni_result[i]<<std::endl;
        //     }
        // }
        // std::cout<<"Checking complete\n";

        //copy result out to cpu

        cudaFree(output_offset);
        cudaFree(uni_result);
    }

    cudaFree(dev_group_cout);
    cudaFree(dev_prefix);
    cudaFree(dev_result_start);
    cudaFree(dev_input_key_start);
    cudaFree(dev_result_end);
    cudaFree(dev_input_key_end);

    return 0;
}

//grouping
void* SearchingBTreeGPURangeWithGroupingZorder(ulli *input_key_start, ulli *input_key_end,
                                               ulli inputSize, ulli *dev_Btree, BTPage *dev_globalPointer) {

    std::cout<<"SearchingBTreeGPU Range With Grouping Zorder"<<std::endl;

    std::vector<ulli> groupList;
    std::vector<ulli> groupPrefix;

    //state1
    ulli *host_zOrder;
    {
        genGrouping2dZ(input_key_start, input_key_end, host_zOrder, inputSize);
    }

    //state1
    {
        //Grouping
        ulli *host_output;
        ulli *host_output2;
        ulli *host_outputZorder;
        genGroupingWithTwodata(host_zOrder, input_key_start, input_key_end, inputSize,
                               host_outputZorder, host_output, host_output2,
                               groupList, groupPrefix,GROUP_SEARCH_SIZE);

        gpuErrchk( cudaPeekAtLastError() );

        input_key_start=host_output;
        input_key_end=host_output2;
    }

    //State2
    ulli *dev_group_cout, *host_group_count;
    ulli *dev_prefix, *host_prefix;

    ulli *dev_input_key_start, *dev_input_key_end;
    ulli *dev_result_start, *dev_result_end, *uni_result;
    cudaEvent_t m_start, m_stop;
    float m_time;

    {

        host_group_count = new ulli[groupList.size()];
        host_prefix = new ulli[groupList.size()];

        for(ulli i=0;i<groupList.size();i++) {
            host_group_count[i] = groupList[i];
            host_prefix[i] = groupPrefix[i];
        }

        cudaSetDevice(0);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );


        cudaMalloc( (void**) &(dev_group_cout), sizeof(ulli)*groupList.size());
        cudaMalloc( (void**) &(dev_prefix), sizeof(ulli)*groupList.size());

        cudaMemcpy( dev_group_cout, host_group_count, sizeof(ulli)*groupList.size(), cudaMemcpyHostToDevice);
        cudaMemcpy( dev_prefix, host_prefix, sizeof(ulli)*groupList.size(), cudaMemcpyHostToDevice);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        cudaMalloc( (void**) &(dev_result_start), sizeof(ulli)*inputSize);
        cudaMalloc( (void**) &(dev_input_key_start), sizeof(ulli)*inputSize );

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        cudaMemset( dev_result_start, 0, sizeof(ulli)*inputSize );
        cudaMemcpy( dev_input_key_start, input_key_start, sizeof(ulli)*inputSize,
                    cudaMemcpyHostToDevice);

        cudaMalloc( (void**) &(dev_result_end), sizeof(ulli)*inputSize);
        cudaMalloc( (void**) &(dev_input_key_end), sizeof(ulli)*inputSize );


        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        cudaMemset( dev_result_end, 0, sizeof(ulli)*inputSize );
        cudaMemcpy( dev_input_key_end, input_key_end, sizeof(ulli)*inputSize,
                    cudaMemcpyHostToDevice);


        ulli m_blocks = groupList.size()>CUDA_MAX_BLOCK ? CUDA_MAX_BLOCK:groupList.size();
        dim3 grid(m_blocks, 1, 1);
        dim3 threads(BLOCK_SIZE, 1, 1);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        cudaEventCreate( &m_start );
        cudaEventCreate( &m_stop );
        cudaEventRecord( m_start, 0 );

        kernelSearchingKeyRangeGroupStateTwoGrouping<<<grid,threads>>>( dev_Btree, dev_globalPointer,
                                                                dev_input_key_start, inputSize,dev_result_start,
                                                                groupList.size(), dev_prefix, dev_group_cout);


    }

    {

        ulli m_blocks = groupList.size()>CUDA_MAX_BLOCK ? CUDA_MAX_BLOCK:groupList.size();
        dim3 grid(m_blocks, 1, 1);
        dim3 threads(BLOCK_SIZE, 1, 1);

        kernelSearchingKeyRangeGroupStateTwoGrouping<<<grid,threads>>>( dev_Btree, dev_globalPointer,
                                                                dev_input_key_end, inputSize,dev_result_end,
                                                                groupList.size(),dev_prefix, dev_group_cout);


        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        cudaDeviceSynchronize();

        cudaDeviceSynchronize();
        cudaEventRecord( m_stop, 0 );
        cudaEventSynchronize( m_stop );
        cudaEventElapsedTime( &m_time, m_start, m_stop);
        cudaEventDestroy( m_start);
        cudaEventDestroy( m_stop);

        std::cout<<"Running time Searching State2 Start-end: "<<m_time<<std::endl;
    }

    //State3
    {
        cudaEvent_t m_start, m_stop;
        float m_time;

        /* use unified memory */
        cudaMalloc( (void**) &(uni_result), sizeof(ulli)*inputSize*DOB_PAGE_SIZE*1);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        cudaMemset( uni_result, 0, sizeof(ulli)*inputSize*DOB_PAGE_SIZE*1 );


        //std::memset(uni_result, 0, sizeof(ulli)*inputSize*DOB_PAGE_SIZE*1.5);

        ulli *output_offset;

        cudaMalloc( (void**) &(output_offset), sizeof(ulli)*inputSize);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        ulli m_blocks = groupList.size()>CUDA_MAX_BLOCK ? CUDA_MAX_BLOCK:groupList.size();
        dim3 grid(m_blocks, 1, 1);
        dim3 threads(BLOCK_SIZE, 1, 1);

        cudaMemset( output_offset, m_blocks*BLOCK_SIZE, sizeof(ulli) );

        cudaEventCreate( &m_start );
        cudaEventCreate( &m_stop );
        cudaEventRecord( m_start, 0 );


        kernelGenerateSearchResultGrouping<<<grid,threads>>>(  dev_input_key_start,
                                                               dev_input_key_end, inputSize,
                                                               dev_result_start /*output from step2*/,
                                                               dev_result_end /*output from step2*/,
                                                               uni_result, output_offset,
                                                               dev_globalPointer, dev_prefix,
                                                               dev_group_cout, groupList.size());


        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        cudaDeviceSynchronize();

        cudaEventRecord( m_stop, 0 );
        cudaEventSynchronize( m_stop );
        cudaEventElapsedTime( &m_time, m_start, m_stop);
        cudaEventDestroy( m_start);
        cudaEventDestroy( m_stop);
        std::cout<<"Running time Searching State3: "<<m_time<<std::endl;

        // for(int i=0;i<inputSize*3;i=i+3) {
        //     if(uni_result[i+1]-uni_result[i]+1 < uni_result[i+2]) {
        //         std::cout<<uni_result[i]<<" "<<uni_result[i+1]<<" "<<uni_result[i+2]<<" Diff:"<<uni_result[i+1]-uni_result[i]<<std::endl;
        //     }
        // }
        // std::cout<<"Checking complete\n";

        //copy result out to cpu

        cudaFree(output_offset);
        cudaFree(uni_result);
    }

    cudaFree(dev_group_cout);
    cudaFree(dev_prefix);
    cudaFree(dev_result_start);
    cudaFree(dev_input_key_start);
    cudaFree(dev_result_end);
    cudaFree(dev_input_key_end);

    return 0;
}

//naive
void* SearchingBTreeGPURange(ulli *input_key_start, ulli *input_key_end, ulli inputSize, ulli *dev_Btree,
                             BTPage *dev_globalPointer) {

    std::cout<<"SearchingBTreeGPU Range"<<std::endl;

    //State2
    ulli *dev_input_key_start, *dev_input_key_end;
    ulli *dev_result_start, *dev_result_end, *uni_result;
    cudaEvent_t m_start, m_stop;
    float m_time;

    {

        cudaSetDevice(0);

        cudaMalloc( (void**) &(dev_result_start), sizeof(ulli)*inputSize);
        cudaMalloc( (void**) &(dev_input_key_start), sizeof(ulli)*inputSize );

        cudaMemset( dev_result_start, 0, sizeof(ulli)*inputSize );
        cudaMemcpy( dev_input_key_start, input_key_start, sizeof(ulli)*inputSize,
                    cudaMemcpyHostToDevice);

        cudaMalloc( (void**) &(dev_result_end), sizeof(ulli)*inputSize);
        cudaMalloc( (void**) &(dev_input_key_end), sizeof(ulli)*inputSize );

        cudaMemset( dev_result_end, 0, sizeof(ulli)*inputSize );
        cudaMemcpy( dev_input_key_end, input_key_end, sizeof(ulli)*inputSize,
                    cudaMemcpyHostToDevice);


        ulli m_blocks = floor(inputSize/BLOCK_SIZE)>CUDA_MAX_BLOCK ? CUDA_MAX_BLOCK:floor(inputSize/BLOCK_SIZE);
        dim3 grid(m_blocks, 1, 1);
        dim3 threads(BLOCK_SIZE, 1, 1);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        cudaEventCreate( &m_start );
        cudaEventCreate( &m_stop );
        cudaEventRecord( m_start, 0 );

        kernelSearchingKeyRangeGroupStateTwo<<<grid,threads>>>( dev_Btree, dev_globalPointer,
                                                                dev_input_key_start, inputSize,dev_result_start);

    }

    {

        ulli m_blocks = floor(inputSize/BLOCK_SIZE)>CUDA_MAX_BLOCK ? CUDA_MAX_BLOCK:floor(inputSize/BLOCK_SIZE);
        dim3 grid(m_blocks, 1, 1);
        dim3 threads(BLOCK_SIZE, 1, 1);

        kernelSearchingKeyRangeGroupStateTwo<<<grid,threads>>>( dev_Btree, dev_globalPointer,
                                                                dev_input_key_end, inputSize,dev_result_end);


        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        cudaDeviceSynchronize();

        cudaDeviceSynchronize();
        cudaEventRecord( m_stop, 0 );
        cudaEventSynchronize( m_stop );
        cudaEventElapsedTime( &m_time, m_start, m_stop);
        cudaEventDestroy( m_start);
        cudaEventDestroy( m_stop);

        std::cout<<"Running time Searching State2 Start-end: "<<m_time<<std::endl;
    }

    //State3
    {
        cudaEvent_t m_start, m_stop;
        float m_time;

        /* use unified memory */
        cudaMalloc( (void**) &(uni_result), sizeof(ulli)*inputSize*DOB_PAGE_SIZE*1);

        gpuErrchk( cudaPeekAtLastError() );

        cudaMemset( uni_result, 0, sizeof(ulli)*inputSize*DOB_PAGE_SIZE*1);

        gpuErrchk( cudaPeekAtLastError() );

        ulli *output_offset;

        cudaMalloc( (void**) &(output_offset), sizeof(ulli)*inputSize);

        gpuErrchk( cudaPeekAtLastError() );

        ulli m_blocks = floor(inputSize/BLOCK_SIZE)>CUDA_MAX_BLOCK ? CUDA_MAX_BLOCK:floor(inputSize/BLOCK_SIZE);
        dim3 grid(m_blocks, 1, 1);
        dim3 threads(BLOCK_SIZE, 1, 1);

        cudaMemset( output_offset, m_blocks*BLOCK_SIZE, sizeof(ulli) );

        gpuErrchk( cudaPeekAtLastError() );

        cudaEventCreate( &m_start );
        cudaEventCreate( &m_stop );
        cudaEventRecord( m_start, 0 );


        kernelGenerateSearchResult<<<grid,threads>>>(  dev_input_key_start,
                                                       dev_input_key_end, inputSize,
                                                       dev_result_start /*output from step2*/,
                                                       dev_result_end /*output from step2*/,
                                                       uni_result, output_offset,
                                                       dev_globalPointer);


        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(0) );

        cudaDeviceSynchronize();

        cudaEventRecord( m_stop, 0 );
        cudaEventSynchronize( m_stop );
        cudaEventElapsedTime( &m_time, m_start, m_stop);
        cudaEventDestroy( m_start);
        cudaEventDestroy( m_stop);
        std::cout<<"Running time Searching State3: "<<m_time<<std::endl;

        // for(int i=0;i<inputSize*3;i=i+3) {
        //     if(uni_result[i+1]-uni_result[i]+1 < uni_result[i+2]) {
        //         std::cout<<uni_result[i]<<" "<<uni_result[i+1]<<" "<<uni_result[i+2]<<" Diff:"<<uni_result[i+1]-uni_result[i]<<std::endl;
        //     }
        // }
        // std::cout<<"Checking complete\n";
        cudaFree(output_offset);
        cudaFree(uni_result);
    }

    cudaFree(dev_result_start);
    cudaFree(dev_input_key_start);
    cudaFree(dev_result_end);
    cudaFree(dev_input_key_end);

    return 0;
}


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
