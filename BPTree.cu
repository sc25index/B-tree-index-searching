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

#define CUDA_RT_CALL( call )                                                                       \
{                                                                                                  \
    cudaError_t cudaStatus = call;                                                                 \
    if ( cudaSuccess != cudaStatus ) {                                                             \
        fprintf(stderr, "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s (%d).\n", \
                        #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); exit(-1); }   \
}


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);

__global__
void kernelMaxIntermidatePage( BTPage *dev_GobalPointer, ulli *dev_prefix_leaf_out, ulli group_num,
                               ulli *dev_currentPage, ulli *dev_intermidate_Prefix) {

    ulli uperLevelSize = ceilf((float)group_num/(float)PAGESIZE);
    int remainderOfLastPage = group_num%PAGESIZE;
    ulli currentGroup = blockIdx.x;
    while(currentGroup < uperLevelSize) {
        ulli tid = threadIdx.x;
        int itemInPage = (currentGroup != uperLevelSize-1)? PAGESIZE : remainderOfLastPage;
        if(itemInPage == 0 && currentGroup == uperLevelSize-1)
            itemInPage = PAGESIZE;

        //allocate page for output
        ulli currentGroupPage = *dev_currentPage + currentGroup;
        if(currentGroupPage >= PAGE_ALLOCATION ) {
            printf("kernelMaxIntermidatePage: %lld :Page overflow!!!\n",currentGroupPage);
            return;
        }
        BTPage *writeableSpace = &dev_GobalPointer[currentGroupPage];

        // filling a data to new page
        while( tid < itemInPage) {
            ulli searchPage = currentGroup*PAGESIZE + tid;
            BTPage *currentPageSearch = &dev_GobalPointer[dev_prefix_leaf_out[searchPage]];

            writeableSpace->key[tid] = currentPageSearch->max;
            writeableSpace->data[tid] = dev_prefix_leaf_out[searchPage];
            tid += blockDim.x;
        }
        __syncthreads();
        if(threadIdx.x == 0) {
                dev_intermidate_Prefix[currentGroup] = currentGroupPage;
                writeableSpace->size = itemInPage;
                writeableSpace->max = dev_GobalPointer[writeableSpace->data[itemInPage-1]].max;
                writeableSpace->min = dev_GobalPointer[writeableSpace->data[0]].min;
                writeableSpace->next = -1;
                writeableSpace->previous = -1;
                writeableSpace->leafFlag = false;
        }
        currentGroup += gridDim.x;
    }
}


__global__
void assignDataToLeaf(const ulli *dev_input_, const ulli *dev_input_key_, ulli *dev_prefix,
                      ulli *dev_group_cout, ulli group_num, BTPage *dev_GobalPointer,
                      ulli *dev_currentPage , ulli *dev_prefix_leaf_out ) {

    ulli currentGroup = blockIdx.x;
    BTPage *writeableSpace;

    while(currentGroup < group_num){

        if(threadIdx.x == 0) {
            writeableSpace = &dev_GobalPointer[  *dev_currentPage + currentGroup ];

            // check for page overflow
            if((*dev_currentPage + currentGroup) > PAGE_ALLOCATION) {
                printf("Page Memory Overflow\n");
            }
            // assign output leaf pointer
            dev_prefix_leaf_out[currentGroup] = *dev_currentPage + currentGroup;

            // assign next page
            if(currentGroup == group_num-1)
                writeableSpace->next = -1;
            else
                writeableSpace->next = currentGroup+1;

            // assign previous page
            if(currentGroup == 0)
                writeableSpace->previous = -1;
            else
                writeableSpace->previous = currentGroup-1;

            // set size
            writeableSpace->size = dev_group_cout[currentGroup];
            writeableSpace->max = dev_input_key_[dev_prefix[currentGroup] + dev_group_cout[currentGroup] - 1 ];
            writeableSpace->min = dev_input_key_[dev_prefix[currentGroup] ];
            writeableSpace->leafFlag = true;
        }

        ulli h_pos = threadIdx.x;
        while(h_pos < dev_group_cout[currentGroup]) {
            writeableSpace = &dev_GobalPointer[  *dev_currentPage + currentGroup ];
            writeableSpace->data[h_pos] = dev_input_[dev_prefix[currentGroup] + h_pos];
            writeableSpace->key[h_pos] = dev_input_key_[dev_prefix[currentGroup] + h_pos];

            h_pos += blockDim.x;
        }
        __syncthreads();

        currentGroup += gridDim.x;
    }

}

void* genBPlusTree(ulli *devInputKey, ulli *devInput, ulli inputSize, std::vector<ulli> groupList,
                   std::vector<ulli> groupPrefix, ulli *&dev_btree_out, ulli &treeLevel_out,
                   ulli *&btreeIndex_out, ulli &bTreeSize_out,
                   BTPage *&dev_globalPointer_in, ulli *&dev_currentPage_in ) {

    //gen leaf node
    int treeLevel = int(ceil(log(groupList.size()) / log(PAGESIZE)));
    //std::cout<<"Tree Level:"<<treeLevel<<std::endl;
    BTPage *dev_btree = 0;

    ulli *dev_group_cout, *host_group_count;
    ulli *dev_prefix, *host_prefix;
    ulli *dev_prefix_leaf_out;
    ulli *dev_intermidate_Prefix;

    BTPage *dev_globalPointer = 0;
    ulli *dev_currentPage = 0;

    host_group_count = new ulli[groupList.size()];
    host_prefix = new ulli[groupList.size()];

    cudaEvent_t m_start, m_stop;
    float m_time;

    cudaSetDevice(0);

    {
        {
            size_t x,y;
            CUDA_RT_CALL(cudaMemGetInfo(&x,&y));
            fprintf(stderr, "Free memory %lu MB, Total memory %lu MB\n", x/1000000, y/1000000);
            fprintf(stderr, "Allocate Memory %lu MB\n", sizeof(BTPage)*PAGE_ALLOCATION/1000000);
        }
        //allocate paging
        //use this line to debug
        //cudaMallocManaged( (void**) &(dev_globalPointer), sizeof(BTPage)*PAGE_ALLOCATION);
        CUDA_RT_CALL(cudaMalloc( (void**) &(dev_globalPointer), sizeof(BTPage)*PAGE_ALLOCATION));
        CUDA_RT_CALL(cudaMallocManaged( (void**) &(dev_currentPage), sizeof(ulli) ));
        CUDA_RT_CALL(cudaMalloc( (void**) &(dev_prefix_leaf_out), sizeof(ulli) * groupList.size() ));
        CUDA_RT_CALL(cudaMemset( dev_globalPointer, 0, sizeof(BTPage)*PAGE_ALLOCATION ));
        CUDA_RT_CALL(cudaMemset( dev_currentPage, 0, sizeof(ulli) ));
    }

    cudaEventCreate( &m_start );
    cudaEventCreate( &m_stop );
    cudaEventRecord( m_start, 0 );

    for(ulli i=0;i<groupList.size();i++) {
        host_group_count[i] = groupList[i];
        host_prefix[i] = groupPrefix[i];
    }

    CUDA_RT_CALL(cudaMalloc( (void**) &(dev_group_cout), sizeof(ulli)*groupList.size()));
    CUDA_RT_CALL(cudaMalloc( (void**) &(dev_prefix), sizeof(ulli)*groupList.size()));

    CUDA_RT_CALL(cudaMemcpy( dev_group_cout, host_group_count, sizeof(ulli)*groupList.size(), cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpy( dev_prefix, host_prefix, sizeof(ulli)*groupList.size(), cudaMemcpyHostToDevice));

    //gen leaf node
    {
        ulli m_blocks = groupList.size()>CUDA_MAX_BLOCK ? CUDA_MAX_BLOCK:groupList.size();
        //std::cout<<"GroupSize = "<<groupList.size()<<std::endl;
        dim3 grid(m_blocks, 1, 1);
        dim3 threads(BLOCK_SIZE, 1, 1);
        assignDataToLeaf<<<grid,threads>>>(devInput, devInputKey, dev_prefix, dev_group_cout, groupList.size(),
                                            dev_globalPointer, dev_currentPage , dev_prefix_leaf_out);

        CUDA_RT_CALL( cudaStreamSynchronize(0) );

        // std::cout<<"Check Leaf\n";
        // for(int i=0;i<groupList.size();i++) {
        //     std::cout<<"======\n";
        //     std::cout<<"Page "<<*dev_currentPage+i<<std::endl;
        //     std::cout<<"MAX:"<<dev_globalPointer[*dev_currentPage+i].max<<"\n";
        //     std::cout<<"MIN:"<<dev_globalPointer[*dev_currentPage+i].min<<"\n";
        //     std::cout<<"Size:"<<dev_globalPointer[*dev_currentPage+i].size<<"\n";
        //     std::cout<<"======\n";
        // }

        *dev_currentPage = *dev_currentPage + groupList.size();

        //clear previous data
        CUDA_RT_CALL(cudaFree(dev_prefix));
        CUDA_RT_CALL(cudaFree(devInput));
        CUDA_RT_CALL(cudaFree(devInputKey));
    }

    // For debuging
    // ulli interNodePointer = *dev_currentPage;

    //build all intermidate node
    {
        ulli intermidateSize = ceil((float)groupList.size()/(float)PAGESIZE);
        CUDA_RT_CALL(cudaMalloc( (void**) &(dev_intermidate_Prefix), sizeof(ulli)*intermidateSize));
        ulli nodeSize = ceil((float)groupList.size()/(float)PAGESIZE);
        ulli PreviousSize = groupList.size();
        while(nodeSize > 0) {
            ulli m_blocks = nodeSize>CUDA_MAX_BLOCK ? CUDA_MAX_BLOCK : nodeSize;
            dim3 grid(m_blocks, 1, 1);
            dim3 threads(BLOCK_SIZE, 1, 1);
            kernelMaxIntermidatePage<<<grid,threads>>>(dev_globalPointer, dev_prefix_leaf_out,
                                                       PreviousSize, dev_currentPage,
                                                       dev_intermidate_Prefix);

            CUDA_RT_CALL( cudaStreamSynchronize(0) );
            *dev_currentPage = *dev_currentPage + m_blocks;
            // std::cout<<"Done a round:"<<nodeSize<<std::endl;
            if(nodeSize == 1) // stop when only 1 node left
                break;
            PreviousSize = nodeSize;
            nodeSize = ceil((float)nodeSize/(float)PAGESIZE);
            ulli *dev_tmp = dev_prefix_leaf_out;
            dev_prefix_leaf_out = dev_intermidate_Prefix;
            dev_intermidate_Prefix = dev_tmp;
        }
        CUDA_RT_CALL(cudaFree(dev_prefix_leaf_out));
    }
    ulli *rootBtreeLocation = new ulli;
    CUDA_RT_CALL(cudaMemcpy( rootBtreeLocation, dev_intermidate_Prefix, sizeof(ulli), cudaMemcpyDeviceToHost));
    dev_btree = &dev_globalPointer[*rootBtreeLocation];
    CUDA_RT_CALL(cudaFree(dev_intermidate_Prefix));

    CUDA_RT_CALL(cudaDeviceSynchronize());
    cudaEventRecord( m_stop, 0 );
    cudaEventSynchronize( m_stop );
    cudaEventElapsedTime( &m_time, m_start, m_stop);
    cudaEventDestroy( m_start);
    cudaEventDestroy( m_stop);

    //std::cout<<"Current Page at:"<<*dev_currentPage<<std::endl;
    //std::cout<<"Running time Building Tree:"<<m_time/1000<<"Sec."<<std::endl;
    std::cout<<"Building the Tree "<<m_time/1000<<std::endl;
    // std::cout<<"Page at "<<*rootBtreeLocation<<std::endl;
    // std::cout<<"MAX:"<<dev_btree->max<<"\n";
    // std::cout<<"MIN:"<<dev_btree->min<<"\n";
    // std::cout<<"Size:"<<dev_btree->size<<"\n";

    // For debuging
    // for(int i=interNodePointer;i<=*rootBtreeLocation;i++) {
    //     std::cout<<"Page at "<<i<<std::endl;
    //     std::cout<<"MAX:"<<dev_globalPointer[i].max<<"\n";
    //     std::cout<<"MIN:"<<dev_globalPointer[i].min<<"\n";
    //     std::cout<<"Size:"<<dev_globalPointer[i].size<<"\n";
    // }

    CUDA_RT_CALL(cudaMalloc( (void**) &(dev_btree_out), sizeof(ulli)));
    CUDA_RT_CALL(cudaMemcpy( dev_btree_out, rootBtreeLocation, sizeof(ulli), cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaDeviceSynchronize());
    dev_globalPointer_in = dev_globalPointer;
    dev_currentPage_in = dev_currentPage;
    //treeLevel_out = treeLevel;
    //btreeIndex_out = btreeIndex;
#if 0
    //ulli foundAt = searchingBTreeWarper(dev_btree,treeLevel,btreeIndex,1000000);

    //std::cout<<"FOUND AT:"<<foundAt<<std::endl;


#if 0
    {
        ulli pageNumber = groupList.size()/PAGESIZE;
        for(int i=int(log(groupList.size())/log(PAGESIZE));i>=0;i--) {
            std::cerr<<"Level "<<i<<": "<<btreeLen[i]<<std::endl;
        }
    }
#endif
/*
    for(ulli i =0;i<LOCAL_BLOCK_SIZE && i<50;i++) {
        std::cout<<dev_btree[i]<<std::endl;
    }

    for(ulli i =0;i<LOCAL_BLOCK_SIZE && i<50;i++) {
        std::cout<<dev_btree[btreeIndex[1]+i]<<std::endl;
    }
*/
    free(host_group_count);
    free(host_prefix);
#endif
    return dev_btree;
}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
