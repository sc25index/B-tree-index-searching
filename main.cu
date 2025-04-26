#include <iostream>
#include <ctime>
#include <vector>
#include <random>
#include <fstream>


#include <limits.h>
#include <omp.h>

#include <cub/cub.cuh>

#include "utils.h"
#include "grouping.h"
#include "BPTree.h"


///////////Main.//////////////////
int main(int argc, char *argv[]) {

    if(argc < 4){
        std::cerr<<"usage:"<<argv[0]<<" inputSize fileNameKey filenameData"<<std::endl;
        return -1;
    }

    if(strlen(argv[1]) > 18){
        std::cerr<<"Error inputSize greater than long long"<<std::endl;
        return -1;
    }

    std::cout<<"Init Algorithm with PageSize "<<PAGESIZE<<std::endl;

    //random input
    ulli inputSize = atoll(argv[1]);
    ulli *input = new ulli[inputSize];
    ulli *inputData = new ulli[inputSize];

    {
        std::ifstream infile;
        infile.open(argv[2], std::ios::binary | std::ios::in);
        if ( infile.fail() ) {
            std::cerr << "Error: " << strerror(errno) <<std::endl;
            return -1;
        }
        infile.read((char*)input, sizeof(ulli) * inputSize);
    }

    {
        std::ifstream infile;
        infile.open(argv[3], std::ios::binary | std::ios::in);
        if ( infile.fail() ) {
            std::cerr << "Error: " << strerror(errno) <<std::endl;
            return -1;
        }
        infile.read((char*)inputData, sizeof(ulli) * inputSize);
    }

    std::cout<<"Init problem complete."<<std::endl;
    std::cout<<"Problem Size: "<<inputSize/1000000<<" Million Key"<<std::endl;

    //struct timeval tvalBefore;
    //gettimeofday (&tvalBefore, NULL);

    std::vector<ulli> groupList;
    std::vector<ulli> groupPrefix;

    //std::cout<<inputSize<<",";

    ulli *dev_output;
    ulli *dev_output_data;
    {
        cudaSetDevice(0);
        ulli *m_input;
        ulli *m_output;
        ulli *m_input_data;
        ulli *m_output_data;
        cudaMalloc( (void**) &(m_input), sizeof(ulli)*inputSize);
        cudaMalloc( (void**) &(m_output), sizeof(ulli)*inputSize);

        cudaMalloc( (void**) &(m_input_data), sizeof(ulli)*inputSize);
        cudaMalloc( (void**) &(m_output_data), sizeof(ulli)*inputSize);

        {
            cudaEvent_t m_start, m_stop;
            float m_time;

            cudaEventCreate( &m_start );
            cudaEventCreate( &m_stop );
            cudaEventRecord( m_start, 0 );

            cudaMemcpy( m_input, input, sizeof(ulli)*inputSize, cudaMemcpyHostToDevice);
            cudaMemset( m_output, 0, sizeof(ulli)*inputSize );

            cudaMemcpy( m_input_data, inputData, sizeof(ulli)*inputSize, cudaMemcpyHostToDevice);
            cudaMemset( m_output_data, 0, sizeof(ulli)*inputSize );

            cudaEventRecord( m_stop, 0 );
            cudaEventSynchronize( m_stop );
            cudaEventElapsedTime( &m_time, m_start, m_stop);
            cudaEventDestroy( m_start);
            cudaEventDestroy( m_stop);

            std::cout<<"I/O in time:"<<m_time/1000<<" Sec."<<std::endl;
        }

        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;

        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, m_input, m_output,
                                        m_input_data, m_output_data, inputSize);

        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // run cuda
        //sort

        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, m_input, m_output,
                                        m_input_data, m_output_data, inputSize);

        cudaFree(m_input);
        cudaFree(m_input_data);
        cudaFree(d_temp_storage);
        dev_output = m_output;
        dev_output_data = m_output_data;

        for(ulli i=0;i<inputSize;i=i+PAGESIZE) {
            groupList.push_back(PAGESIZE);
            groupPrefix.push_back(i);
        }
    }

    ulli *dev_btree = 0;
    ulli treeLevel = 0;
    ulli *btreeIndex =0;
    ulli bTreeSize = 0;

    BTPage *dev_globalPointer = 0;
    ulli *dev_currentPage = 0;

    std::cout<<"Data sorted and building the Tree."<<std::endl;

    genBPlusTree(dev_output, dev_output_data, inputSize, groupList, groupPrefix, dev_btree,
                 treeLevel, btreeIndex, bTreeSize, dev_globalPointer, dev_currentPage);

    //std::cout<<groupList.size()<<std::endl;
    //std::cout<<groupPrefix.size()<<std::endl;

    // int next=0;
    // int lastNext=0;
    // while(next != -1) {
    //     for(int i=0;i<dev_globalPointer[next].size;i++) {
    //         if(dev_globalPointer[next].data[i]!=dev_globalPointer[next].key[i])
    //             std::cout<<"Something wrong with the tree!!!!"<<std::endl;
    //     }
    //     lastNext = next;
    //     next = dev_globalPointer[next].next;
    // }
    // std::cout<<"Checking complete\nTotal page:"<<lastNext<<std::endl;

    if(argc == 6) {
        ulli inputSize_key = atoll(argv[4]);
        ulli *input_key = new ulli[inputSize_key];

        std::cout<<"======================================================================="<<std::endl;

        {
            size_t x,y;
            cudaMemGetInfo(&x,&y);
            fprintf(stderr, "Free memory %lu MB, Total memory %lu MB\n", x/1000000, y/1000000);
        }

        {
            std::ifstream infile;
            infile.open(argv[5], std::ios::binary | std::ios::in);
            if ( infile.fail() ) {
                std::cerr << "Error: " << strerror(errno)<<std::endl;
                return -1;
            }
            infile.read((char*)input_key, sizeof(ulli) * inputSize_key);

            std::cout<<"Init Searching key complete."<<std::endl;
            std::cout<<"Problem Key Size: "<<inputSize_key/1000000<<" Million Key"<<std::endl;
        }


        //std::cout<<dev_globalPointer<<" "<<dev_currentPage<<std::endl;

        //grouping with two steps
        ulli *result0= (ulli*)SearchingBTreeGPUCombineStep(input_key, inputSize_key,
                                                          dev_btree, dev_globalPointer);


        std::cout<<"======================================================================="<<std::endl;

        {
            size_t x,y;
            cudaMemGetInfo(&x,&y);
            fprintf(stderr, "Free memory %lu MB, Total memory %lu MB\n", x/1000000, y/1000000);
        }

        {
            std::ifstream infile;
            infile.open(argv[5], std::ios::binary | std::ios::in);
            if ( infile.fail() ) {
                std::cerr << "Error: " << strerror(errno)<<std::endl;
                return -1;
            }
            infile.read((char*)input_key, sizeof(ulli) * inputSize_key);

            std::cout<<"Init Searching key complete."<<std::endl;
            std::cout<<"Problem Key Size: "<<inputSize_key/1000000<<" Million Key"<<std::endl;
        }

        // Naive with one state
        ulli *result1= (ulli*)SearchingBTreeGPUOnlyOneState(input_key, inputSize_key,
                                                           dev_btree, dev_globalPointer);

        std::cout<<"======================================================================="<<std::endl;

        {
            size_t x,y;
            cudaMemGetInfo(&x,&y);
            fprintf(stderr, "Free memory %lu MB, Total memory %lu MB\n", x/1000000, y/1000000);
        }

        {
            std::ifstream infile;
            infile.open(argv[5], std::ios::binary | std::ios::in);
            if ( infile.fail() ) {
                std::cerr << "Error: " << strerror(errno)<<std::endl;
                return -1;
            }
            infile.read((char*)input_key, sizeof(ulli) * inputSize_key);

            std::cout<<"Init Searching key complete."<<std::endl;
            std::cout<<"Problem Key Size: "<<inputSize_key/1000000<<" Million Key"<<std::endl;
        }

        // grouping idea with three steps
        ulli *result2= (ulli*)SearchingBTreeGPU(input_key, inputSize_key,
                                              dev_btree, dev_globalPointer);


        std::cout<<"======================================================================="<<std::endl;

        {
            size_t x,y;
            cudaMemGetInfo(&x,&y);
            fprintf(stderr, "Free memory %lu MB, Total memory %lu MB\n", x/1000000, y/1000000);
        }

        {
            std::ifstream infile;
            infile.open(argv[5], std::ios::binary | std::ios::in);
            if ( infile.fail() ) {
                std::cerr << "Error: " << strerror(errno)<<std::endl;
                return -1;
            }
            infile.read((char*)input_key, sizeof(ulli) * inputSize_key);

            std::cout<<"Init Searching key complete."<<std::endl;
            std::cout<<"Problem Key Size: "<<inputSize_key/1000000<<" Million Key"<<std::endl;
        }

        //naive with two steps
        ulli *result3= (ulli*)SearchingBTreeGPUNaive(input_key, inputSize_key,
                                                    dev_btree, dev_globalPointer);

        std::cout<<"======================================================================="<<std::endl;

        {
            size_t x,y;
            cudaMemGetInfo(&x,&y);
            fprintf(stderr, "Free memory %lu MB, Total memory %lu MB\n", x/1000000, y/1000000);
        }

    }

    if(argc == 7) {
        ulli inputSize_key = atoll(argv[4]);
        ulli *input_key_start = new ulli[inputSize_key];
        ulli *input_key_end = new ulli[inputSize_key];

        {
            std::ifstream infile;
            infile.open(argv[5], std::ios::binary | std::ios::in);
            if ( infile.fail() ) {
                std::cerr << "Error: " << strerror(errno)<<std::endl;
                return -1;
            }
            infile.read((char*)input_key_start, sizeof(ulli) * inputSize_key);
        }

        {
            std::ifstream infile;
            infile.open(argv[6], std::ios::binary | std::ios::in);
            if ( infile.fail() ) {
                std::cerr << "Error: " << strerror(errno)<<std::endl;
                return -1;
            }
            infile.read((char*)input_key_end, sizeof(ulli) * inputSize_key);
        }

        std::cout<<"Init Searching key complete."<<std::endl;
        std::cout<<"Problem Key Size: "<<inputSize_key/1000000<<" Million Key"<<std::endl;
        //std::cout<<dev_globalPointer<<" "<<dev_currentPage<<std::endl;
        SearchingBTreeGPURange(input_key_start,input_key_end, inputSize_key, dev_btree, dev_globalPointer);

        std::cout<<"======================================================================="<<std::endl;

        {
            std::ifstream infile;
            infile.open(argv[5], std::ios::binary | std::ios::in);
            if ( infile.fail() ) {
                std::cerr << "Error: " << strerror(errno)<<std::endl;
                return -1;
            }
            infile.read((char*)input_key_start, sizeof(ulli) * inputSize_key);
        }

        {
            std::ifstream infile;
            infile.open(argv[6], std::ios::binary | std::ios::in);
            if ( infile.fail() ) {
                std::cerr << "Error: " << strerror(errno)<<std::endl;
                return -1;
            }
            infile.read((char*)input_key_end, sizeof(ulli) * inputSize_key);
        }

        SearchingBTreeGPURangeWithGrouping(input_key_start,input_key_end, inputSize_key, dev_btree, dev_globalPointer);

        std::cout<<"======================================================================="<<std::endl;

        {
            std::ifstream infile;
            infile.open(argv[5], std::ios::binary | std::ios::in);
            if ( infile.fail() ) {
                std::cerr << "Error: " << strerror(errno)<<std::endl;
                return -1;
            }
            infile.read((char*)input_key_start, sizeof(ulli) * inputSize_key);
        }

        {
            std::ifstream infile;
            infile.open(argv[6], std::ios::binary | std::ios::in);
            if ( infile.fail() ) {
                std::cerr << "Error: " << strerror(errno)<<std::endl;
                return -1;
            }
            infile.read((char*)input_key_end, sizeof(ulli) * inputSize_key);
        }

        SearchingBTreeGPURangeWithGroupingZorder(input_key_start,input_key_end, inputSize_key, dev_btree, dev_globalPointer);

    }
    //std::cout<<PAGESIZE<<","<<inputSize/1000000<<","<<report_running_time(tvalBefore);
    //std::cout<<std::endl;

    //check
    //std::cout<<"Tree"<<std::endl;
    //printTreeCPU(root);


    return 0;
}
