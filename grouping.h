#ifndef __INCLUDED_GROUPING_H__
#define __INCLUDED_GROUPING_H__

#include <vector>

void radixGroupGPU(ulli *dev_input, ulli *dev_output, std::vector<ulli> &groupList,
                    std::vector<ulli> &groupPrefix, int pass);

__host__ __device__ int bucketIdentify(ulli input, int nPass);

__global__ void kernelRadixPass( const ulli *dev_input_, ulli *dev_Hist, int nPass, ulli inputSize);

__global__ void kernelRadixPassRelocate( const ulli *dev_input, ulli *dev_output, ulli* dev_histogram,
                                         ulli *dev_Prefix, int nPass, ulli inputSize);


void* genGrouping(ulli *input, ulli inputSize, std::vector<ulli> &groupList,
					std::vector<ulli> &groupPrefix, ulli Local_Group_Size);

void genGroupingWithOnedata(ulli *input, ulli *input_data, ulli inputSize,
                            ulli *&output, ulli* &output_data,
                            std::vector<ulli> &groupList, std::vector<ulli> &groupPrefix,
                            ulli Local_Group_Size);

void genGroupingWithTwodata(ulli *input, ulli *input_data, ulli *input_data2, ulli inputSize,
                            ulli *&output, ulli* &output_data, ulli *&output_data2,
                            std::vector<ulli> &groupList, std::vector<ulli> &groupPrefix,
                            ulli Local_Group_Size);

#endif
