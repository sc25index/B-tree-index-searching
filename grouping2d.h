#ifndef __INCLUDED_GROUPING_2D_H__
#define __INCLUDED_GROUPING_2D_H__

#include <vector>

__global__ extern void kernelPrefixSum( ulli *hist, ulli * prefixSum);

void genGrouping2d(ulli *input, ulli* input2, ulli inputSize, std::vector<ulli> &groupList,
                    ulli *&output, ulli* &output2, std::vector<ulli> &groupPrefix,
                    ulli Local_Group_Size) ;

#endif
