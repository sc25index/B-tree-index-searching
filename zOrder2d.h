#ifndef __INCLUDED_ZORDER_2D_H__
#define __INCLUDED_ZORDER_2D_H__

#include <vector>

__global__ extern void kernelPrefixSum( ulli *hist, ulli * prefixSum);

void genGrouping2dZ(ulli *input, ulli* input2, ulli* &output, ulli inputSize) ;

#endif
