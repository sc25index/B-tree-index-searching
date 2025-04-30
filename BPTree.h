#ifndef __INCLUDED_BPTree_H__
#define __INCLUDED_BPTree_H__

#include <vector>

#define PAGE_ALLOCATION 200041 //21000 //200041 //30000

#define PAGESIZE 5000
#define GROUP_SEARCH_SIZE 33000 //32000 //35000 //40000 //30000
#define DOB_PAGE_SIZE 1

#define INVALID_DATA 0xffffffffffffffff

typedef struct {
    ulli key[PAGESIZE];
    ulli data[PAGESIZE];
    bool leafFlag;
    int size;
    ulli max;
    ulli min;
    int previous;
    int next;
}BTPage;


void* genBPlusTree(ulli *devInputKey, ulli *devInput, ulli inputSize, std::vector<ulli> groupList,
                   std::vector<ulli> groupPrefix, ulli *&dev_btree_out, ulli &treeLevel_out,
                   ulli *&btreeIndex_out, ulli &bTreeSize_out,
                   BTPage *&dev_globalPointer_in, ulli *&dev_currentPage_in );


//grouping idea 3 step
// void* SearchingBTreeGPU(ulli *input_key, ulli inputSize, ulli *dev_Btree, BTPage *dev_globalPointer);

//naive one step
void* SearchingBTreeGPUOnlyOneState(ulli *input_key, ulli inputSize,
                                    ulli *dev_Btree, BTPage *dev_globalPointer);

//grouping idea 2 step
void* SearchingBTreeGPUCombineStep(ulli *input_key, ulli inputSize, ulli *dev_Btree,
                                   BTPage *dev_globalPointer);

//naive idea 2 step
// void* SearchingBTreeGPUNaive(ulli *input_key, ulli inputSize, ulli *dev_Btree, BTPage *dev_globalPointer);

//naive range
void* SearchingBTreeGPURange(ulli *input_key_start, ulli *input_key_end, ulli inputSize, ulli *dev_Btree,
                             BTPage *dev_globalPointer);

//grouping range
void* SearchingBTreeGPURangeWithGrouping(ulli *input_key_start, ulli *input_key_end,
                                         ulli inputSize, ulli *dev_Btree, BTPage *dev_globalPointer);

void* SearchingBTreeGPURangeWithGroupingZorder(ulli *input_key_start, ulli *input_key_end,
                                               ulli inputSize, ulli *dev_Btree, BTPage *dev_globalPointer);
#endif
