#include <iostream>
#include <ctime>
#include <random>
#include <fstream>
#include <string.h>

#include <limits.h>
#include <omp.h>

typedef unsigned long long int ulli;

///////////Main.//////////////////
int main(int argc, char *argv[]) {
    
    if(argc < 3){
        std::cerr<<"usage:"<<argv[0]<<" randomSize fileName"<<std::endl;
        return -1;
    }

    if(strlen(argv[1]) > 18){
        std::cerr<<"Error inputSize greater than long long"<<std::endl;
        return -1;
    }

    //random input
    ulli inputSize = atoll(argv[1]);
    ulli *input = new ulli[inputSize];
    
    ulli tmp = inputSize;
    {

        std::mt19937 mt( std::time(0) ) ;
        std::uniform_real_distribution<double> dist(1, pow(2,40));
        // std::normal_distribution<double> dist(pow(2,50)/2,0.1);

        #pragma omp parallel for
        for(ulli i=0;i<inputSize;i++) {
            input[i] = inputSize-i;
        }
        
        // ulli maxNumber =2147483648;//2^31
        ulli maxNumber = 8.9e11;
        // ulli maxNumber = 1.5e10;
        #pragma omp parallel for
        for(ulli i=inputSize;i<maxNumber;i++) {
            ulli j = ((ulli)dist(mt))%i;
            if(j<inputSize) {
                input[j] = i;
            }
        }
    }
    // for( int j=0;j<inputSize;j++){
    //     printf("%lld,",input[j]);
    // }
    std::ofstream ofile(argv[2], std::ios::binary);
    ofile.write((char*) input, sizeof(ulli)*inputSize);
    ofile.close();

    return 0;
}
