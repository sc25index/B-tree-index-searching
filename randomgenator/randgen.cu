#include <iostream>
//#include "utils.h"
//#include "grouping.h"
//#include "BPTree.h"
#include <ctime>
#include <vector>
#include <random>
#include <fstream>
#include <algorithm>
#include <stdlib.h>
#include <string>
#include "randlib/src/randlib.c"

#include <limits.h>
// #include <omp.h>
// #include <python3.10/Python.h>
// #include <python3.10/listobject.h>
// #include "matplotlibcpp.h"
// namespace plt = matplotlibcpp;


typedef unsigned long long int ulli;
int first = TRUE;      // Static first time flag for zipf


int zipf(double alpha, int n);

int randNumGen(int max);

///////////Main.//////////////////
int main(int argc, char *argv[]) {


    std::ifstream infile;
    infile.open(argv[1], std::ios::binary | std::ios::in);
    if ( infile.fail() ) {
        std::cerr << "Error: " <<strerror(errno) << std::endl; 
        return -1;
    }
    infile.seekg(0,infile.end);
    long long length = infile.tellg();
    infile.seekg(0,infile.beg);
    std::cout<<"length of array: "<<length/sizeof(ulli)<<std::endl;
    
    ulli * buffer = new ulli [length/sizeof(ulli)];

    long long al=length/sizeof(ulli);

    std::cout<<"total:"<<al<<std::endl;
    infile.read((char*) buffer,length);

    std::cout<<std::endl;
    // std::cout << "sorting-------------------------"<<std::endl;
    // std::sort(buffer,buffer+al);
    // std::cout<<buffer[0]<<":"<<buffer[al-1]<<std::endl;
    


    // //random query input
    ulli inputSize = atoll(argv[2]);
    ulli *input = new ulli[inputSize];
    ulli maxNumber;
    maxNumber = * std::max_element(buffer,buffer+al); //buffer[al-1];

    std::ofstream ofile;
    int dflag;

    // distribution flag: 0-uniform; 1-normal; 2-zipf; 3-multiGaussian
    dflag=atoi(argv[3]); 

    std::cout<<"Maxnumber is "<<maxNumber<<std::endl;
    // std::random_device rd;
    // std::mt19937 gen(rd());
    switch(dflag){
        case 0:{  //uniform
                // std::mt19937 mt( std::time(0) ) ;
                std::random_device rd;
                std::mt19937 gen(rd());
                
                std::uniform_int_distribution<int> dist(0, maxNumber);

                std::cout<<"uniform......"<<std::endl;
            
        
                for(ulli i=0;i<inputSize;i++) {  
                    // ulli j= (ulli)dist(gen);  
                    input[i] = (ulli)dist(gen);      
                    // input[i] = buffer[100000];     
                    // input[i] = (ulli) 10000000;            
                }  
                // for (ulli i=0;i<inputSize;i++){
                //     input[i]= (ulli)dist(gen); 
                // }
                // std::string filename="uniform-"+std::to_string(inputSize/1000000)+"M-tree"+std::to_string(al/1000000)+"M";
                std::string filename="exp-uniform-"+std::to_string(al/1000000)+"-Q"+std::to_string(inputSize/1000000);
                   
                ofile.open(filename, std::ios::binary);
                ofile.write((char *)input, sizeof(ulli)*inputSize);
                ofile.close();
                std::cout<<"File saved: "<<filename <<std::endl;

                }
            break;
            case 1:{ //normal
                std::random_device rd;
                std::mt19937 gen(rd());                
                std::cout<<"normal......"<<std::endl;
                
                
                ulli xMean=(ulli) maxNumber/2;
                double stdDeviation = (double)0.35*xMean;
                std::normal_distribution<double> dist(xMean, stdDeviation);
                ulli newx;
        
                for(ulli i=0;i<inputSize;i++) {  

                    // ulli j= (ulli)dist(gen);  
                    // if (j<0.0 || j>=al)
                    //     input[i]=buffer[(ulli) al/2];
                    // else
                    //     input[i] = buffer[j];
                    newx= (ulli)dist(gen);
                    if(newx<0.0 || newx>maxNumber )
                        input[i] =xMean;
                    else
                        input [i] = (ulli)newx;
                    // std::cout<<","<<j<<std::endl;                           
                }  
               
                // std::string filename="normal-"+std::to_string(inputSize/1000000)+"M-tree"+std::to_string(al/1000000)+"M";
                std::string filename="exp-normal-"+std::to_string(al/1000000)+"-Q"+std::to_string(inputSize/1000000);
                   
                ofile.open(filename, std::ios::binary);
                ofile.write((char *)input, sizeof(ulli)*inputSize);
                ofile.close();
                std::cout<<"File saved: "<<filename <<std::endl;
                }
            break;
            case 2:{ //zipf
                // std::random_device rd;
                // std::mt19937 gen(rd());
                double Zfactor=1.0;
                // first = TRUE; /* reset the starting point for zipf generator */

                std::cout<<"zipf......"<<std::endl;

                for(ulli i=0;i<inputSize;i++) {  

                    ulli j = (ulli) (zipf(Zfactor, maxNumber));				 
                    input[i] = j;
                    // std::cout<<","<<j;                           
                }  

                std::string filename="zipf-"+std::to_string(al/1000000)+"-Q"+std::to_string(inputSize/1000000);
                   
                ofile.open(filename, std::ios::binary);
                ofile.write((char *)input, sizeof(ulli)*inputSize);
                ofile.close();
                std::cout<<"File saved: "<<filename <<std::endl;

            }
            break;
            case 3:{//multiple Gaussian

                std::cout<<"multiple Gaussian......"<<std::endl;
                double numOfDistributions = 5.0; // the number of Gaussian
                double stdDeviation = 0.1;
                ulli xMean=(ulli) maxNumber/2;
                ulli newx;


                for (int i=0;i<numOfDistributions;i++){
                    xMean=(ulli)randNumGen(maxNumber);
                    for (ulli a=0;a<(inputSize/numOfDistributions);a++){
                        newx= (ulli)gennor(xMean,stdDeviation);
                        if(newx<0.0 || newx>maxNumber )
                            input[(ulli)(i*(inputSize/numOfDistributions))+a] =xMean;
                        else
                            input [(ulli)(i*(inputSize/numOfDistributions))+a] = (ulli)newx;
                    }

                }

                // std::cout<<"Shuffling......"<<std::endl;
                // std::random_device rd;
                // std::mt19937 gen(rd()); 
                // shuffle(input, input+inputSize, gen);
                // std::cout<<"Shuffled."<<std::endl;

                // std::string filename="mulGaussian-"+std::to_string(inputSize/1000000)+"M-tree"+std::to_string(al/1000000)+"M-"+std::to_string(int(numOfDistributions));
                std::string filename="mulGaussian-"+std::to_string(al/1000000)+"-Q"+std::to_string(inputSize/1000000);   
                ofile.open(filename, std::ios::binary);
                ofile.write((char *)input, sizeof(ulli)*inputSize);
                ofile.close();
                std::cout<<"File saved: "<<filename <<std::endl;
            }
            break;
            case 4:{ //normal
                ulli xMean=(ulli) maxNumber/2;
                double stdDeviation = (double)0.63*xMean;
                ulli newx;

                std::random_device rd;
                std::mt19937 gennormal(rd());                
                std::cout<<"normal.....>>>>>>>>>>>>>>>."<<std::endl;
                std::normal_distribution<double> distnormal(xMean, stdDeviation);
                

        
                for(ulli i=0;i<inputSize;i++) {  

                    newx = (ulli) distnormal(gennormal);
                    if(newx<0.0 || newx>maxNumber )
                        input[i] =xMean;
                    else
                        input [i] = newx;
                    // std::cout<<","<<j<<std::endl;                           
                }  
               
                // std::string filename="normal-"+std::to_string(inputSize/1000000)+"M-tree"+std::to_string(al/1000000)+"M";
                std::string filename="normal-"+std::to_string(al/1000000)+"-Q"+std::to_string(inputSize/1000000);
                   
                ofile.open(filename, std::ios::binary);
                ofile.write((char *)input, sizeof(ulli)*inputSize);
                ofile.close();
                std::cout<<"File saved: "<<filename <<std::endl;
                }
            break;
            default:
                fprintf(stderr, "Wrong distribution code specified!\n");

        }

    // std::vector<ulli> ids = []() {std::vector<ulli> v; v.resize(inputSize); for (ulli i = 0; i < inputSize; ++i) v[i] = i; return v; }();
    // plt::hist(ids,1000);
    // plt::plot(input);
    infile.close();
    delete[] buffer;


    return 0;



}

//===========================================================================
//=  Function to generate Zipf (power law) distributed random variables     =
//=    - Input: alpha and N                                                 =
//=    - Output: Returns with Zipf distributed random variable              =
//=  This is modified from the code from Ken Christensen                    =
//=  Changed the linear search of probabilities into a binary search        =
//===========================================================================
int zipf(double alpha, int n)
{
	static double c, *temp;          // Normalization constant
	double z;                     // Uniform random number (0 < z < 1)
	double sum_prob;              // Sum of probabilities
	double zipf_value;            // Computed exponential value to be returned
	int    i;                     // Loop counter
	int head, tail, mid;			// for binary search
	
	// Compute normalization constant on first call only
	if (first == TRUE)
	{
		temp = (double *)calloc(n, sizeof(double));
		for (i=1; i<=n; i++) {
			c = c + (1.0 / pow((double) i, alpha));
			temp[i-1] = 0.0f;
		}
		c = 1.0 / c;
		sum_prob = 0.0;
		for (i=1; i<=n; i++) {
			sum_prob += c / pow((double) i, alpha);
			temp[i-1] = sum_prob;
			//printf("[%d, %lf]", i-1, sum_prob);
		}
		//printf("\n");
		first = FALSE;
	}
	
	/* generate a uniform double variable within [0, 1.0) */
	z = (double)(rand())/RAND_MAX;
	
	// Map z to the value
	sum_prob = 0;
	
	head = 0;
	tail = n;
	mid = n/2;
	//printf("For z = %lf\n", z);
	while(head <= tail) {
		//printf("working on [%d, %d, %d]\n", head, mid, tail);
		if(z > temp[mid]) {
			head = mid + 1;
			mid = head + (tail - head)/2;
		} else if (z > temp[mid-1]) {
			zipf_value = mid + 1;
			break;
		} else {
			tail = mid - 1 ;
			mid = head + (tail - head)/2;
		}
	}
	// Assert that zipf_value is between 1 and N
	assert((zipf_value >=1) && (zipf_value <= n));
	
	return(zipf_value);
}
int randNumGen(int max) {
    return (int)(rand()%max);
}