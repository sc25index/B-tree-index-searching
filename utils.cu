#include <iostream>
#include <math.h>
#include <cfloat>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "utils.h"

using namespace std;


int intlog(double base, double x) {
    return (int)(log(x) / log(base));
}

/* Compute the running time given start time */
double report_running_time(struct timeval startTime)
{
    long sec_diff, usec_diff;
    struct timezone Idunno;
    struct timeval endTime;

    gettimeofday(&endTime, &Idunno);
    sec_diff = endTime.tv_sec - startTime.tv_sec;
    usec_diff= endTime.tv_usec-startTime.tv_usec;
    if(usec_diff < 0) {
        sec_diff --;
        usec_diff += 1000000;
    }
    return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}

// Obtaining the correct seed value for random generator
unsigned time_seed()
{
    time_t now = time ( 0 );
    unsigned char *p = (unsigned char *)&now;
    unsigned seed = 0;

    for (size_t i = 0; i < sizeof now; i++ )
        seed = seed * ( UCHAR_MAX + 2U ) + p[i];
    return seed;
}


// Some computation to get uniform distributed random number
double uniform_deviate ( double seed ) {
    return seed * ( 1.0 / ( RAND_MAX + 1.0 ) );
}

void errorExit(char *msg) {
    cerr << "Failed: " << msg << endl;
    exit(1);
}


void printProgress(const char* format, ...)
{
      va_list argv;
      char *message;

      va_start(argv, format);
      vasprintf(&message, format, argv);
      va_end(argv);
      fprintf(stderr, "%s", message);
      fprintf(stderr, "\r");
      fflush(stdout);
}
