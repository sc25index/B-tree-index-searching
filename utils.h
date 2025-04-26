#ifndef __INCLUDED_UTILS_H__
#define __INCLUDED_UTILS_H__

typedef unsigned long long int ulli;

typedef struct ulli2s {
    ulli a;
    ulli b;
} ulli2;

int intlog(double base, double x);

/* Compute the running time given start time */
double report_running_time(struct timeval startTime);

// Obtaining the correct seed value for random generator
unsigned time_seed();

// Some computation to get uniform distributed random number
double uniform_deviate ( double seed );

void errorExit(char *msg);

void printProgress(const char* format, ...);


#endif // __INCLUDED_UTILS_H__
