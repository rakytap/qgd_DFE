#include <stdint.h>
#include <stdlib.h>
#include <MaxSLiCInterface.h>

// #include "yourSlicHeaderHere.h"

#define SIZE 8

int main(void)
{
    const uint32_t dataIn[SIZE] = { 1, 0, 2, 0, 4, 1, 8, 3 };
    uint32_t dataOut[SIZE];

    // yourSlicCallHere(
    // 	SIZE,
    // 	dataIn, SIZE * sizeof dataIn[0],
    // 	dataOut, SIZE * sizeof dataOut[0]);

    for(size_t i = 0; i < SIZE; i++)
    {
    	printf("dataIn[%lu] = %u\n", i, dataIn[i]);
    }

    for(size_t i = 0; i < SIZE; i++)
    {
    	printf("dataOut[%lu] = %u\n", i, dataOut[i]);
    }

    printf("Done!\n");

    return 0;
}
