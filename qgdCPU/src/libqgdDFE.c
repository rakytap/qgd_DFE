#include <stdint.h>
#include <stdlib.h>
#include <MaxSLiCInterface.h>
#include <math.h>

#include <time.h>

#include "qgdDFE.h"

#define SIZE 8


/**
 * \brief ???????????
 * 
 */
typedef struct { 
	double real;
	double imag;
} Complex16;

/// static variable to indicate whether DFE is initialized
static bool initialized = false;
static max_file_t* maxfile = NULL;
static max_engine_t *engine = NULL;





/**
@brief Interface function to releive DFE
*/
void releive_DFE()
{

    if (!initialized) return;

//#ifdef DEBUG
    printf("Unloading Maxfile\n");
//#endif

    // unload the max files from the devices
    initialized = false;

    max_unload(engine);

    max_file_free(maxfile);

    qgdDFE_free();  
}


/**
@brief Interface function to releive DFE
*/
int initialize_DFE()
{

    if (initialized) return 0;
  
	
    maxfile = qgdDFE_init();


    if (!maxfile) return 1;

    engine = max_load(maxfile, "local:*");

    if (!engine) { 
        max_file_free(maxfile); 
        return 1; 
    }


    initialized = true;

#ifdef DEBUG
    printf("Maxfile uploaded to DFE\n");
#endif

    return 0;
}


/**
 * \brief ???????????
 * 
 */
typedef struct { 
	char amplitude[14*2];
	int32_t base_idx;
} state_vector_type;


/**
 * \brief ???????????
 * 
 */
int load2LMEM( float* data, int element_num ) {

    // test whether the DFE engine can be initialized
    if ( initialize_DFE() ) {
        printf("Failed to initialize the DFE engine\n");
        return 1;
    }

    // upload data to DFE LMEM
    qgdDFE_writeLMem_actions_t interface_actions;
    interface_actions.param_element_num = element_num;
    interface_actions.instream_fromcpu = (void*)data;

    qgdDFE_writeLMem_run( engine, &interface_actions);


//#ifdef DEBUG
    printf("Data uploaded to DFE LMEM\n");
//#endif

    return 0;

}



/**
 * \brief ???????????
 * 
 */
int downloadFromLMEM( float* data, int element_num ) {

    // test whether the DFE engine can be initialized
    if ( initialize_DFE() ) {
        printf("Failed to initialize the DFE engine\n");
        return 1;
    }

    // download data from DFE LMEM
    qgdDFE_readLMem_actions_t interface_actions;
    interface_actions.param_element_num = element_num;
    interface_actions.outstream_tocpu = (void*)data;

    qgdDFE_readLMem_run( engine, &interface_actions);


//#ifdef DEBUG
    printf("Data downloaded from DFE LMEM\n");
//#endif


    return 0;

}



/**
 * \brief ???????????
 * 
 */
int calcqgdKernelDFE(int element_num)
{

/*
    // test whether the DFE engine can be initialized
    if ( initialize_DFE() ) {
        printf("Failed to initialize the DFE engine\n");
        return 1;
    }
*/


    const uint32_t dataIn[SIZE] = { 1, 0, 2, 0, 4, 1, 8, 3 };
    uint32_t dataOut[SIZE];
printf("%d\n", element_num);

    qgdDFE_actions_t interface_actions;
    interface_actions.ticks_qgdDFEKernel = element_num*2;
    interface_actions.param_element_num = element_num*2;
    //interface_actions.instream_x = dataIn;
    //interface_actions.instream_size_x = SIZE * sizeof dataIn[0];
    //interface_actions.outstream_y = dataOut; 
    //interface_actions.outstream_size_y = SIZE * sizeof dataOut[0];     


    qgdDFE_run(	engine, &interface_actions);
/*
    qgdDFE(
     	SIZE,
     	dataIn, SIZE * sizeof dataIn[0],
     	dataOut, SIZE * sizeof dataOut[0]);
*/
/*
    for(size_t i = 0; i < SIZE; i++)
    {
    	printf("dataIn[%lu] = %u\n", i, dataIn[i]);
    }

    for(size_t i = 0; i < SIZE; i++)
    {
    	printf("dataOut[%lu] = %u\n", i, dataOut[i]);
    }

    printf("Done!\n");
*/
    return 0;

}


