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

typedef struct {
  float real;
  float imag;
} Complex8;


/**
 * \brief ???????????
 * 
 */
typedef struct {
	int32_t Theta;
	int32_t Phi;
	int32_t Lambda;
	int8_t target_qbit;
	int8_t control_qbit;
	int8_t gate_type;
	int8_t padding;
} gate_kernel_type;



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

//printf("burst size in bytes: %d\n", max_get_burst_size(maxfile, NULL));

    return 0;
}



/**
 * \brief ???????????
 * 
 */
int load2LMEM( Complex8* data, size_t dim ) {

    // test whether the DFE engine can be initialized
    if ( initialize_DFE() ) {
        printf("Failed to initialize the DFE engine\n");
        return 1;
    }

    size_t element_num = dim*dim;

    // convert data to fixpoint number representation into (0,31) and a sign bit and transpose the matrix to get column-major representation
    int32_t* data_fix = (int32_t*)malloc( 2*element_num*sizeof(int32_t) );
    for (size_t row_idx=0; row_idx<dim; row_idx++) {
        for (size_t col_idx=0; col_idx<dim; col_idx++) {
            data_fix[2*(col_idx*dim+row_idx)] = ceil(data[row_idx*dim+col_idx].real*(1<<30));
            data_fix[2*(col_idx*dim+row_idx)+1] = ceil(data[row_idx*dim+col_idx].imag*(1<<30));
        }
    }

    // upload data to DFE LMEM
    qgdDFE_writeLMem_actions_t interface_actions;
    interface_actions.param_element_num = 2*element_num;
    interface_actions.instream_fromcpu_0 = (void*)data_fix;
    interface_actions.instream_fromcpu_1 = (void*)data_fix;
    interface_actions.instream_fromcpu_2 = (void*)data_fix;
    interface_actions.instream_fromcpu_3 = (void*)data_fix;     

    qgdDFE_writeLMem_run( engine, &interface_actions);

    free( data_fix );
 //   free( data_fix_1 );
 //   free( data_fix_2 );
 //   free( data_fix_3 );            


//#ifdef DEBUG
    printf("Data uploaded to DFE LMEM\n");
//#endif

    return 0;

}



/**
 * \brief ???????????
 * 
 */
int downloadFromLMEM( Complex8** data, size_t dim ) {

    // test whether the DFE engine can be initialized
    if ( initialize_DFE() ) {
        printf("Failed to initialize the DFE engine\n");
        return 1;
    }

    size_t element_num = dim*dim;

    // cast fix point to floats
    // convert data to fixpoint number representation into (0,31) and a sign bit
    int32_t** data_fix[4];
    for( size_t idx=0; idx<4; idx++) {
        data_fix[idx] = (int32_t*)malloc( 2*element_num*sizeof(int32_t) );
    }     


    // download data from DFE LMEM
    qgdDFE_readLMem_actions_t interface_actions;
    interface_actions.param_element_num = 2*element_num;
    interface_actions.outstream_tocpu_0 = (void*)data_fix[0];
    interface_actions.outstream_tocpu_1 = (void*)data_fix[1];
    interface_actions.outstream_tocpu_2 = (void*)data_fix[2];
    interface_actions.outstream_tocpu_3 = (void*)data_fix[3];            

    qgdDFE_readLMem_run( engine, &interface_actions);

    for( size_t idx=0; idx<4; idx++) {
        Complex8* data_loc = data[idx];
        int32_t* data_fix_loc = data_fix[idx];        
        for (size_t row_idx=0; row_idx<dim; row_idx++) {
            for (size_t col_idx=0; col_idx<dim; col_idx++) {
                data_loc[row_idx*dim+col_idx].real = ((float)data_fix_loc[2*(col_idx*dim+row_idx)]/(1<<30));
                data_loc[row_idx*dim+col_idx].imag = ((float)data_fix_loc[2*(col_idx*dim+row_idx)+1]/(1<<30));
            }
        }
    }

    for( size_t idx=0; idx<4; idx++) {
        free( data_fix[idx] );
    }            

//#ifdef DEBUG
    printf("Data downloaded from DFE LMEM\n");
//#endif


    return 0;

}



/**
 * \brief ???????????
 * 
 */
int calcqgdKernelDFE(size_t dim, gate_kernel_type* gates, int gatesNum)
{

/*
    // test whether the DFE engine can be initialized
    if ( initialize_DFE() ) {
        printf("Failed to initialize the DFE engine\n");
        return 1;
    }
*/


    if ( gatesNum % qgdDFE_CHAINED_GATES_NUM > 0 ) {
        printf("The number of gates should be a multiple of %d, but %d was given\n", qgdDFE_CHAINED_GATES_NUM, gatesNum);
        return -1;
    }

    const uint32_t dataIn[SIZE] = { 1, 0, 2, 0, 4, 1, 8, 3 };
    uint32_t dataOut[SIZE];

    size_t element_num = dim*dim;
//int targetQubit = 2;

printf("%d, control qbit: %d, target qbit: %d\n", element_num, gates->control_qbit, gates->target_qbit);
printf("size of gate_kernel_type %d bytes\n", sizeof(gate_kernel_type));

    qgdDFE_actions_t interface_actions;
    //interface_actions.ticks_qgdDFEKernel = element_num*2;
    interface_actions.param_element_num  = element_num;
    interface_actions.param_dim = dim;
    interface_actions.param_gatesNum = gatesNum;

    interface_actions.instream_gatesfromcpu_0 = (void*)gates;
    interface_actions.instream_size_gatesfromcpu_0 = sizeof(gate_kernel_type)*gatesNum;
    interface_actions.instream_gatesfromcpu_1 = (void*)gates;
    interface_actions.instream_size_gatesfromcpu_1 = sizeof(gate_kernel_type)*gatesNum;
    interface_actions.instream_gatesfromcpu_2 = (void*)gates;
    interface_actions.instream_size_gatesfromcpu_2 = sizeof(gate_kernel_type)*gatesNum;
    interface_actions.instream_gatesfromcpu_3 = (void*)gates;
    interface_actions.instream_size_gatesfromcpu_3 = sizeof(gate_kernel_type)*gatesNum;
    
    //interface_actions.routing_string = "gatesDFEFanout10 -> gatesDFEChain10, gatesDFEFanout20->gatesDFEChain20, gatesDFEFanout21 -> gatesDFEChain21, gatesDFEFanout30 -> gatesDFEChain30, gatesDFEFanout31 -> gatesDFEChain31, gatesDFEFanout32 -> gatesDFEChain32, gatesDFEFanout32 -> gatesDFEChain32";
	
    //interface_actions.instream_x = dataIn;
    //interface_actions.instream_size_x = SIZE * sizeof dataIn[0];
    //interface_actions.outstream_y = dataOut; 
    //interface_actions.outstream_size_y = SIZE * sizeof dataOut[0];     


    qgdDFE_run(	engine, &interface_actions);
printf("gates num:%d\n", gatesNum );
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


