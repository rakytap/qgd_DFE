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
	uint8_t metadata;
} gate_kernel_type;



/// static variable to indicate whether DFE is initialized
static bool initialized = false;
static max_file_t* maxfile = NULL;

#ifndef __DUAL__
/// static pointer to DFE
static max_engine_t *engine = NULL;
#else
/// static pointer to DFE array
static max_group_t* engine_group = NULL;
#endif


/**
@brief Interface function to releive DFE
*/
int get_chained_gates_num() {

#ifdef qgdDFE_GATES_NUM_PER_KERNEL
    return qgdDFE_CHAINED_GATES_NUM*qgdDFE_GATES_NUM_PER_KERNEL;
#else
    return qgdDFE_CHAINED_GATES_NUM;
#endif    
}



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

#ifndef __DUAL__
    max_unload(engine);
#else
    max_unload_group(engine_group);
#endif

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

#ifndef __DUAL__
    engine = max_load(maxfile, "local:*");

    if (!engine) { 
        max_file_free(maxfile); 
        return 1; 
    }
#else

    engine_group = max_load_group(maxfile, MAXOS_EXCLUSIVE, "local:*", 2);

    if (!engine_group) { 
        max_file_free(maxfile); 
        return 1; 
    }

#endif


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

#ifndef __DUAL__
    // upload data to DFE LMEM
    qgdDFE_writeLMem_actions_t interface_actions;
    interface_actions.param_element_num = 2*element_num;
    interface_actions.instream_fromcpu = (void*)data_fix;

    qgdDFE_writeLMem_run( engine, &interface_actions);

#else

    // upload data to DFE LMEM
    qgdDFE_writeLMem_actions_t *pinterface_actions[2];
    qgdDFE_writeLMem_actions_t interface_actions[2];
    interface_actions[0].param_element_num = 2*element_num;
    interface_actions[0].instream_fromcpu = (void*)data_fix;
    pinterface_actions[0] = &interface_actions[0];

    interface_actions[1].param_element_num = 2*element_num;
    interface_actions[1].instream_fromcpu = (void*)data_fix;
    pinterface_actions[1] = &interface_actions[1];

    max_run_t* run0 = qgdDFE_writeLMem_run_group_nonblock(engine_group, pinterface_actions[0]);
    max_run_t* run1 = qgdDFE_writeLMem_run_group_nonblock(engine_group, pinterface_actions[1]); 
    max_wait(run0); 
    max_wait(run1);

#endif

    free( data_fix );            


//#ifdef DEBUG
    printf("Data uploaded to DFE LMEM\n");
//#endif

    return 0;

}



/**
 * \brief ???????????
 * 
 */
/*
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
*/



/**
 * \brief ???????????
 * 
 */
int calcqgdKernelDFE_oneShot(size_t dim, gate_kernel_type* gates, int gatesNum, int gateSetNum, double* trace )
{



    if ( gatesNum % (get_chained_gates_num()) > 0 ) {
        printf("The number of gates should be a multiple of %d, but %d was given\n", qgdDFE_CHAINED_GATES_NUM, gatesNum);
        return -1;
    }

    
    if ( gateSetNum % 4 > 0 ) {
        printf("The number of set of gates should be a multiple of 4, but %d was given\n", gateSetNum);
        return -1;
    }    
   
    int gateSetNum_splitted = gateSetNum/4;
    size_t element_num = dim*dim;
/*      
    uint64_t tick_counts[4];
    for (int idx=0; idx<4; idx++) {
        tick_counts[idx] = element_num*gateSetNum_splitted*gatesNum/get_chained_gates_num();
    }
  */
  

  

    // numner og gatechain iterations over a single gateSet
    int iterationNum = gatesNum/get_chained_gates_num();
//    printf("iteration num: %d\n", iterationNum );


#ifndef __DUAL__

   // allocate memory for output    
   int64_t* trace_fix = (int64_t*)malloc( 2*sizeof(int64_t)*gateSetNum );  


    // organize input gates into 32 bit chunks
    void* gates_chunked = malloc(sizeof(gate_kernel_type)*gatesNum*gateSetNum);

    int idx_chunked = 0;
    for( int iterationNum_idx=0; iterationNum_idx<iterationNum; iterationNum_idx++ ) {
        for( int gateSet_idx=0; gateSet_idx<gateSetNum_splitted; gateSet_idx++ ) {
            for( int gate_idx=0; gate_idx<get_chained_gates_num(); gate_idx++ ) {

                int idx_orig    = gateSet_idx*gatesNum + iterationNum_idx*get_chained_gates_num() + gate_idx;

//printf("%d, %d\n", idx_chunked, idx_orig);
 
                const void* gates_0 = (void*)&gates[idx_orig + 0*gatesNum*gateSetNum_splitted];
                const void* gates_1 = (void*)&gates[idx_orig + 1*gatesNum*gateSetNum_splitted];
                const void* gates_2 = (void*)&gates[idx_orig + 2*gatesNum*gateSetNum_splitted];
                const void* gates_3 = (void*)&gates[idx_orig + 3*gatesNum*gateSetNum_splitted];

                memcpy( gates_chunked + idx_chunked*64 + 0, gates_0, 4 );
                memcpy( gates_chunked + idx_chunked*64 + 4, gates_1, 4 );
                memcpy( gates_chunked + idx_chunked*64 + 8, gates_2, 4 );
                memcpy( gates_chunked + idx_chunked*64 + 12, gates_3, 4 );

                memcpy( gates_chunked + idx_chunked*64 + 16, gates_0+4, 4 );
                memcpy( gates_chunked + idx_chunked*64 + 20, gates_1+4, 4 );
                memcpy( gates_chunked + idx_chunked*64 + 24, gates_2+4, 4 );
                memcpy( gates_chunked + idx_chunked*64 + 28, gates_3+4, 4 );

                memcpy( gates_chunked + idx_chunked*64 + 32, gates_0+8, 4 );
                memcpy( gates_chunked + idx_chunked*64 + 36, gates_1+8, 4 );
                memcpy( gates_chunked + idx_chunked*64 + 40, gates_2+8, 4 );
                memcpy( gates_chunked + idx_chunked*64 + 44, gates_3+8, 4 );

                memcpy( gates_chunked + idx_chunked*64 + 48, gates_0+12, 4 );
                memcpy( gates_chunked + idx_chunked*64 + 52, gates_1+12, 4 );
                memcpy( gates_chunked + idx_chunked*64 + 56, gates_2+12, 4 );
                memcpy( gates_chunked + idx_chunked*64 + 60, gates_3+12, 4 );

                idx_chunked++;

            }
        }
    }



    qgdDFE_actions_t interface_actions;

    interface_actions.param_element_num          = element_num;
    interface_actions.param_dim                  = dim;
    interface_actions.param_gatesNum             = gatesNum;
    interface_actions.param_gateSetNum_0         = gateSetNum_splitted;
    interface_actions.param_gateSetNum_1         = gateSetNum_splitted;
    interface_actions.param_gateSetNum_2         = gateSetNum_splitted;
    interface_actions.param_gateSetNum_3         = gateSetNum_splitted; 
    interface_actions.instream_gatesfromcpu      = gates_chunked;
    interface_actions.instream_size_gatesfromcpu = sizeof(gate_kernel_type)*gatesNum*gateSetNum;
    interface_actions.ticks_GateDataSplitKernel  = gatesNum*gateSetNum;
    interface_actions.outstream_trace2cpu        = trace_fix;
    interface_actions.outstream_size_trace2cpu   = 2*sizeof(int64_t)*gateSetNum;
    interface_actions.ticks_TraceMergeKernel     = 2*gateSetNum_splitted;   
  

    qgdDFE_run(	engine, &interface_actions);  

    free( gates_chunked );
    gates_chunked = NULL;

#else


    int gateSetNum_splitted0 = gateSetNum_splitted/2;
    int gateSetNum_splitted1 = gateSetNum_splitted - gateSetNum_splitted0;
    int gateSetNum0 = gateSetNum_splitted0*4;
    int gateSetNum1 = gateSetNum_splitted1*4;

   // allocate memory for output    
   int64_t* trace_fix0 = (int64_t*)malloc( 2*sizeof(int64_t)*gateSetNum0 ); 
   int64_t* trace_fix1 = (int64_t*)malloc( 2*sizeof(int64_t)*gateSetNum1 ); 


    gate_kernel_type* gates0 = gates;
    gate_kernel_type* gates1 = gates + gatesNum*gateSetNum0;

    // organize input gates into 32 bit chunks
    void* gates_chunked0 = malloc(sizeof(gate_kernel_type)*gatesNum*gateSetNum0); ////TODO gateSetNum_splitted0
    void* gates_chunked1 = malloc(sizeof(gate_kernel_type)*gatesNum*gateSetNum1); ////TODO gateSetNum_splitted1

    int idx_chunked = 0;
    for( int iterationNum_idx=0; iterationNum_idx<iterationNum; iterationNum_idx++ ) {
        for( int gateSet_idx=0; gateSet_idx<gateSetNum_splitted0; gateSet_idx++ ) { ////TODO gateSetNum_splitted0
            for( int gate_idx=0; gate_idx<get_chained_gates_num(); gate_idx++ ) {

                int idx_orig    = gateSet_idx*gatesNum + iterationNum_idx*get_chained_gates_num() + gate_idx;
//printf("%d, %d\n", idx_chunked, idx_orig);
 
                const void* gates_0 = (void*)&gates0[idx_orig + 0*gatesNum*gateSetNum_splitted0]; ////TODO gateSetNum_splitted0
                const void* gates_1 = (void*)&gates0[idx_orig + 1*gatesNum*gateSetNum_splitted0]; ////TODO gateSetNum_splitted0
                const void* gates_2 = (void*)&gates0[idx_orig + 2*gatesNum*gateSetNum_splitted0]; ////TODO gateSetNum_splitted0
                const void* gates_3 = (void*)&gates0[idx_orig + 3*gatesNum*gateSetNum_splitted0]; ////TODO gateSetNum_splitted0

                memcpy( gates_chunked0 + idx_chunked*64 + 0, gates_0, 4 );
                memcpy( gates_chunked0 + idx_chunked*64 + 4, gates_1, 4 );
                memcpy( gates_chunked0 + idx_chunked*64 + 8, gates_2, 4 );
                memcpy( gates_chunked0 + idx_chunked*64 + 12, gates_3, 4 );

                memcpy( gates_chunked0 + idx_chunked*64 + 16, gates_0+4, 4 );
                memcpy( gates_chunked0 + idx_chunked*64 + 20, gates_1+4, 4 );
                memcpy( gates_chunked0 + idx_chunked*64 + 24, gates_2+4, 4 );
                memcpy( gates_chunked0 + idx_chunked*64 + 28, gates_3+4, 4 );

                memcpy( gates_chunked0 + idx_chunked*64 + 32, gates_0+8, 4 );
                memcpy( gates_chunked0 + idx_chunked*64 + 36, gates_1+8, 4 );
                memcpy( gates_chunked0 + idx_chunked*64 + 40, gates_2+8, 4 );
                memcpy( gates_chunked0 + idx_chunked*64 + 44, gates_3+8, 4 );

                memcpy( gates_chunked0 + idx_chunked*64 + 48, gates_0+12, 4 );
                memcpy( gates_chunked0 + idx_chunked*64 + 52, gates_1+12, 4 );
                memcpy( gates_chunked0 + idx_chunked*64 + 56, gates_2+12, 4 );
                memcpy( gates_chunked0 + idx_chunked*64 + 60, gates_3+12, 4 );

                idx_chunked++;

            }
        }
    }


    idx_chunked = 0;
    for( int iterationNum_idx=0; iterationNum_idx<iterationNum; iterationNum_idx++ ) {
        for( int gateSet_idx=0; gateSet_idx<gateSetNum_splitted1; gateSet_idx++ ) { ////TODO gateSetNum_splitted1
            for( int gate_idx=0; gate_idx<get_chained_gates_num(); gate_idx++ ) {

                int idx_orig    = gateSet_idx*gatesNum + iterationNum_idx*get_chained_gates_num() + gate_idx;
//printf("%d, %d\n", idx_chunked, idx_orig);
 
                const void* gates_0 = (void*)&gates1[idx_orig + 0*gatesNum*gateSetNum_splitted1]; ////TODO gateSetNum_splitted1
                const void* gates_1 = (void*)&gates1[idx_orig + 1*gatesNum*gateSetNum_splitted1]; ////TODO gateSetNum_splitted1
                const void* gates_2 = (void*)&gates1[idx_orig + 2*gatesNum*gateSetNum_splitted1]; ////TODO gateSetNum_splitted1
                const void* gates_3 = (void*)&gates1[idx_orig + 3*gatesNum*gateSetNum_splitted1]; ////TODO gateSetNum_splitted1

                memcpy( gates_chunked1 + idx_chunked*64 + 0, gates_0, 4 );
                memcpy( gates_chunked1 + idx_chunked*64 + 4, gates_1, 4 );
                memcpy( gates_chunked1 + idx_chunked*64 + 8, gates_2, 4 );
                memcpy( gates_chunked1 + idx_chunked*64 + 12, gates_3, 4 );

                memcpy( gates_chunked1 + idx_chunked*64 + 16, gates_0+4, 4 );
                memcpy( gates_chunked1 + idx_chunked*64 + 20, gates_1+4, 4 );
                memcpy( gates_chunked1 + idx_chunked*64 + 24, gates_2+4, 4 );
                memcpy( gates_chunked1 + idx_chunked*64 + 28, gates_3+4, 4 );

                memcpy( gates_chunked1 + idx_chunked*64 + 32, gates_0+8, 4 );
                memcpy( gates_chunked1 + idx_chunked*64 + 36, gates_1+8, 4 );
                memcpy( gates_chunked1 + idx_chunked*64 + 40, gates_2+8, 4 );
                memcpy( gates_chunked1 + idx_chunked*64 + 44, gates_3+8, 4 );

                memcpy( gates_chunked1 + idx_chunked*64 + 48, gates_0+12, 4 );
                memcpy( gates_chunked1 + idx_chunked*64 + 52, gates_1+12, 4 );
                memcpy( gates_chunked1 + idx_chunked*64 + 56, gates_2+12, 4 );
                memcpy( gates_chunked1 + idx_chunked*64 + 60, gates_3+12, 4 );

                idx_chunked++;

            }
        }
    }


    qgdDFE_actions_t *pinterface_actions[2];
    qgdDFE_actions_t interface_actions[2];

    interface_actions[0].param_element_num          = element_num;
    interface_actions[0].param_dim                  = dim;
    interface_actions[0].param_gatesNum             = gatesNum;
    interface_actions[0].param_gateSetNum_0         = gateSetNum_splitted0; ////TODO gateSetNum_splitted0
    interface_actions[0].param_gateSetNum_1         = gateSetNum_splitted0; ////TODO gateSetNum_splitted0
    interface_actions[0].param_gateSetNum_2         = gateSetNum_splitted0; ////TODO gateSetNum_splitted0
    interface_actions[0].param_gateSetNum_3         = gateSetNum_splitted0; ////TODO gateSetNum_splitted0
    interface_actions[0].instream_gatesfromcpu      = gates_chunked0;
    interface_actions[0].instream_size_gatesfromcpu = sizeof(gate_kernel_type)*gatesNum*gateSetNum0; ////TODO gateSetNum0
    interface_actions[0].ticks_GateDataSplitKernel  = gatesNum*gateSetNum0; ////TODO gateSetNum0
    interface_actions[0].outstream_trace2cpu        = trace_fix0; 
    interface_actions[0].outstream_size_trace2cpu   = 2*sizeof(int64_t)*gateSetNum0; ////TODO gateSetNum0
    interface_actions[0].ticks_TraceMergeKernel     = 2*gateSetNum_splitted0;  ////TODO gateSetNum_splitted0
    pinterface_actions[0] = &interface_actions[0];


    interface_actions[1].param_element_num          = element_num;
    interface_actions[1].param_dim                  = dim;
    interface_actions[1].param_gatesNum             = gatesNum;
    interface_actions[1].param_gateSetNum_0         = gateSetNum_splitted1; ////TODO gateSetNum_splitted1
    interface_actions[1].param_gateSetNum_1         = gateSetNum_splitted1; ////TODO gateSetNum_splitted1
    interface_actions[1].param_gateSetNum_2         = gateSetNum_splitted1; ////TODO gateSetNum_splitted1
    interface_actions[1].param_gateSetNum_3         = gateSetNum_splitted1; ////TODO gateSetNum_splitted1
    interface_actions[1].instream_gatesfromcpu      = gates_chunked1;
    interface_actions[1].instream_size_gatesfromcpu = sizeof(gate_kernel_type)*gatesNum*gateSetNum1; ////TODO gateSetNum1
    interface_actions[1].ticks_GateDataSplitKernel  = gatesNum*gateSetNum1; ////TODO gateSetNum1
    interface_actions[1].outstream_trace2cpu        = trace_fix1; 
    interface_actions[1].outstream_size_trace2cpu   = 2*sizeof(int64_t)*gateSetNum1; ////TODO gateSetNum1
    interface_actions[1].ticks_TraceMergeKernel     = 2*gateSetNum_splitted1; ////TODO gateSetNum_splitted1
    pinterface_actions[1] = &interface_actions[1]; 


    max_run_t* run0 = qgdDFE_run_group_nonblock(engine_group, pinterface_actions[0]);
    max_run_t* run1 = qgdDFE_run_group_nonblock(engine_group, pinterface_actions[1]); 
    max_wait(run0); 
    max_wait(run1);

    free( gates_chunked0 );
    gates_chunked0 = NULL;

    free( gates_chunked1 );
    gates_chunked1 = NULL;

#endif

   

    double* trace0 = trace;
    for (size_t jdx=0; jdx<gateSetNum_splitted0; jdx++ ) {      
        trace0[jdx+0*gateSetNum_splitted0] = ((double)trace_fix0[8*jdx+4]/(1<<30));
        trace0[jdx+1*gateSetNum_splitted0] = ((double)trace_fix0[8*jdx+5]/(1<<30));
        trace0[jdx+2*gateSetNum_splitted0] = ((double)trace_fix0[8*jdx+6]/(1<<30));
        trace0[jdx+3*gateSetNum_splitted0] = ((double)trace_fix0[8*jdx+7]/(1<<30));
    }


    double* trace1 = trace + gateSetNum0;
    for (size_t jdx=0; jdx<gateSetNum_splitted0; jdx++ ) {      
        trace1[jdx+0*gateSetNum_splitted1] = ((double)trace_fix1[8*jdx+4]/(1<<30));
        trace1[jdx+1*gateSetNum_splitted1] = ((double)trace_fix1[8*jdx+5]/(1<<30));
        trace1[jdx+2*gateSetNum_splitted1] = ((double)trace_fix1[8*jdx+6]/(1<<30));
        trace1[jdx+3*gateSetNum_splitted1] = ((double)trace_fix1[8*jdx+7]/(1<<30));
    }


    free( trace_fix0 );
    trace_fix0 = NULL;

    free( trace_fix1 );
    trace_fix1 = NULL;

  

    return 0;

}


/**
 * \brief ???????????
 * 
 */
int calcqgdKernelDFE(size_t dim, gate_kernel_type* gates, int gatesNum, int gateSetNum, double* trace )
{


   int gateNumOneShotLimit = 1 << 20;
   int gateSetNumOneShotLimit = gateNumOneShotLimit/gatesNum;
   gateSetNumOneShotLimit = gateSetNumOneShotLimit - (gateSetNumOneShotLimit % 4);
   
   for (int  processedGateSet = 0; processedGateSet<gateSetNum; ) {
   
       int gateSetToProcess;
       if ( gateSetNum - processedGateSet > gateSetNumOneShotLimit + 4) {
           gateSetToProcess = gateSetNumOneShotLimit;
       }
       else {
           gateSetToProcess = gateSetNum - processedGateSet;     
       }

       calcqgdKernelDFE_oneShot(dim, gates+processedGateSet*gatesNum, gatesNum, gateSetToProcess, trace+processedGateSet );
       
       
       processedGateSet = processedGateSet + gateSetToProcess;

   
   
   }
}

