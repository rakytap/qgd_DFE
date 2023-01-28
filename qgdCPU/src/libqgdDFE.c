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


/**
 * \brief Fixed point data related to a gate operation
 * \param Theta Value of Theta/2
 * \param Phi Value of Phi
 * \param Lambda Value of Lambda
 * \param target_qbit Qubit on which the gate is applied
 * \param control_qbit The control qubit. For single qubit operations control_qbit=-1
 * \param gate_type Gate type according to enumeration of gate_type defined in SQUANDER
 * \param metadata metadata The most significat bit is set to 1 for derivated gate operation. Set the (8-i)-th bit to 1 if the i-th element of the 2x2 gate kernel should be zero in the derivated gate operation. (If the 0st and 3nd element in kernel matrix should be zero then metadat should be 5 + (1<<7), since 5 = 0101. The the leading 1<<7 bit indicates that a derivation is processed.)
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

/// number of maximally available DFEs
static size_t available_num   = 0;
/// number of utilized DFEs
static size_t accelerator_num = 0;

/// static pointer to DFE array
static max_group_t* engine_group = NULL;
/// static pointer to DFE
static max_engine_t *engine = NULL;


/**
@brief Call to get the number of gates in one chain of gates operations
*/
int get_chained_gates_num() {

#ifdef qgdDFE_GATES_NUM_PER_KERNEL
    return qgdDFE_CHAINED_GATES_NUM*qgdDFE_GATES_NUM_PER_KERNEL;
#else
    return qgdDFE_CHAINED_GATES_NUM;
#endif    
}


/**
@brief Call to set the number of DFEs to be used in the calculations
*/
void set_accelerator_num( const size_t accelerator_num_in ) {

    accelerator_num = accelerator_num_in;

}



/**
@brief Call to get the number of DFEs to be used in the calculations
*/
size_t get_accelerator_num() {

    return accelerator_num;

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

    if ( accelerator_num == 1 ) {
        max_unload(engine);
    }
    else if ( accelerator_num == 2 ) {
        max_unload_group(engine_group);
    }
    else if ( accelerator_num == 3 ) {
        max_unload_group(engine_group);
    }
    else {
        printf("releive_DFE: Unsopported number of DFEs on this system\n.");
    }

    max_file_free(maxfile);

    qgdDFE_free();  
}


/**
@brief Interface function to initialize DFE
*/
int initialize_DFE()
{

    if (initialized) return 0;
  
	
    maxfile = qgdDFE_init();


    if (!maxfile) return 1;


printf("Number of available engines: %d, number of free engines %d\n", max_count_engines_present(maxfile, "local:*"), max_count_engines_free (maxfile, "local:*") );

    if ( accelerator_num == 0 ) {
        initialized = false;
        return 0;
    }
    else if ( accelerator_num == 1 ) {

        engine = max_load(maxfile, "local:*");

        if (!engine) { 
            max_file_free(maxfile); 
            return 1; 
        }

    }
    else if ( accelerator_num == 2 ) {

        engine_group = max_load_group(maxfile, MAXOS_EXCLUSIVE, "local:*", 2);

        if (!engine_group) { 
            max_file_free(maxfile); 
            return 1; 
        }

    }
    else if ( accelerator_num == 3 ) {

        engine_group = max_load_group(maxfile, MAXOS_EXCLUSIVE, "local:*", 3);
  
        if (!engine_group) { 
            max_file_free(maxfile); 
            return 1; 
        }
    }
    else {
        printf("initialize_DFE: Unsopported number of DFEs on this system\n.");
    }


    initialized = true;

#ifdef DEBUG
    printf("Maxfile uploaded to DFE\n");
#endif

//printf("burst size in bytes: %d\n", max_get_burst_size(maxfile, NULL));

    return 0;
}



/**
 * \brief Call to upload the unitary up to the DFE
 * 
 */
int load2LMEM( Complex16* data, size_t rows, size_t cols ) {

    // test whether the DFE engine can be initialized
    if ( initialize_DFE() ) {
        printf("Failed to initialize the DFE engine\n");
        return 1;
    }

    if ( accelerator_num == 0 ) {
        return 0;
    }

    size_t element_num = rows*cols;

    // convert data to fixpoint number representation into (0,31) and a sign bit and transpose the matrix to get column-major representation
    int32_t* data_fix = (int32_t*)malloc( 2*element_num*sizeof(int32_t) );
    for (size_t row_idx=0; row_idx<rows; row_idx++) {
        for (size_t col_idx=0; col_idx<cols; col_idx++) {
            data_fix[2*(col_idx*rows+row_idx)] = ceil(data[row_idx*cols+col_idx].real*(1<<30));
            data_fix[2*(col_idx*rows+row_idx)+1] = ceil(data[row_idx*cols+col_idx].imag*(1<<30));
        }
    }


    if ( accelerator_num == 1 ) {

        qgdDFE_writeLMem_actions_t interface_actions;
        interface_actions.param_element_num = 2*element_num;
        interface_actions.instream_fromcpu = (void*)data_fix;

        qgdDFE_writeLMem_run( engine, &interface_actions);

    }
    else if ( accelerator_num == 2 ) {

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

    }
    else if ( accelerator_num == 3 ) {

        // upload data to DFE LMEM
        qgdDFE_writeLMem_actions_t *pinterface_actions[3];
        qgdDFE_writeLMem_actions_t interface_actions[3];
        interface_actions[0].param_element_num = 2*element_num;
        interface_actions[0].instream_fromcpu = (void*)data_fix;
        pinterface_actions[0] = &interface_actions[0];

        interface_actions[1].param_element_num = 2*element_num;
        interface_actions[1].instream_fromcpu = (void*)data_fix;
        pinterface_actions[1] = &interface_actions[1];

        interface_actions[2].param_element_num = 2*element_num;
        interface_actions[2].instream_fromcpu = (void*)data_fix;
        pinterface_actions[2] = &interface_actions[2];

        max_run_t* run0 = qgdDFE_writeLMem_run_group_nonblock(engine_group, pinterface_actions[0]);
        max_run_t* run1 = qgdDFE_writeLMem_run_group_nonblock(engine_group, pinterface_actions[1]); 
        max_run_t* run2 = qgdDFE_writeLMem_run_group_nonblock(engine_group, pinterface_actions[2]); 
        max_wait(run0); 
        max_wait(run1);
        max_wait(run2);

    }
    else {
        printf("load2LMEM: Unsopported number of DFEs on this system\n.");
    }


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
int calcqgdKernelDFE_oneShot(size_t rows, size_t cols, gate_kernel_type* gates, int gatesNum, int gateSetNum, double* trace )
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

  

    // numner og gatechain iterations over a single gateSet
    int iterationNum = gatesNum/get_chained_gates_num();
//    printf("iteration num: %d\n", iterationNum );



    if ( accelerator_num == 1 ) {


        // allocate memory for output    
        int64_t* trace_fix = (int64_t*)malloc( 3*sizeof(int64_t)*gateSetNum );  


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
 
        interface_actions.param_rows                 = rows;
        interface_actions.param_cols                 = cols;
        interface_actions.param_gatesNum             = gatesNum;
        interface_actions.param_gateSetNum           = gateSetNum_splitted;
        interface_actions.instream_gatesfromcpu      = gates_chunked;
        interface_actions.instream_size_gatesfromcpu = sizeof(gate_kernel_type)*gatesNum*gateSetNum;
        //interface_actions.ticks_GateDataSplitKernel  = gatesNum*gateSetNum;
        interface_actions.outstream_trace2cpu        = trace_fix;
  

        qgdDFE_run(	engine, &interface_actions);  

        free( gates_chunked );
        gates_chunked = NULL;


        for (size_t jdx=0; jdx<gateSetNum_splitted; jdx++ ) {      

            int index_offset = 3*jdx;
            trace[index_offset]   = ((double)trace_fix[12*jdx+8]/(1<<30));
            trace[index_offset+1] = ((double)trace_fix[12*jdx+4]/(1<<30));
            trace[index_offset+2] = ((double)trace_fix[12*jdx+0]/(1<<30));

            index_offset = index_offset + 3*gateSetNum_splitted;
            trace[index_offset]   = ((double)trace_fix[12*jdx+9]/(1<<30));
            trace[index_offset+1] = ((double)trace_fix[12*jdx+5]/(1<<30));
            trace[index_offset+2] = ((double)trace_fix[12*jdx+1]/(1<<30));

            index_offset = index_offset + 3*gateSetNum_splitted;
            trace[index_offset]   = ((double)trace_fix[12*jdx+10]/(1<<30));
            trace[index_offset+1] = ((double)trace_fix[12*jdx+6]/(1<<30));
            trace[index_offset+2] = ((double)trace_fix[12*jdx+2]/(1<<30));

            index_offset = index_offset + 3*gateSetNum_splitted;
            trace[index_offset]   = ((double)trace_fix[12*jdx+11]/(1<<30));
            trace[index_offset+1] = ((double)trace_fix[12*jdx+7]/(1<<30));
            trace[index_offset+2] = ((double)trace_fix[12*jdx+3]/(1<<30));
        }


        free( trace_fix );
        trace_fix = NULL;



    }
    else if ( accelerator_num == 2 ) {

        int gateSetNum_splitted0 = gateSetNum_splitted/2;
        int gateSetNum_splitted1 = gateSetNum_splitted - gateSetNum_splitted0;
        int gateSetNum0 = gateSetNum_splitted0*4;
        int gateSetNum1 = gateSetNum_splitted1*4;

       // allocate memory for output    
       int64_t* trace_fix0 = (int64_t*)malloc( 3*sizeof(int64_t)*gateSetNum0 ); 
       int64_t* trace_fix1 = (int64_t*)malloc( 3*sizeof(int64_t)*gateSetNum1 ); 


        gate_kernel_type* gates0 = gates;
        gate_kernel_type* gates1 = gates + gatesNum*gateSetNum0;

        // organize input gates into 32 bit chunks
        void* gates_chunked0 = malloc(sizeof(gate_kernel_type)*gatesNum*gateSetNum0);
        void* gates_chunked1 = malloc(sizeof(gate_kernel_type)*gatesNum*gateSetNum1);

        int idx_chunked = 0;
        for( int iterationNum_idx=0; iterationNum_idx<iterationNum; iterationNum_idx++ ) {
            for( int gateSet_idx=0; gateSet_idx<gateSetNum_splitted0; gateSet_idx++ ) {
                for( int gate_idx=0; gate_idx<get_chained_gates_num(); gate_idx++ ) {

                    int idx_orig    = gateSet_idx*gatesNum + iterationNum_idx*get_chained_gates_num() + gate_idx;
//printf("%d, %d\n", idx_chunked, idx_orig);
 
                    const void* gates_0 = (void*)&gates0[idx_orig + 0*gatesNum*gateSetNum_splitted0];
                    const void* gates_1 = (void*)&gates0[idx_orig + 1*gatesNum*gateSetNum_splitted0];
                    const void* gates_2 = (void*)&gates0[idx_orig + 2*gatesNum*gateSetNum_splitted0];
                    const void* gates_3 = (void*)&gates0[idx_orig + 3*gatesNum*gateSetNum_splitted0];

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
            for( int gateSet_idx=0; gateSet_idx<gateSetNum_splitted1; gateSet_idx++ ) { 
                for( int gate_idx=0; gate_idx<get_chained_gates_num(); gate_idx++ ) {

                    int idx_orig    = gateSet_idx*gatesNum + iterationNum_idx*get_chained_gates_num() + gate_idx;
//printf("%d, %d\n", idx_chunked, idx_orig);
 
                    const void* gates_0 = (void*)&gates1[idx_orig + 0*gatesNum*gateSetNum_splitted1]; 
                    const void* gates_1 = (void*)&gates1[idx_orig + 1*gatesNum*gateSetNum_splitted1]; 
                    const void* gates_2 = (void*)&gates1[idx_orig + 2*gatesNum*gateSetNum_splitted1];
                    const void* gates_3 = (void*)&gates1[idx_orig + 3*gatesNum*gateSetNum_splitted1];

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

        interface_actions[0].param_rows                 = rows;
        interface_actions[0].param_cols                 = cols;
        interface_actions[0].param_gatesNum             = gatesNum;
        interface_actions[0].param_gateSetNum           = gateSetNum_splitted0;
        interface_actions[0].instream_gatesfromcpu      = gates_chunked0;
        interface_actions[0].instream_size_gatesfromcpu = sizeof(gate_kernel_type)*gatesNum*gateSetNum0;
        interface_actions[0].outstream_trace2cpu        = trace_fix0; 
        pinterface_actions[0] = &interface_actions[0];


        interface_actions[1].param_rows                 = rows;
        interface_actions[1].param_cols                 = cols;
        interface_actions[1].param_gatesNum             = gatesNum;
        interface_actions[1].param_gateSetNum           = gateSetNum_splitted1;
        interface_actions[1].instream_gatesfromcpu      = gates_chunked1;
        interface_actions[1].instream_size_gatesfromcpu = sizeof(gate_kernel_type)*gatesNum*gateSetNum1;
        interface_actions[1].outstream_trace2cpu        = trace_fix1; 
        pinterface_actions[1] = &interface_actions[1]; 


        max_run_t* run0 = qgdDFE_run_group_nonblock(engine_group, pinterface_actions[0]);
        max_run_t* run1 = qgdDFE_run_group_nonblock(engine_group, pinterface_actions[1]); 
        max_wait(run0); 
        max_wait(run1);

        free( gates_chunked0 );
        gates_chunked0 = NULL;

        free( gates_chunked1 );
        gates_chunked1 = NULL;


        double* trace0 = trace;
        for (size_t jdx=0; jdx<gateSetNum_splitted0; jdx++ ) { 

            int index_offset = 3*jdx;
            trace[index_offset]   = ((double)trace_fix0[12*jdx+8]/(1<<30));
            trace[index_offset+1] = ((double)trace_fix0[12*jdx+4]/(1<<30));
            trace[index_offset+2] = ((double)trace_fix0[12*jdx+0]/(1<<30));

            index_offset = index_offset + 3*gateSetNum_splitted0;
            trace[index_offset]   = ((double)trace_fix0[12*jdx+9]/(1<<30));
            trace[index_offset+1] = ((double)trace_fix0[12*jdx+5]/(1<<30));
            trace[index_offset+2] = ((double)trace_fix0[12*jdx+1]/(1<<30));        

            index_offset = index_offset + 3*gateSetNum_splitted0;
            trace[index_offset]   = ((double)trace_fix0[12*jdx+10]/(1<<30));
            trace[index_offset+1] = ((double)trace_fix0[12*jdx+6]/(1<<30));
            trace[index_offset+2] = ((double)trace_fix0[12*jdx+2]/(1<<30));        

            index_offset = index_offset + 3*gateSetNum_splitted0;
            trace[index_offset]   = ((double)trace_fix0[12*jdx+11]/(1<<30));
            trace[index_offset+1] = ((double)trace_fix0[12*jdx+7]/(1<<30));
            trace[index_offset+2] = ((double)trace_fix0[12*jdx+3]/(1<<30));        

        }


        double* trace1 = trace + 3*gateSetNum0;
        for (size_t jdx=0; jdx<gateSetNum_splitted1; jdx++ ) { 

            int index_offset = 3*jdx;
            trace1[index_offset]   = ((double)trace_fix1[12*jdx+8]/(1<<30));
            trace1[index_offset+1] = ((double)trace_fix1[12*jdx+4]/(1<<30));
            trace1[index_offset+2] = ((double)trace_fix1[12*jdx+0]/(1<<30));        

            index_offset = index_offset + 3*gateSetNum_splitted1;
            trace1[index_offset]   = ((double)trace_fix1[12*jdx+9]/(1<<30));
            trace1[index_offset+1] = ((double)trace_fix1[12*jdx+5]/(1<<30));
            trace1[index_offset+2] = ((double)trace_fix1[12*jdx+1]/(1<<30));
    
            index_offset = index_offset + 3*gateSetNum_splitted1;
            trace1[index_offset]   = ((double)trace_fix1[12*jdx+10]/(1<<30));
            trace1[index_offset+1] = ((double)trace_fix1[12*jdx+6]/(1<<30));
            trace1[index_offset+2] = ((double)trace_fix1[12*jdx+2]/(1<<30));        

            index_offset = index_offset + 3*gateSetNum_splitted1;
            trace1[index_offset]   = ((double)trace_fix1[12*jdx+11]/(1<<30));
            trace1[index_offset+1] = ((double)trace_fix1[12*jdx+7]/(1<<30));
            trace1[index_offset+2] = ((double)trace_fix1[12*jdx+3]/(1<<30));        

        }


        free( trace_fix0 );
        trace_fix0 = NULL;

        free( trace_fix1 );
        trace_fix1 = NULL;

    }
    else if ( accelerator_num == 3 ) {


        int gateSetNum_splitted0 = gateSetNum_splitted/3;
        int gateSetNum_splitted1 = gateSetNum_splitted/3;
        int gateSetNum_splitted2 = gateSetNum_splitted/3;

        int gateSetNum_splitted_remainder = gateSetNum_splitted % 3;
        if ( gateSetNum_splitted_remainder>0 ) gateSetNum_splitted0++;
        if ( gateSetNum_splitted_remainder>1 ) gateSetNum_splitted1++;

        int gateSetNum0 = gateSetNum_splitted0*4;
        int gateSetNum1 = gateSetNum_splitted1*4;
        int gateSetNum2 = gateSetNum_splitted2*4;

        // allocate memory for output    
        int64_t* trace_fix0 = (int64_t*)malloc( 3*sizeof(int64_t)*gateSetNum0 ); 
        int64_t* trace_fix1 = (int64_t*)malloc( 3*sizeof(int64_t)*gateSetNum1 ); 
        int64_t* trace_fix2 = (int64_t*)malloc( 3*sizeof(int64_t)*gateSetNum2 ); 


        gate_kernel_type* gates0 = gates;
        gate_kernel_type* gates1 = gates + gatesNum*gateSetNum0;
        gate_kernel_type* gates2 = gates1 + gatesNum*gateSetNum1;

        // organize input gates into 32 bit chunks
        void* gates_chunked0 = malloc(sizeof(gate_kernel_type)*gatesNum*gateSetNum0); 
        void* gates_chunked1 = malloc(sizeof(gate_kernel_type)*gatesNum*gateSetNum1); 
        void* gates_chunked2 = malloc(sizeof(gate_kernel_type)*gatesNum*gateSetNum2); 

        int idx_chunked = 0;
        for( int iterationNum_idx=0; iterationNum_idx<iterationNum; iterationNum_idx++ ) {
            for( int gateSet_idx=0; gateSet_idx<gateSetNum_splitted0; gateSet_idx++ ) {
                for( int gate_idx=0; gate_idx<get_chained_gates_num(); gate_idx++ ) {



                    int idx_orig    = gateSet_idx*gatesNum + iterationNum_idx*get_chained_gates_num() + gate_idx;
//printf("%d, %d\n", idx_chunked, idx_orig);
 
                    const void* gates_0 = (void*)&gates0[idx_orig + 0*gatesNum*gateSetNum_splitted0]; 
                    const void* gates_1 = (void*)&gates0[idx_orig + 1*gatesNum*gateSetNum_splitted0]; 
                    const void* gates_2 = (void*)&gates0[idx_orig + 2*gatesNum*gateSetNum_splitted0]; 
                    const void* gates_3 = (void*)&gates0[idx_orig + 3*gatesNum*gateSetNum_splitted0]; 

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
            for( int gateSet_idx=0; gateSet_idx<gateSetNum_splitted1; gateSet_idx++ ) {
                for( int gate_idx=0; gate_idx<get_chained_gates_num(); gate_idx++ ) {

                    int idx_orig    = gateSet_idx*gatesNum + iterationNum_idx*get_chained_gates_num() + gate_idx;
//printf("%d, %d\n", idx_chunked, idx_orig);
 
                    const void* gates_0 = (void*)&gates1[idx_orig + 0*gatesNum*gateSetNum_splitted1];
                    const void* gates_1 = (void*)&gates1[idx_orig + 1*gatesNum*gateSetNum_splitted1];
                    const void* gates_2 = (void*)&gates1[idx_orig + 2*gatesNum*gateSetNum_splitted1];
                    const void* gates_3 = (void*)&gates1[idx_orig + 3*gatesNum*gateSetNum_splitted1];

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



        idx_chunked = 0;
        for( int iterationNum_idx=0; iterationNum_idx<iterationNum; iterationNum_idx++ ) {
            for( int gateSet_idx=0; gateSet_idx<gateSetNum_splitted2; gateSet_idx++ ) {
                for( int gate_idx=0; gate_idx<get_chained_gates_num(); gate_idx++ ) {

                    int idx_orig    = gateSet_idx*gatesNum + iterationNum_idx*get_chained_gates_num() + gate_idx;
//printf("%d, %d\n", idx_chunked, idx_orig);
 
                    const void* gates_0 = (void*)&gates2[idx_orig + 0*gatesNum*gateSetNum_splitted2];
                    const void* gates_1 = (void*)&gates2[idx_orig + 1*gatesNum*gateSetNum_splitted2];
                    const void* gates_2 = (void*)&gates2[idx_orig + 2*gatesNum*gateSetNum_splitted2];
                    const void* gates_3 = (void*)&gates2[idx_orig + 3*gatesNum*gateSetNum_splitted2];

                    memcpy( gates_chunked2 + idx_chunked*64 + 0, gates_0, 4 );
                    memcpy( gates_chunked2 + idx_chunked*64 + 4, gates_1, 4 );
                    memcpy( gates_chunked2 + idx_chunked*64 + 8, gates_2, 4 );
                    memcpy( gates_chunked2 + idx_chunked*64 + 12, gates_3, 4 );

                    memcpy( gates_chunked2 + idx_chunked*64 + 16, gates_0+4, 4 );
                    memcpy( gates_chunked2 + idx_chunked*64 + 20, gates_1+4, 4 );
                    memcpy( gates_chunked2 + idx_chunked*64 + 24, gates_2+4, 4 );
                    memcpy( gates_chunked2 + idx_chunked*64 + 28, gates_3+4, 4 );

                    memcpy( gates_chunked2 + idx_chunked*64 + 32, gates_0+8, 4 );
                    memcpy( gates_chunked2 + idx_chunked*64 + 36, gates_1+8, 4 );
                    memcpy( gates_chunked2 + idx_chunked*64 + 40, gates_2+8, 4 );
                    memcpy( gates_chunked2 + idx_chunked*64 + 44, gates_3+8, 4 );

                    memcpy( gates_chunked2 + idx_chunked*64 + 48, gates_0+12, 4 );
                    memcpy( gates_chunked2 + idx_chunked*64 + 52, gates_1+12, 4 );
                    memcpy( gates_chunked2 + idx_chunked*64 + 56, gates_2+12, 4 );
                    memcpy( gates_chunked2 + idx_chunked*64 + 60, gates_3+12, 4 );

                    idx_chunked++;

                }
            }
        }


        qgdDFE_actions_t *pinterface_actions[3];
        qgdDFE_actions_t interface_actions[3];

        interface_actions[0].param_rows                 = rows;
        interface_actions[0].param_cols                 = cols;
        interface_actions[0].param_gatesNum             = gatesNum;
        interface_actions[0].param_gateSetNum           = gateSetNum_splitted0;
        interface_actions[0].instream_gatesfromcpu      = gates_chunked0;
        interface_actions[0].instream_size_gatesfromcpu = sizeof(gate_kernel_type)*gatesNum*gateSetNum0;
        interface_actions[0].outstream_trace2cpu        = trace_fix0; 
        pinterface_actions[0] = &interface_actions[0];


        interface_actions[1].param_rows                 = rows;
        interface_actions[1].param_cols                 = cols;
        interface_actions[1].param_gatesNum             = gatesNum;
        interface_actions[1].param_gateSetNum           = gateSetNum_splitted1;
        interface_actions[1].instream_gatesfromcpu      = gates_chunked1;
        interface_actions[1].instream_size_gatesfromcpu = sizeof(gate_kernel_type)*gatesNum*gateSetNum1;
        interface_actions[1].outstream_trace2cpu        = trace_fix1; 
        pinterface_actions[1] = &interface_actions[1]; 


        interface_actions[2].param_rows                 = rows;
        interface_actions[2].param_cols                 = cols;
        interface_actions[2].param_gatesNum             = gatesNum;
        interface_actions[2].param_gateSetNum           = gateSetNum_splitted2;
        interface_actions[2].instream_gatesfromcpu      = gates_chunked2;
        interface_actions[2].instream_size_gatesfromcpu = sizeof(gate_kernel_type)*gatesNum*gateSetNum2;
        interface_actions[2].outstream_trace2cpu        = trace_fix2; 
        pinterface_actions[2] = &interface_actions[2]; 


        max_run_t* run0 = qgdDFE_run_group_nonblock(engine_group, pinterface_actions[0]);
        max_run_t* run1 = qgdDFE_run_group_nonblock(engine_group, pinterface_actions[1]); 
        max_run_t* run2 = qgdDFE_run_group_nonblock(engine_group, pinterface_actions[2]); 
        max_wait(run0); 
        max_wait(run1);
        max_wait(run2);

        free( gates_chunked0 );
        gates_chunked0 = NULL;

        free( gates_chunked1 );
        gates_chunked1 = NULL;

        free( gates_chunked2 );
        gates_chunked2 = NULL;


        double* trace0 = trace;
        for (size_t jdx=0; jdx<gateSetNum_splitted0; jdx++ ) {      

            int index_offset = 3*jdx;
            trace[index_offset]   = ((double)trace_fix0[12*jdx+8]/(1<<30));
            trace[index_offset+1] = ((double)trace_fix0[12*jdx+4]/(1<<30));
            trace[index_offset+2] = ((double)trace_fix0[12*jdx+0]/(1<<30));        

            index_offset = index_offset + 3*gateSetNum_splitted0;
            trace[index_offset]   = ((double)trace_fix0[12*jdx+9]/(1<<30));
            trace[index_offset+1] = ((double)trace_fix0[12*jdx+5]/(1<<30));
            trace[index_offset+2] = ((double)trace_fix0[12*jdx+1]/(1<<30));        

            index_offset = index_offset + 3*gateSetNum_splitted0;
            trace[index_offset]   = ((double)trace_fix0[12*jdx+10]/(1<<30));
            trace[index_offset+1] = ((double)trace_fix0[12*jdx+6]/(1<<30));
            trace[index_offset+2] = ((double)trace_fix2[12*jdx+2]/(1<<30));        

            index_offset = index_offset + 3*gateSetNum_splitted0;
            trace[index_offset]   = ((double)trace_fix0[12*jdx+11]/(1<<30));
            trace[index_offset+1] = ((double)trace_fix0[12*jdx+7]/(1<<30));
            trace[index_offset+2] = ((double)trace_fix2[12*jdx+3]/(1<<30));        

        }


        double* trace1 = trace + 3*gateSetNum0;
        for (size_t jdx=0; jdx<gateSetNum_splitted1; jdx++ ) {  

            int index_offset = 3*jdx;
            trace1[index_offset]   = ((double)trace_fix1[12*jdx+8]/(1<<30));
            trace1[index_offset+1] = ((double)trace_fix1[12*jdx+4]/(1<<30));
            trace1[index_offset+2] = ((double)trace_fix1[12*jdx+0]/(1<<30));        

            index_offset = index_offset + 3*gateSetNum_splitted1;
            trace1[index_offset]   = ((double)trace_fix1[12*jdx+9]/(1<<30));
            trace1[index_offset+1] = ((double)trace_fix1[12*jdx+5]/(1<<30));
            trace1[index_offset+2] = ((double)trace_fix1[12*jdx+1]/(1<<30));        

            index_offset = index_offset + 3*gateSetNum_splitted1;
            trace1[index_offset]   = ((double)trace_fix1[12*jdx+10]/(1<<30));
            trace1[index_offset+1] = ((double)trace_fix1[12*jdx+6]/(1<<30));
            trace1[index_offset+2] = ((double)trace_fix1[12*jdx+2]/(1<<30));        

            index_offset = index_offset + 3*gateSetNum_splitted1;
            trace1[index_offset]   = ((double)trace_fix1[12*jdx+11]/(1<<30));
            trace1[index_offset+1] = ((double)trace_fix1[12*jdx+7]/(1<<30));
            trace1[index_offset+2] = ((double)trace_fix1[12*jdx+3]/(1<<30));        

        }


        double* trace2 = trace + 3*gateSetNum0 + 3*gateSetNum1;
        for (size_t jdx=0; jdx<gateSetNum_splitted2; jdx++ ) {      

            int index_offset = 3*jdx;
            trace2[index_offset]   = ((double)trace_fix2[12*jdx+8]/(1<<30));
            trace2[index_offset+1] = ((double)trace_fix2[12*jdx+4]/(1<<30));
            trace2[index_offset+2] = ((double)trace_fix2[12*jdx+0]/(1<<30));        

            index_offset = index_offset + 3*gateSetNum_splitted2;
            trace2[index_offset]   = ((double)trace_fix2[12*jdx+9]/(1<<30));
            trace2[index_offset+1] = ((double)trace_fix2[12*jdx+5]/(1<<30));
            trace2[index_offset+2] = ((double)trace_fix2[12*jdx+1]/(1<<30));        

            index_offset = index_offset + 3*gateSetNum_splitted2;
            trace2[index_offset]   = ((double)trace_fix2[12*jdx+10]/(1<<30));
            trace2[index_offset+1] = ((double)trace_fix2[12*jdx+6]/(1<<30));
            trace2[index_offset+2] = ((double)trace_fix2[12*jdx+2]/(1<<30));        

            index_offset = index_offset + 3*gateSetNum_splitted2;
            trace2[index_offset]   = ((double)trace_fix2[12*jdx+11]/(1<<30));
            trace2[index_offset+1] = ((double)trace_fix2[12*jdx+7]/(1<<30));
            trace2[index_offset+2] = ((double)trace_fix2[12*jdx+3]/(1<<30));        

        }


        free( trace_fix0 );
        trace_fix0 = NULL;
    
        free( trace_fix1 );
        trace_fix1 = NULL;

        free( trace_fix2 );
        trace_fix2 = NULL;


    }
    else {
        printf("calcqgdKernelDFE_oneShot: Unsopported number of DFEs on this system\n.");
    }


  

    return 0;

}


/**
 * \brief ???????????
 * 
 */
int calcqgdKernelDFE(size_t rows, size_t cols, gate_kernel_type* gates, int gatesNum, int gateSetNum, double* trace )
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

       calcqgdKernelDFE_oneShot(rows, cols, gates+processedGateSet*gatesNum, gatesNum, gateSetToProcess, trace+3*processedGateSet );
       
       
       processedGateSet = processedGateSet + gateSetToProcess;

   
   
   }
}

