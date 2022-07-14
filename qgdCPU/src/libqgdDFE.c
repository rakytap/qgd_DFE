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
 * \brief ???????????
 * 
 */
typedef struct { 
	char amplitude[14*2];
	int32_t base_idx;
} state_vector_type;




int calcqgdKernelDFE()
{


    const uint32_t dataIn[SIZE] = { 1, 0, 2, 0, 4, 1, 8, 3 };
    uint32_t dataOut[SIZE];

    qgdDFE(
     	SIZE,
     	dataIn, SIZE * sizeof dataIn[0],
     	dataOut, SIZE * sizeof dataOut[0]);

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

/*

	uint64_t base_num = power_of_2( qbit_num );

	//////////// setting up input parameters for DFE //////////////////
	uint64_t InitialCount0 = 0;
	uint64_t InitialCount1 = base_num/4;
	uint64_t InitialCount2 = base_num/2;
	uint64_t InitialCount3 = 3*base_num/4;


printf("number of qubits: %d\n", qbit_num);
printf("base nums: %ld\n", base_num);

	// preallocate output array
	Complex16* state_vector = output_state;

	// calculations on the DFE
	max_file_t* maxfile = QFTDFE_init();
	max_engine_t *engine = max_load( maxfile, "*");

printf("burst size in bytes: %d\n", max_get_burst_size(maxfile, NULL));

	// upload initial state to LMEM
	printf("Upload LMEM\n");
	QFTDFE_writeLMem_actions_t lmem_write_action;
	lmem_write_action.param_baseNum = base_num;
	lmem_write_action.instream_fromcpu = (void*)input_state;

	QFTDFE_writeLMem_run( engine, &lmem_write_action);


	// calculate QFT
	printf("Start DFE calculations\n");
	QFTDFE_actions_t interface_actions;

	uint64_t baseNumDiffIndex = base_num/4;
	uint64_t baseNumIndexStartIncrement = base_num;
	uint64_t iterationNum = 1;

	for (int targetQbit=qbit_num-1; targetQbit>3; targetQbit=targetQbit-2) {


		interface_actions.param_FourierFactor = power_of_2( targetQbit );
		interface_actions.param_baseNumDiffIndex = baseNumDiffIndex;
		interface_actions.param_baseNumIndexStart = 0;
		interface_actions.param_burstSize = max_get_burst_size(maxfile, NULL);
		interface_actions.param_iterationNum = iterationNum;
		interface_actions.param_targetQbit = targetQbit;
		interface_actions.param_ticksMax = base_num/16;
		//interface_actions.routing_string = "split_inA -> tosplitLMEM, split_baseIndex -> tosplitLMEM, fromSplitLMEM_0 -> kernel0_input_switch, fromSplitLMEM_1 -> kernel1_input_switch, fromSplitLMEM_2 -> kernel2_input_switch, fromSplitLMEM_3 -> kernel3_input_switch, split_kernel0_output -> toMergeLMEM, split_kernel1_output -> toMergeLMEM, split_kernel2_output -> toMergeLMEM, split_kernel3_output -> toMergeLMEM, fromMergeLMEM -> merge_outA";

		QFTDFE_run( engine, &interface_actions);

		baseNumDiffIndex = baseNumDiffIndex/4;
		baseNumIndexStartIncrement = baseNumIndexStartIncrement/4;
		iterationNum = iterationNum*4;

	} 






	// download the transformed state from LMEM
	printf("Download from LMEM\n");
	QFTDFE_readLMem_actions_t lmem_read_action;
	lmem_read_action.param_baseNum = base_num;
	lmem_read_action.outstream_tocpu = (void*)state_vector;

	QFTDFE_readLMem_run( engine, &lmem_read_action );

	// release the DFE
	max_unload( engine );
	QFTDFE_free();

printf("first iteration done\n");




	for (uint64_t idx=0; idx<base_num; idx++ ) {

//printf("%d: %f + i*%f\n", state_vector_out[idx].base_idx, state_vector_out[idx].amplitude.real, state_vector_out[idx].amplitude.imag);

		//output_state[state_vector_out[idx].base_idx] = state_vector_out[idx].amplitude;

	}



*/
    return 0;
}


