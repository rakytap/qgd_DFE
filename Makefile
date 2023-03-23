all: SIM CPU

DFE:
	@$(MAKE)  DFE -C ./qgdDFE/

SIM:
	@$(MAKE)  SIM -C ./qgdDFE/

CPU:
	@$(MAKE) MODE=release MAXFILE_DIRS=/home/rakyta.peter/qgd_DFE/qgdDFE/builds/simulation -C ./qgdCPU/ SIMULATION=true
	@$(MAKE) run_sim MAXFILE_DIRS=/home/rakyta.peter/qgd_DFE/qgdDFE/builds/simulation -C ./qgdCPU/

CPUDFE:
	@$(MAKE) MODE=release MAXFILE_DIRS=/home/rakyta.peter/qgd_DFE/qgdDFE/builds/bitstream_6x18_355Mhz_staggered_9qubit_rectangular_input_2ndcorrection_trace_offset -C ./qgdCPU/
	@$(MAKE) MODE=release MAXFILE_DIRS=/home/rakyta.peter/qgd_DFE/qgdDFE/builds/bitstream_6x13_345Mhz_staggered_10qubit_rectangular_input_2ndcorrection_trace_offset -C ./qgdCPU/ TENQUBITS=true



.PHONY: all

