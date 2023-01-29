all: SIM CPU

DFE:
	@$(MAKE)  DFE -C ./qgdDFE/

SIM:
	@$(MAKE)  SIM -C ./qgdDFE/

CPU:
	@$(MAKE) MODE=release MAXFILE_DIRS=/home/rakytap/qgd_DFE/qgdDFE/builds/simulation -C ./qgdCPU/ 
	@$(MAKE) run_sim MAXFILE_DIRS=/home/rakytap/qgd_DFE/qgdDFE/builds/simulation -C ./qgdCPU/

CPUDFE:
	@$(MAKE) MODE=release MAXFILE_DIRS=/home/rakytap/qgd_DFE/qgdDFE/builds/bitstream_6x18_350Mhz_staggered_9qubit_rectangular_input_2ndcorrection -C ./qgdCPU/
	@$(MAKE) MODE=release MAXFILE_DIRS=/home/rakytap/qgd_DFE/qgdDFE/builds/bitstream_6x13_350MHz_staggered_10qubit_rectangular_input_2ndcorrection -C ./qgdCPU/ TENQUBITS=true



.PHONY: all

