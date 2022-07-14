all: SIM CPU

DFE:
	@$(MAKE)  DFE -C ./qgdDFE/

SIM:
	@$(MAKE)  SIM -C ./qgdDFE/

CPU:
	@$(MAKE) MODE=release MAXFILE_DIRS=/home/rakytap/qgd_DFE/qgdDFE/builds/simulation -C ./qgdCPU/ 
	@$(MAKE) run_sim MAXFILE_DIRS=/home/rakytap/qgd_DFE/qgdDFE/builds/simulation -C ./qgdCPU/

CPUDFE:
	@$(MAKE) MODE=release MAXFILE_DIRS=/home/rakytap/qgd_DFE/qgdDFE/builds/bitstream -C ./qgdCPU/


.PHONY: all

