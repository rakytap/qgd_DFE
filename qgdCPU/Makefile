##### Build directories #####
DIST_DIR = dist
RELEASE_DIR := "$(DIST_DIR)/release"
DEBUG_DIR := "$(DIST_DIR)/debug"

##### Target executable or shared-object name. ####
ifeq ($(strip $(TENQUBITS)), true)
	TARGET := libqgdDFE_10qubits.so
else ifeq ($(strip $(SIMULATION)), true)
	TARGET := libqgdDFE_SIM.so
else
	TARGET := libqgdDFE.so
endif

##### SLiC configuration #####
USE_SLIC=1
include $(MAXCOMPILERDIR)/lib/Makefile.include

ifeq ($(strip $(DUAL)),true)
	COMPVAR := -D__DUAL__
endif

ifeq ($(strip $(TRIAL)),true)
	COMPVAR := -D__TRIAL__
endif


MAXFILES := $(foreach D,$(MAXFILE_DIRS),$(wildcard $(D)/*.max))

##### Simulator configuration #####
SIM_NAME := $(USER)_$(basename $(TARGET))_sim
SIM_CARD := ALVEOU250

##### Compiler flags #####
INCLUDES :=	\
	src \
	include \
	$(MAXELEROSDIR)/include \
	$(MAXCOMPILERDIR)/include \
	$(MAXCOMPILERDIR)/include/slic \
	$(MAXFILE_HEADERDIR) \
	$(MAXUDPFPDIR)/include \
	$(MAXPOWERDIR)/src/maxpower/kernel/random/runtime \
	$(MAXFILE_DIRS)

INCLUDE_FLAGS := $(foreach DIR,$(INCLUDES),-I$(DIR))

# Flags that are common to C and C++ in both debug AND release mode.
COMMON_COMPILER_FLAGS := $(INCLUDE_FLAGS) -fPIC -Wall -march=native -march=native -Wno-unused-function -D_GNU_SOURCE -D__USE_MISC

# Debug and release specific compiler flags.
RELEASE_COMPILER_FLAGS := -g -O3 $(COMPVAR)
DEBUG_COMPILER_FLAGS   := -g3 -O0 $(COMPVAR)

# C and C++ specific compiler flags.
CXX_SPECIFIC_COMPILER_FLAGS := -std=c++11
C_SPECIFIC_COMPILER_FLAGS   := -std=gnu99 -fgnu89-inline

# Final compiler flags for C/C++ and debug/release combinations.
RELEASE_CXXFLAGS := $(MAXCOMPILER_INC) $(COMMON_COMPILER_FLAGS) $(CXX_SPECIFIC_COMPILER_FLAGS) $(RELEASE_COMPILER_FLAGS)
RELEASE_CFLAGS   := $(MAXCOMPILER_INC) $(COMMON_COMPILER_FLAGS) $(C_SPECIFIC_COMPILER_FLAGS) $(RELEASE_COMPILER_FLAGS)

DEBUG_CXXFLAGS := $(MAXCOMPILER_INC) $(COMMON_COMPILER_FLAGS) $(CXX_SPECIFIC_COMPILER_FLAGS) $(DEBUG_COMPILER_FLAGS)
DEBUG_CFLAGS   := $(MAXCOMPILER_INC) $(COMMON_COMPILER_FLAGS) $(C_SPECIFIC_COMPILER_FLAGS) $(DEBUG_COMPILER_FLAGS)

##### Select release or debug mode (default is release). #####
ifeq ($(strip $(MODE)),debug)
	RESULT_DIR = $(DEBUG_DIR)
	CFLAGS = $(DEBUG_CFLAGS)
	CXXFLAGS = $(DEBUG_CXXFLAGS)
else
	RESULT_DIR = $(RELEASE_DIR)
	CFLAGS = $(RELEASE_CFLAGS)
	CXXFLAGS = $(RELEASE_CXXFLAGS)
endif

BUILD_DIR := $(RESULT_DIR)/build

##### Linker flags. #####
LIBS := \
	-lpthread

STATIC_LIBS :=

LIBRARY_PATHS :=

LFLAGS := $(MAXCOMPILER_LIBS) $(LIBRARY_PATHS) $(LIBS)

##### Figure out if we are building an executable or shared-object. #####
ifeq ($(strip $(suffix $(TARGET))),.so)
LFLAGS += -shared
BIN_DIR := $(RESULT_DIR)/lib
else
BIN_DIR := $(RESULT_DIR)/bin
endif

##### Helper macros. #####

# Helper macro for src2o (see below). Converts C++ sources to object paths.
# Usage: $(call cxx2o,SOURCES)
define cxx2o
$(1:%.cpp=$(BUILD_DIR)/%.cpp.o)
endef

# Helper macro for c2o. Converts C sources to object paths.
# Usage: $(call c2o,SOURCES)
define c2o
$(1:%.c=$(BUILD_DIR)/%.c.o)
endef

# Converts C/C++ sources to object paths.
# Usage: $(call src2o,SOURCES)
define src2o
$(call cxx2o,$(filter %.cpp,$1)) $(call c2o,$(filter %.c,$1))
endef

# Converts maxfile to slic object path.
# Usage: $(call max2o,MAXFILE)
define max2o
$(1:%.max=$(BUILD_DIR)/%.max.o)
endef

##### Sources. #####
SRCS := $(shell find src -name '*.cpp') $(shell find src -name '*.c')

##### Make targets. #####

.PHONY: all
all: $(BIN_DIR)/$(TARGET)

.PHONY: help
help:
	@echo "Targets:"
	@echo "  all   [MODE=debug|release]: Build in either release or debug mode."
	@echo "  clean [MODE=debug|release]: Remove build files and binaries of either release or debug build."
	@echo "  distclean: Remove all build files and binaries."
	@echo "  start_sim: TODO document me!"
	@echo "  stop_sim:  TODO document me!"

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(BIN_DIR)

.PHONY: distclean
distclean:
	rm -rf $(DIST_DIR)

.PHONY: stop_sim
stop_sim:
	maxcompilersim -n ${SIM_NAME} stop

.PHONY: start_sim
start_sim:
	maxcompilersim -n ${SIM_NAME} -c ${SIM_CARD} restart

.PHONY: run_sim
run_sim: start_sim
	@ echo "Running $(BIN_DIR)/$(TARGET)"
	@ echo
	@ MAXELEROSDIR="$(MAXCOMPILERDIR)/lib/maxeleros-sim" \
		LD_PRELOAD="$(MAXCOMPILERDIR)/lib/maxeleros-sim/lib/libmaxeleros.so:$(LD_PRELOAD)" \
		SLIC_CONF="$(SLIC_CONF);use_simulation=$(SIM_NAME)" \
		hwloc-bind --membind node:1 --cpubind node:1 -- python /home/rakyta.peter/SQUANDER_test/test_optmization_problem_combined.py
	@ echo
	@ echo "$(BIN_DIR)/$(TARGET) Terminated successfully."
	@ echo

.PHONY: maxide_build_config
maxide_build_config:
	@ if [ -z "$(MAXIDE_BUILD_CONFIG_OUTPUT)" ]; then echo "MAXIDE_BUILD_CONFIG_OUTPUT should be set" >&2 && exit 1; fi
	@ mkdir -p $(MAXIDE_BUILD_CONFIG_OUTPUT)
	@ rm -f "$(MAXIDE_BUILD_CONFIG_OUTPUT)/CFLAGS"
	@ rm -f "$(MAXIDE_BUILD_CONFIG_OUTPUT)/CXXFLAGS"
	@ rm -f "$(MAXIDE_BUILD_CONFIG_OUTPUT)/LFLAGS"
	@ # CFLAGS
	@ for var in $(CFLAGS); do echo "$$var" >> "$(MAXIDE_BUILD_CONFIG_OUTPUT)/CFLAGS"; done
	@ # CXXFLAGS
	@ for var in $(CXXFLAGS); do echo "$$var" >> "$(MAXIDE_BUILD_CONFIG_OUTPUT)/CXXFLAGS"; done
	@ # LFLAGS + STATIC_LIBS + SHARED
	@ for var in $(LFLAGS); do echo "$$var" >> "$(MAXIDE_BUILD_CONFIG_OUTPUT)/LFLAGS"; done
	@ for var in $(STATIC_LIBS); do echo "$$var" >> "$(MAXIDE_BUILD_CONFIG_OUTPUT)/LFLAGS"; done
	@ for var in $(SHARED); do echo "$$var" >> "$(MAXIDE_BUILD_CONFIG_OUTPUT)/LFLAGS"; done

##### Make rules. #####

# Link into executable/SO.
$(BIN_DIR)/$(TARGET): $(call src2o,$(SRCS)) $(call max2o,$(MAXFILES))
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $^ $(STATIC_LIBS) $(SHARED) -o $@ $(LFLAGS)

# Include all .d files.
DEPS := $(shell [[ -d $(BUILD_DIR) ]] && find $(BUILD_DIR) -name "*.d")
-include $(DEPS)

# Compile C.
$(BUILD_DIR)/%.c.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MD -MF $(@:%.c.o=%.c.d) -c $< -o $@ $(LFLAGS)

# Compile C++.
$(BUILD_DIR)/%.cpp.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -MD -MF $(@:%.cpp.o=%.cpp.d) -c $< -o $@ $(LFLAGS)

# Compile SLiC objects.
$(BUILD_DIR)/%.max.o: %.max $(MAXFILES_CONFIG_FILE)
	@mkdir -p $(@D)
	$(SLICCOMPILE) $< $@
