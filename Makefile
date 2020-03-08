# Compile RK integrator
#   Compile with g++ or Intel C++ Compiler (/opt/intel/bin/icpc)
#   Compile with the most aggressive optimization setting (O3)
#   Use the most pedantic compiler settings: must compile with no warnings at all
#
# The user may override any desired internal variable by redefining it via command-line:
#   make CXX=g++ [...]
#   make OPTL=-O2 [...]
#   make FLAGS="-Wall -g" [...]
#
# David de la Torre / Arnau Miro
# November 2018

# Optimization, host and CPU type
#
OPTL = 2
HOST = Host
TUNE = haswell

# Options
#
VECTORIZATION  = ON
OPENMP_PARALL  = OFF
FORCE_GCC      = OFF
DEBUGGING      = OFF

# Paths to the installed binaries
#
INSTALL_PATH = $(shell pwd)
BIN_PATH     = $(INSTALL_PATH)/bin
LIBS_PATH    = $(INSTALL_PATH)/lib
INC_PATH     = $(INSTALL_PATH)/include


# Compilers
#
# Automatically detect if the intel compilers are installed and use
# them, otherwise default to the GNU compilers
ifeq ($(FORCE_GCC),ON) 
	# Forcing the use of GCC
	# C Compiler
	CC = gcc
	# C++ Compiler
	CXX = g++
	# Fortran Compiler
	FC = gfortran
else
	ifeq (,$(shell which icc))
		# C Compiler
		CC = gcc
		# C++ Compiler
		CXX = g++
		# Fortran Compiler
		FC = gfortran
	else
		# C Compiler
		CC = icc
		# C++ Compiler
		CXX = icpc
		# Fortran Compiler
		FC = ifort
	endif
endif


# Compiler flags
#
ifeq ($(CC),gcc)
	# Using GCC as a compiler
	ifeq ($(DEBUGGING),ON)
		# Debugging flags
		CFLAGS   += -O0 -g -rdynamic -fPIC
		CXXFLAGS += -O0 -g -rdynamic -fPIC
		FFLAGS   += -O0 -g -rdynamic -fPIC
	else
		CFLAGS   += -O$(OPTL) -ffast-math -fPIC
		CXXFLAGS += -O$(OPTL) -ffast-math -fPIC
		FFLAGS   += -O$(OPTL) -ffast-math -fPIC
	endif
	# Vectorization flags
	ifeq ($(VECTORIZATION),ON)
		CFLAGS   += -march=native -ftree-vectorize
		CXXFLAGS += -march=native -ftree-vectorize
		FFLAGS   += -march=native -ftree-vectorize
	endif
	# OpenMP flag
	ifeq ($(OPENMP_PARALL),ON)
		CFLAGS   += -fopenmp
		CXXFLAGS += -fopenmp
	endif
else
	# Using INTEL as a compiler
	ifeq ($(DEBUGGING),ON)
		# Debugging flags
		CFLAGS   += -O0 -g -traceback -fPIC
		CXXFLAGS += -O0 -g -traceback -fPIC
		FFLAGS   += -O0 -g -traceback -fPIC
	else
		CFLAGS   += -O$(OPTL) -fPIC
		CXXFLAGS += -O$(OPTL) -fPIC
		FFLAGS   += -O$(OPTL) -fPIC
	endif
	# Vectorization flags
	ifeq ($(VECTORIZATION),ON)
		CFLAGS   += -x$(HOST) -mtune=$(TUNE)
		CXXFLAGS += -x$(HOST) -mtune=$(TUNE)
		FFLAGS   += -x$(HOST) -mtune=$(TUNE)
	endif
	# OpenMP flag
	ifeq ($(OPENMP_PARALL),ON)
		CFLAGS   += -qopenmp
		CXXFLAGS += -qopenmp
	endif
endif
# C++ standard
CXXFLAGS += -std=c++11
# Header includes
CXXFLAGS += -I${INC_PATH}


# Defines
#
DFLAGS =


# Paths to the various libraries to compile
#
RK_PATH   = src/RK
RK_OBJS   = $(patsubst %.cpp,%.o,$(wildcard $(RK_PATH)/*.cpp))
RK_INCL   = $(wildcard $(RK_PATH)/*.h)
PYTH_PATH = src/Python


# Targets
#
# Detect the operating system and generate the LIBRK
ifeq ($(shell uname -s),Darwin)
	LIBRK = libRK.dylib
else
	LIBRK = libRK.so
endif

# One rule to compile them all, one rule to find them,
# One rule to bring them all and in the compiler link them.
all: paths libs examples python
	@echo ""
	@echo "CFD tools deployed successfully"

# Paths
#
paths:
	@mkdir -p $(BIN_PATH) $(LIBS_PATH) $(INC_PATH)

# Examples
#
examples: example1 example2 example3 example4
	@echo ""
	@echo "Examples compiled successfully"
example1: src/example1.o
	$(CXX) $(CXXFLAGS) -o $@ $< -lRK -L$(LIBS_PATH)
	@mv $@ $(BIN_PATH)
example2: src/example2.o
	$(CXX) $(CXXFLAGS) -o $@ $< -lRK -L$(LIBS_PATH)
	@mv $@ $(BIN_PATH)
example3: src/example3.o
	$(CXX) $(CXXFLAGS) -o $@ $< -lRK -L$(LIBS_PATH)
	@mv $@ $(BIN_PATH)
example4: src/example4.o
	$(CXX) $(CXXFLAGS) -o $@ $< -lRK -L$(LIBS_PATH)
	@mv $@ $(BIN_PATH)

# Libraries
#
libs: includes staticlibs
	@echo ""
	@echo "Libraries compiled successfully"
staticlibs: libRK.a

sharedlibs: $(LIBRK)

libRK.a: $(RK_OBJS)
	ar rc $@ $(RK_OBJS)
	@mv $@ $(LIBS_PATH)

$(LIBRK): $(RK_OBJS)
	$(CXX) $(CXXFLAGS) -shared -Wl,-soname,$@ -o $@ $< $(DFLAGS)
	@mv $@ $(LIBS_PATH)

includes: 
	@cp $(RK_INCL) $(INC_PATH)

# Python
#
python: pypackages
	@echo ""
	@echo "Python programs deployed successfully"
pypackages: pyRK
	-@mkdir -p ${LIBS_PATH}/site-packages
	@echo ""
	@echo "Python libraries installed in <${LIBS_PATH}/site-packages>"
	@echo "You might want to add this folder to PYTHONPATH"
pyRK: includes $(LIBRK)
	@rm -rf ${LIBS_PATH}/site-packages/RungeKutta
	@rsync -rupE $(PYTH_PATH)/RungeKutta ${LIBS_PATH}/site-packages/
	@mv ${LIBS_PATH}/$(LIBRK) ${LIBS_PATH}/site-packages/RungeKutta/

# Generic object makers
#
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $< $(DFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(DFLAGS)

%.o: %.f
	$(FC) $(FFLAGS) -c -o $@ $< $(DFLAGS)

%.o: %.f90
	$(FC) $(FFLAGS) -c -o $@ $< $(DFLAGS)


# Clean
#
clean:
	-@rm -f *.o $(wildcard **/*.o) $(wildcard **/*/*.o)
cleanall:  clean 
	-@rm -rf $(wildcard $(INC_PATH)/*)
	-@rm -rf $(wildcard $(LIBS_PATH)/*)
	-@rm -rf $(wildcard $(BIN_PATH)/*)
uninstall: cleanall
	-@rm -rf $(BIN_PATH)
	-@rm -rf $(LIBS_PATH)
	-@rm -rf $(INC_PATH)
