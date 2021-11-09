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

# Include user-defined build configuration file
include options.cfg

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
		CFLAGS   += -O$(OPTL) -fPIC
		CXXFLAGS += -O$(OPTL) -fPIC
		FFLAGS   += -O$(OPTL) -fPIC
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
DFLAGS = -DNPY_NO_DEPRECATED_API

# Paths to the various libraries to compile
#
ifeq ($(USE_CPP),ON)
	RK_PATH   = pyRKIntegrator/src/cpp
	RK_OBJS   = $(patsubst %.cpp,%.o,$(wildcard $(RK_PATH)/*.cpp))
else
	RK_PATH   = pyRKIntegrator/src/c
	RK_OBJS   = $(patsubst %.c,%.o,$(wildcard $(RK_PATH)/*.c))
endif
RK_INCL   = $(wildcard $(RK_PATH)/*.h)


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
all: paths libs examples requirements python install
	@echo ""
	@echo "pyRKIntegrator deployed successfully"

# Paths
#
paths:
	@mkdir -p $(BIN_PATH) $(LIBS_PATH) $(INC_PATH)

# Examples
#
examples: example1 example2 example3 example4 example5
	@echo ""
	@echo "Examples compiled successfully"
example1: Examples/example1.o
	$(CXX) $(CXXFLAGS) -o $@ $< -lRK -L$(LIBS_PATH)
	@mv $@ $(BIN_PATH)
example2: Examples/example2.o
	$(CXX) $(CXXFLAGS) -o $@ $< -lRK -L$(LIBS_PATH)
	@mv $@ $(BIN_PATH)
example3: Examples/example3.o
	$(CXX) $(CXXFLAGS) -o $@ $< -lRK -L$(LIBS_PATH)
	@mv $@ $(BIN_PATH)
example4: Examples/example4.o
	$(CXX) $(CXXFLAGS) -o $@ $< -lRK -L$(LIBS_PATH)
	@mv $@ $(BIN_PATH)
example5: Examples/example5.o
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
python: setup.py
	@CFLAGS="${CFLAGS}" CXXFLAGS="${CXXFLAGS}" LDSHARED="${CC} -shared" ${PYTHON} $< build_ext --inplace
	@echo "Python programs deployed successfully"

requirements: requirements.txt
	@${PIP} install -r $<

install: 
	@CFLAGS="${CFLAGS}" CXXFLAGS="${CXXFLAGS}" LDSHARED="${CC} -shared" ${PIP} install .

install_dev: 
	@CFLAGS="${CFLAGS}" CXXFLAGS="${CXXFLAGS}" LDSHARED="${CC} -shared" ${PIP} install -e .

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
	-@cd pyRKIntegrator; rm -f *.o $(wildcard **/*.o) $(wildcard **/*/*.o) $(wildcard **/*/*/*.o)
	-@cd pyRKIntegrator; rm -f *.pyc $(wildcard **/*.pyc) $(wildcard **/*/*.pyc)
	-@cd pyRKIntegrator; rm -f RungeKutta.c RungeKutta.cpp RungeKutta.html
	-@rm -rf build __pycache__
cleanall:  clean 
	-@rm -rf $(wildcard $(INC_PATH)/*)
	-@rm -rf $(wildcard $(LIBS_PATH)/*)
	-@rm -rf $(wildcard $(BIN_PATH)/*)
uninstall: cleanall
	-@cd pyRKIntegrator; rm *.so
	-@rm -rf $(BIN_PATH)
	-@rm -rf $(LIBS_PATH)
	-@rm -rf $(INC_PATH)
	-@${PIP} uninstall pyRKIntegrator
	-@rm -rf pyRKIntegrator.egg-info