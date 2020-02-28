#Compile RK integrator
#   Compile with g++ or Intel C++ Compiler (/opt/intel/bin/icpc)
#   Compile with the most aggressive optimization setting (O3)
#   Use the most pedantic compiler settings: must compile with no warnings at all
#
# Use:
#   make RK.o --> compiles only the RK subroutine into a binary object
#   make test --> Compiles all subroutines into binary objects, then compiles the test program
#   make clean --> Removes all the compiled binary objects and programs
#   make --> Compile all subroutines and programs
#
# The user may override any desired internal variable by redefining it via command-line:
#   make CXX=g++ [...]
#   make OPTL=-O2 [...]
#   make FLAGS="-Wall -g" [...]
#
#David de la Torre / Arnau Miro
#November 2018

## OPTIMIZATION LEVEL ##
# Check or run the profiling to see which optimization level
# is best for your system.
OPTL = fast

## COMPILERS ##
# Detect if the intel compilers are installed and use
# them, otherwise default to the GNU compilers
ifeq (,$(shell which icpc))
	CXX = g++
	CXXFLAGS = -O$(OPTL) -ffast-math -std=c++11#-Wall -Wextra -Werror -ansi -pedantic-errors
else
	CXX = icpc
	CXXFLAGS = -O$(OPTL) -xHost -mtune=haswell -std=c++11
endif

## OBJECTS ##

# Define all binary objects required for the main programs
OBJS = RK.o

# Detect the operating system and generate the RKLIBN
ifeq ($(shell uname -s),Darwin)
	RKLIBN = RK.dylib
else
	RKLIBN = RK.so
endif

## COMPILATION RULES ##

# Default rule (compile all programs)
ALL: RKlib test

# Compile the RK library to be used in python
RKlib: RK.cpp
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $(RKLIBN) RK.cpp

# Compile test program
test: main.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(OBJS)

# Generic rule to compile any required binary object *.o from its *.cpp source
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

# Clean-up: all binary objects and compiled programs
clean:
	@rm *.o test