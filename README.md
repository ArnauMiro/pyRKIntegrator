[![Build status](https://github.com/ArnauMiro/pyRKIntegrator/actions/workflows/build_python.yml/badge.svg)](https://github.com/ArnauMiro/pyRKIntegrator/actions)
[![Build status](https://github.com/ArnauMiro/pyRKIntegrator/actions/workflows/build_gcc.yml/badge.svg)](https://github.com/ArnauMiro/pyRKIntegrator/actions)
[![Build status](https://github.com/ArnauMiro/pyRKIntegrator/actions/workflows/build_intel.yml/badge.svg)](https://github.com/ArnauMiro/pyRKIntegrator/actions)
[![License](https://img.shields.io/badge/license-GPL3-orange)](https://opensource.org/license/gpl-3-0/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10598094.svg)](https://doi.org/10.5281/zenodo.10598094)


# Runge-Kutta and Runge-Kutta-Nystrom integrators

This project contains a series of adaptive generic Runge-Kutta and Runge-Kutta-Nystrom integrators developed in C++, with a Python interface.

The code has been optimized under AVX/AVX2 vectorization directives using the Intel(r) C++ compiler (icpc). Although this is the **default** compilation mode, it is not strictly required as the makefile will automatically default to GNU if the Intel compilers are not found.

The default optimization mode is *fast* although it can be changed using the variable **OPTL**, e.g.,
```bash
make OPTL=2
```

## Deployment

A _Makefile_ is provided within the tool to automate the installation for easiness of use for the user. To install the tool simply create a virtual environment as stated below or use the system Python. Once this is done simply type:
```bash
make
```
This will install all the requirements and install the package to your active python. To uninstall simply use
```bash
make uninstall
```

The previous operations can be done one step at a time using
```bash
make requirements
```
to install all the requirements;
```bash
make python
```
to compile and;
```bash
make install
```
to install the tool.

### Virtual environment

The package can be installed in a Python virtual environement to avoid messing with the system Python installation.
Next, we will use [Conda](https://docs.conda.io/projects/conda/en/latest/index.html) for this purpose.
Assuming that Conda is already installed, we can create a virtual environment with a specific python version and name (`my_env`) using
```bash
conda create -n my_env python=3.8
```
The environment is placed in `~/.conda/envs/my_env`.
Next we activate it be able to install packages using `conda` itself or another Python package manager in the environment directory:
```bash
conda activate my_env
```
Then just follow the instructions as stated above.

## Runge-Kutta methods

The Runge-Kutta methods are a family of numerical integrators for ODEs. They require a function so that
> dydx = f(x,y)

The C++ API requires defining a the function to integrate as
```C++
void odefun(double x, double y[], int n, double dydx[])
```
where *y* and *dydx* have *n* components already allocated. The Runge-Kutta integrator is called as follows
```c++
RK_PARAM rkp = rkdefaults(xspan);             // Standard Runge-Kutta parameters
RK_OUT   rko = odeRK(testfun,xspan,y0,n,rkp); // Runge-Kutta integrator
```
where *rkp* contains the ODE integration parameters (see **RK.h**) and *rko* is the output structure containing:
*	A return value, *rko.retval*, that is > 0 in case of success or < 0 in case of failue. A value of 0 indicates that the *odeRK* routine has not been run.
*	The number of integration steps *rko.n*.
*	An error value *rko.err*.
*	The solution, *rko.x* and *rko.y*.
The available Runge-Kutta schemes are:
*	Euler-Heun 1(2) (eulerheun12)
*	Bogacki-Shampine 2(3) (bogackishampine23)
*	Fehlberg 4(5) (fehlberg45)
*	Cash-Karp 4(5) (cashkarp45)
*	Dormand-Prince 4-5 (dormandprince45)
*	Calvo 5(6) (calvo56)
*	Dormand-Prince 7(8) (dormandprince78)
*	Curtis 8(10) (curtis810)
*	Hiroshi 9(12) (hiroshi912)

## Runge-Kutta-Nystrom methods

The Runge-Kutta-Nystrom methods are a family of numerical integrators for smooth second order ODEs. They require a function so that
> dy2dx2 = f(x,y)

The C++ API requires defining a the function to integrate as
```C++
void odefun(double x, double y[], int n, double dy2dx2[])
```
where *rkp* contains the ODE integration parameters (see **RK.h**) and *rko* is the output structure containing:
*	A return value, *rko.retval*, that is > 0 in case of success or < 0 in case of failue. A value of 0 indicates that the *odeRK* routine has not been run.
*	The number of integration steps *rko.n*.
*	An error value *rko.err*.
*	The solution, *rko.x*, *rko.y* and *rko.dy*.
The available Runge-Kutta-Nystrom schemes are:
*	Runge-Kutta-Nystrom 3(4) (rkn34)
*	Runge-Kutta-Nystrom 4(6) (rkn46)
*	Runge-Kutta-Nystrom 6(8) (rkn68)
*	Runge-Kutta-Nystrom 10(12) (rkn1012)

## C and C++ implementations

There is a C and a C++ implementation of the algorithms. The language can be chosen in the **Makefile** by setting the *USE_CPP* variable. The compilation of the examples and the python tools will follow.
```bash
make USE_CPP=ON/OFF
```

## The Python interface

The Python interface allows using the Runge-Kutta and Runge-Kutta-Nystrom integrators using Python's cython. The wrapper can be compiled using
```bash
make python
```
Note that this will create a **.so** that must be in the same directory of **RungeKutta.py**. Otherwise, the Python interface is accessed without the speedup of compiled code.

In Python, the Runge-Kutta module should be first using
```python
import pyRKIntegrator as rk
```
The function to integrate must be defined as
```python
def odefun(x,y,n,dydx):  # For Runge-Kutta
def odefun(x,y,n,dy2dx): # For Runge-Kutta-Nystrom
```
where *y* and *dydx* or *dy2dx* are vectors that have already been allocated (of size *n*). Access to the parameters structure is provided by the **odeset** class:
```python
params = rk.odeset()  # For Runge-Kutta and Runge-Kutta-Nystrom
```
This will initiate using default parameters, which can be changed by using key arguments or by modifying the class fields. The access to the integratos is:
```python
x,y,err    = rk.odeRK(odefun,xspan,y0,odeset)      # For Runge-Kutta
x,y,dy,err = rk.odeRKN(odefun,xspan,y0,dy0,odeset) # For Runge-Kutta-Nystrom
```
Examples 1 to 4 include examples of advanced features to use with the integrator.

## Event and output functions

Event detection and output functions are a characteristic of this set of integrators. They can be set using the *RK_PARAM* structure in C++
```c++
RK_PARAM rkp;
rkp.eventfcn = eventfun;
rkp.outputfcn = outputfun;
```
or the *odeset* class in Python
```python
odeset = rk.odeset(eventfun=eventfun,outputfun=outputfun)
```

### Output functions

Output functions allow retrieving the integrated values right after a successful integration step. A value of either *1* or *0* must be returned to continue or stop the integration. The implementation in C++ is
```c++
int outputfun(double x, double y[], int n) {
	/* Some code */
	return 1; // or 0 to stop integration
}
```
and in Python
```python
def outputfun(x,y,n):
	# Some code
	return 1 # or 0 to stop integration
```

### Event functions

Event functions allow for solving a problem so that 
> g(x) - val = 0

using a root solving algorithm. This is useful, for example, to detect when the integrator reaches a certain value (see **example2.py** for a more comprehensive usage). The implementation in C++ is
```c++
int eventfun(double x, double y[], int n, double value[1],int direction[1]) {
	/* Some code */
	return 1; // or 0 to stop integration
}
```
and in Python
```python
def eventfun(x,y,n,value,direction):
	# Some code
	return 1 # or 0 to stop integration
```
Where
*	*value* is a mathematical expression describing the event. An event occurs when value(i) is equal to zero.
*	*direction*
    -	0 if all zeros are to be located.
    -	+1 locates only zeros where the event function is increasing.
    -	-1 locates only zeros where the event function is decreasing.
Regarding *direction*, only 0 is implemented.