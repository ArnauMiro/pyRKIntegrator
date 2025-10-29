#!/usr/bin/env cpython
#
# RUNGE-KUTTA Integration.
#
# Exporting of compiled functions.
#cython: legacy_implicit_noexcept=True


# Expose functions from the RK cpp header
cdef extern from "RK.h":
	# Typedef for RK parameters
	ctypedef struct RK_PARAM:
		double h0
		double eps
		double epsevf
		double minstep
		double secfact
		double secfact_max
		double secfact_min
		int    (*eventfcn)(double,double*,int,double*,int*) noexcept # Event function must return continue or stop
		int    (*outputfcn)(double,double*,int) noexcept             # Output function must return continue or stop
	# Typedef for RK output
	ctypedef struct RK_OUT:
		int    retval
		int    n
		double err
		double *x
		double *y
		double *dy
	# Expose the Runge-Kutta basic integrator
	cdef RK_OUT c_odeRK "odeRK" (const char *scheme, void (*odefun)(double,double*,int,double*),
		double xspan[2], double y0[], const int n, const RK_PARAM *rkp)
	# Expose the Runge-Kutta-Nystrom integrator
	cdef RK_OUT c_odeRKN "odeRKN"(const char *scheme, void (*odefun)(double,double*,int,double*),
		double xspan[2], double y0[], double dy0[], const int n, const RK_PARAM *rkp);
	# Expose the free rkout
	cdef void freerkout(const RK_OUT *rko)
 	# Expose the check tableau
	cdef int check_tableau(const char *scheme)