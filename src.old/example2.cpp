/*
	EXAMPLE 2
	
	Using the odeRK with parameters and output/event functions.

	Arnau Miro, 2018
	Last rev: 2020
*/

#include <cstdlib>
#include <cstdio>
#include <cmath>

#include "RK.h"

// Parameters
const double g    = 9.81;
const double k    = 1.;
const double m    = 10.;
const double tend = 10.;
const double h0   = 100.;
const double v0   = 0.;

// Function to integrate
void eqnfreefall(double t, double *y, int n, double *dydx) {
/*
	t    = time
	y[0] = velocity
	y[1] = altitude
*/
	double v = y[0];
	double h = y[1];
	dydx[0] = - g - (k/m)*v;
	dydx[1] = v;
}

// Define an output function to print the value
// of velocity and position at each successful timestep
int printvalues(double t, double *y, int n) {
/*
	Output function. Must be as:

	outputfcn(x,y,n)

	Where:
		> x: integration instant
		> y: variables (of size n)
		> n: number of variables

	Must return:
		> 1: continue
		> 0: stop
*/
	double v = y[0];
	double h = y[1];
	std::printf("time %.2f s, h = %.2f, v = %.2f\n",t,h,v);
	return 1;
}

// Define a function to stop the integration once a
// certain criteria is met
int stoponzero(double x, double *y, int n, double *value, int *direction) {
/*
	Event function. Must be as:

	eventfcn(x,y,n,value,direction)

	Where:
		> x: integration instant
		> y: variables (of size n)
		> n: number of variables

		> value: is a mathematical expression describing the event. 
				 An event occurs when value(i) is equal to zero.
		> direction: 0 if all zeros are to be located.
					+1 locates only zeros where the event function is increasing.
					-1 locates only zeros where the event function is decreasing.

	Must return:
		> 1: continue
		> 0: stop
*/
	value[0] = y[1];
	return 0;
}

int main() {

	// Set span and initial solution
	const int n     = 2;
	double tspan[2] = {0.,tend};
	double y0[n]    = {v0,h0};

    // Use Runge-Kutta parameters to generate 
    // the integrator parameters structure
	RK_PARAM rkp = rkparams(tspan); // set defaults for h0 and hmin
    rkp.eventfcn  = stoponzero;
    rkp.outputfcn = printvalues;

    // Launch the integrator
    std::printf("Free-fall:\n");
    RK_OUT rko = odeRK("dormandprince45",eqnfreefall,tspan,y0,n,&rkp);
    std::printf("Integration error: %.2e\n",rko.err);

	// Write results to a file
	if (rko.retval > 0) 
        writerkout("out.txt",&rko,n);

	// Finish
	freerkout(&rko);
	return 0;
}