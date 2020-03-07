/*
	RUNGE-KUTTA Integration

	Library to perform numerical integration using Runge-Kutta methods, implemented
	based on MATLAB's ode45.

	The function to call is ODERK, a generic Runge-Kutta variable step integrator.

	The inputs of this function are:
		> scheme: Runge-Kutta scheme to use. Options are:
			* Euler-Heun 1(2) (eulerheun12)
			* Bogacki-Shampine 2(3) (bogackishampine23)
			* Fehlberg 4(5) (fehlberg45)
			* Cash-Carp 4(5) (cashcarp45)
			* Dormand-Prince 4-5 (dormandprince45)
			* Calvo 5(6) (calvo56)
			* Dormand-Prince 7(8) (dormandprince78)
			* Curtis 8(10) (curtis810)
			* Hiroshi 9(12) (hiroshi912)
		> odefun: a function so that
			dydx = odefun(x,y,n)
		where n is the number of y variables to integrate.
		> xspan: integration start and end point.
		> y0: initial conditions (must be size n).
		> n: number of initial conditions.

		An additional parameter can be passed to the integrator, which contains
		parameters for the integrator:
			> h0:       Initial step for the interval
			> hmin:      Minimum interval allowed
			> eps:       Tolerance to meet.
			> eventfcn:  Event function.
			> outputfcn: Output function.

	If the inputted tolerance is not met, the intergrator will use hmin and will
	produce a successful step. Warning! Accuracy might be compromised.

	The function will return a structure containing the following information:
		> retval: Return value. Will be negative in case of errors or positive if successful.
			* 1: indicates a successful run.
			* 2: indicates that at a certain point hmin was used for the step.
		> n: Number of steps taken.
		> err: Maximum error achieved.
		> x: solution x values of size n.
		> y: solution y values of size n per each variable.

	Arnau Miro, 2018
	Last rev: 2020
*/

#include <cstdlib>
#include <cstdio>
#include <cmath>

#include "RK.h"

void testfun(double x, double *y, int n, double *dydx) {

	dydx[0] = std::cos(x) + std::sin(y[0]);
	dydx[1] = std::sin(x) + std::cos(y[1]);
}

int main() {

	// Integration bounds and initial solution
	const int n = 2;
	double xspan[2] = {0.,10.};
	double y0[n]    = {0.,0.};

	// Runge-Kutta parameters
	RK_PARAM rkp = rkparams(xspan);

	// Runge-Kutta
	RK_OUT rko = odeRK("dormandprice45",testfun,xspan,y0,n,&rkp);

	// Write results to a file
	printf("retval %d error %.2e\n",rko.retval,rko.err);
	if (rko.retval > 0) {
		FILE *myfile;
		myfile = fopen("out.txt","w");

		for(int ii=0; ii<rko.n; ii++){
			fprintf(myfile,"%f\t",rko.x[ii]);
			for (int jj=0; jj<n; jj++)
				fprintf(myfile,"%f\t",rko.y[2*ii + jj]);
			fprintf(myfile, "\n");
		}
		fclose(myfile);
	}

	// Finish
	freerkout(&rko);
	return 0;
}