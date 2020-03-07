/*
	EXAMPLE 1
	
	Using the odeRK with odeset and different schemes.

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
	RK_OUT rko = odeRK("dormandprince45",testfun,xspan,y0,n,&rkp);

	// Write results to a file
	printf("retval %d error %.2e\n",rko.retval,rko.err);
	if (rko.retval > 0) 
        writerkout("out.txt",&rko,n);

	// Finish
	freerkout(&rko);
	return 0;
}