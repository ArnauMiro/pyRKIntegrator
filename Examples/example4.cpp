/*
	EXAMPLE 4
	
	Using the odeRKN with odeset and different schemes.

	Arnau Miro, 2018
	Last rev: 2020
*/

#include <cstdlib>
#include <cstdio>
#include <cmath>

#include <string>
#include <vector>

#ifdef USE_C
extern "C" {
	#include "RK.h"
}
#else
	#include "RK.h"
#endif

// Schemes to test
std::vector<std::string> schemes = {
    "rkn34",
    "rkn46",
    "rkn68",
    "rkn1012"
};

void testfun(double x, double *y, int n, double *dy2dx) {

	dy2dx[0] = -std::sin(x) + std::cos(y[0]);
	dy2dx[1] = std::cos(x) - std::sin(y[1]);
}

int main() {

	// Integration bounds and initial solution
	const int n = 2;
	double xspan[2] = {0.,10.};
	double y0[n]    = {0.,0.};
	double dy0[n]   = {1.,1.};

	// Runge-Kutta parameters
	RK_PARAM rkp = rkparams(xspan);

	// Runge-Kutta
    for (std::string scheme : schemes) {
        std::printf("scheme %s, ",scheme.c_str());
		// Launch the integrator
		RK_OUT rko = odeRKN(scheme.c_str(),testfun,xspan,y0,dy0,n,&rkp);
	    // Finish
		std::printf("error = %.2e with %d steps\n",rko.err,rko.n);
	    freerkout(&rko);
    }

	return 0;
}