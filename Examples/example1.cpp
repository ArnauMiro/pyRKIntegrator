/*
	EXAMPLE 1
	
	Using the odeRK with odeset and different schemes.

	Arnau Miro, 2018
	Last rev: 2020
*/

#include <cstdlib>
#include <cstdio>
#include <cmath>

#include <string>
#include <vector>

#include "RK.h"

// Schemes to test
std::vector<std::string> schemes = {
	"eulerheun12",
	"bogackishampine23",
	"dormandprince34a",	
    "fehlberg45",
    "cashkarp45",
    "dormandprince45",
    "dormandprince45a",
    "calvo56",
    "dormandprince78",
    "curtis810",
    "hiroshi912"
};

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
    for (std::string scheme : schemes) {
        std::printf("scheme %s, ",scheme.c_str());
		// Launch the integrator
		RK_OUT rko = odeRK(scheme.c_str(),testfun,xspan,y0,n,&rkp);
	    // Finish
		std::printf("error = %.2e with %d steps\n",rko.err,rko.n);
	    freerkout(&rko);
    }

	return 0;
}