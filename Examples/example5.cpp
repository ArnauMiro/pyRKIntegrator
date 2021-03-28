/*
	EXAMPLE 3
	
	Two-Body problem.

	Arnau Miro, 2018
	Last rev: 2020
*/

#include <cstdlib>
#include <cstdio>
#include <cmath>

#include <string>
#include <vector>

#include "RK.h"


#define PI 4.*atan(1)

// Schemes to test
std::vector<std::string> schemes = {
    "rkn34",
    "rkn46",
    "rkn68",
    "rkn1012"
};

// Parameters
const double G  = 6.674e-11*1.e-9; // km3/kg s2
const double M1 = 5.97219e24;      // kg
const double M2 = 1.9891e30;       // kg

const double R0 = 1.5e8;           // km
const double V0 = sqrt(G*M2/R0);
const double T  = 2*PI / sqrt(G*M2/R0/R0/R0);

// Define the function to integrate
void TwoBody(double t, double *var, int n, double *varp2) {
/*
	Body 1: Perturbated
		var[0] = rx   varp[0] = vx    varp2[0] = ax
		var[1] = ry   varp[1] = vy    varp2[1] = ay
		var[2] = rz   varp[2] = vz    varp2[2] = az
	Body 2: Perturber
		var[3] = rx   varp[3] = vx    varp2[3] = ax
		var[4] = ry   varp[4] = vy    varp2[4] = ay
		var[5] = rz   varp[5] = vz    varp2[5] = az
*/
	double r = sqrt( (var[3]-var[0])*(var[3]-var[0]) + 
                     (var[4]-var[1])*(var[4]-var[1]) + 
                     (var[5]-var[2])*(var[5]-var[2]) 
                   );
	// Perturbated
	varp2[0] = ( G*M2/r/r/r ) * (var[3]-var[0]);
	varp2[1] = ( G*M2/r/r/r ) * (var[4]-var[1]);
	varp2[2] = ( G*M2/r/r/r ) * (var[5]-var[2]);

	// Perturber
	varp2[3] = ( G*M1/r/r/r ) * (var[0]-var[3]);
	varp2[4] = ( G*M1/r/r/r ) * (var[1]-var[4]);
	varp2[5] = ( G*M1/r/r/r ) * (var[2]-var[5]);
}

int main() {
	// Integration bounds and initial solution
	const int n = 6;
	double tspan[2] = {0.,10.*T};
	double y0[n]    = {R0,0.,0.,0.,0.,0.};
	double dy0[n]   = {0.,V0,0.,0.,0.,0.};

	// Runge-Kutta parameters
	RK_PARAM rkp = rkparams(tspan);
    rkp.h0  = 24.*3600.;
    rkp.eps = 1e-6;

	// Runge-Kutta
    for (std::string scheme : schemes) {
        std::printf("scheme %s, ",scheme.c_str());
		// Launch the integrator
	    RK_OUT rko = odeRKN(scheme.c_str(),TwoBody,tspan,y0,dy0,n,&rkp);
	    // Finish
		std::printf("error = %.2e with %d steps\n",rko.err,rko.n);
	    freerkout(&rko);
    }

	return 0;
}