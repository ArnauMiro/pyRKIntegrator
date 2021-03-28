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
    "fehlberg45",
    "cashkarp45",
    "dormandprince45",
    "dormandprince45a",
    "calvo56",
    "dormandprince78",
    "curtis810",
    "hiroshi912"
};

// Parameters
const double G  = 6.674e-11*1.e-9; // km3/kg s2
const double M1 = 5.97219e24;      // kg
const double M2 = 1.9891e30;       // kg

const double R0 = 1.5e8;           // km
const double V0 = sqrt(G*M2/R0);
const double T  = 2*PI / sqrt(G*M2/R0/R0/R0);

// Define the function to integrate
void TwoBody(double t, double *var, int n, double *varp) {
/*
	Body 1: Perturbated
		var[0] = rx   var[3] = vx
		var[1] = ry   var[4] = vy
		var[2] = rz   var[5] = vz
	Body 2: Perturber
		var[6] = rx   var[9]  = vx
		var[7] = ry   var[10] = vy
		var[8] = rz   var[11] = vz
*/
	double r = sqrt( (var[6]-var[0])*(var[6]-var[0]) + 
                     (var[7]-var[1])*(var[7]-var[1]) + 
                     (var[8]-var[2])*(var[8]-var[2]) 
                   );

	// Perturbated
	varp[0] = var[3];
	varp[1] = var[4];
	varp[2] = var[5];
	varp[3] = ( G*M2/r/r/r ) * (var[6]-var[0]);
	varp[4] = ( G*M2/r/r/r ) * (var[7]-var[1]);
	varp[5] = ( G*M2/r/r/r ) * (var[8]-var[2]);

	// Perturber
	varp[6]  = var[9];
	varp[7]  = var[10];
	varp[8]  = var[11];
	varp[9]  = ( G*M1/r/r/r ) * (var[0]-var[6]);
	varp[10] = ( G*M1/r/r/r ) * (var[1]-var[7]);
	varp[11] = ( G*M1/r/r/r ) * (var[2]-var[8]);
}

int main() {
	// Integration bounds and initial solution
	const int n = 12;
	double tspan[2] = {0.,2.*T};
	double y0[n]    = {R0,0.,0.,0.,V0,0.,0.,0.,0.,0.,0.,0.};

	// Runge-Kutta parameters
	RK_PARAM rkp = rkparams(tspan);
    rkp.h0  = 24.*3600.;
    rkp.eps = 1e-6;

	// Runge-Kutta
    for (std::string scheme : schemes) {
        std::printf("scheme %s, ",scheme.c_str());
		// Launch the integrator
	    RK_OUT rko = odeRK(scheme.c_str(),TwoBody,tspan,y0,n,&rkp);
	    // Finish
		std::printf("error = %.2e with %d steps\n",rko.err,rko.n);
	    freerkout(&rko);
    }

	return 0;
}