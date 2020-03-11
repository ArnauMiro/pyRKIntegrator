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

#ifndef RK_H
#define RK_H

#include <string>
#include <algorithm>

/*
	RUNGE-KUTTA SCHEMES

	Schemes for the Runge-Kutta integrator.
*/
typedef enum _RK_SCHEME {
	NONE,
	// Runge-Kutta methods
	EULERHEUN12,
	BOGACKISHAMPINE23,
	DORMANDPRINCE34A,
	FEHLBERG45,
	CASHKARP45,
	DORMANDPRINCE45,
	DORMANDPRINCE45A,
	CALVO56,
	DORMANDPRINCE78,
	CURTIS810,
	HIROSHI912,
	// Runge-Kutta-Nystrom methods
	RKN34,
	RKN46,
	RKN68,
	RKN1012
}RK_SCHEME;

inline const char *RKS2str(const RK_SCHEME rks){
	switch (rks) {
		case EULERHEUN12:       return "eulerheun12";
		case BOGACKISHAMPINE23: return "bogackishampine23";
		case DORMANDPRINCE34A:  return "dormandprince34a";
		case FEHLBERG45:        return "fehlberg45";
		case CASHKARP45:        return "cashkarp45";
		case DORMANDPRINCE45:   return "dormandprince45";
		case DORMANDPRINCE45A:  return "dormandprince45a";
		case CALVO56:           return "calvo56";
		case DORMANDPRINCE78:   return "dormandprince78";
		case CURTIS810:         return "curtis810";
		case HIROSHI912:        return "hiroshi912";
		case RKN34:             return "rnk34";
		case RKN46:             return "rnk46";
		case RKN68:             return "rnk68";
		case RKN1012:           return "rnk1012";
	}
	return NULL;
}

inline RK_SCHEME str2RKS(const char *str) {
	std::string rkstr(str);
	std::transform(rkstr.begin(), rkstr.end(), rkstr.begin(), ::tolower);
	if (rkstr == "eulerheun12")       return EULERHEUN12;
	if (rkstr == "bogackishampine23") return BOGACKISHAMPINE23;
	if (rkstr == "dormandprince34a")  return DORMANDPRINCE34A;
	if (rkstr == "fehlberg45")        return FEHLBERG45;
	if (rkstr == "cashkarp45")        return CASHKARP45;
	if (rkstr == "dormandprince45")   return DORMANDPRINCE45;
	if (rkstr == "dormandprince45a")  return DORMANDPRINCE45A;
	if (rkstr == "calvo56")           return CALVO56;
	if (rkstr == "dormandprince78")   return DORMANDPRINCE78;
	if (rkstr == "curtis810")         return CURTIS810;
	if (rkstr == "hiroshi912")        return HIROSHI912;
	if (rkstr == "rkn34")             return RKN34;
	if (rkstr == "rkn46")             return RKN46;
	if (rkstr == "rkn68")             return RKN68;
	if (rkstr == "rkn1012")           return RKN1012;
	return NONE;
}

/*
	RUNGE-KUTTA PARAMETERS

	Parameters for the Runge-Kutta integrator.
*/
typedef struct _RK_PARAM {
	double h0;
	double eps;
	double epsevf;
	double minstep;
	double secfact;
	double secfact_max;
	double secfact_min;
	int (*eventfcn)(double,double*,int,double*,int*); // Event function must return continue or stop
	int (*outputfcn)(double,double*,int);             // Output function must return continue or stop
} RK_PARAM;

inline RK_PARAM rkparams(const double xspan[2]){
	RK_PARAM rkp;
	// Initial and max step based on xspan
	rkp.h0          = (xspan[1] - xspan[0])/10.;
	// Tolerance
	rkp.eps         = 1e-8;
	rkp.epsevf      = 1e-4;
	rkp.minstep     = 1.e-12;
	// Security factors
	rkp.secfact     = 0.9;
	rkp.secfact_max = 5.;
	rkp.secfact_min = 0.2;
	// No event or output function
	rkp.eventfcn    = NULL;
	rkp.outputfcn   = NULL;
	return rkp;
}

/*
	RUNGE-KUTTA OUTPUT

	Output for the Runge-Kutta integrator.
*/
typedef struct _RK_OUT {
	int retval, n;
	double err;
	double *x, *y, *dy;
} RK_OUT;

extern "C" void freerkout(const RK_OUT *rko);
void writerkout(const char *outfile, const RK_OUT *rko, const int nvars);

/*
	RUNGE KUTTA METHOD CLASS

*/
class RKMethod {
	public:
		RKMethod(const RK_SCHEME rks);
		~RKMethod();

		// Number of steps of the method
		int nstep;
		// Order of the method
		double alpha;
		// Coefficients of the Runge-Kutta method
		double *C, *A, *B, *Bhat, *Bp, *Bphat;

		int Aind(int ii, int jj) {return( this->nstep*ii + jj );}

		int CheckTableau();

	private:
		bool alloc;
		// List of implemented Runge-Kutta methods
		void EulerHeun12();
		void BogackiShampine23();
		void DormandPrince34A();
		void Fehlberg45();
		void CashKarp45();
		void DormandPrince45();
		void DormandPrince45A();
		void Calvo56();
		void DormandPrince78();
		void Curtis810();
		void Hiroshi912();

		// List of implemented Runge-Kutta-Nystrom methods
		void RungeKuttaNystrom34();
		void RungeKuttaNystrom46();
		void RungeKuttaNystrom68();
		void RungeKuttaNystrom1012();
};

/*
	FUNCTIONS
*/

extern "C" RK_OUT odeRK(const char *scheme, void (*odefun)(double,double*,int,double*),
	double xspan[2], double y0[], const int n, const RK_PARAM *rkp);
extern "C" RK_OUT odeRKN(const char *scheme, void (*odefun)(double,double*,int,double*),
	double xspan[2], double y0[], double dy0[], const int n, const RK_PARAM *rkp);

#endif
