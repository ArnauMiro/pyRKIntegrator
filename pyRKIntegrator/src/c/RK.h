/*
	RUNGE-KUTTA Integration

	Library to perform numerical integration using Runge-Kutta methods, implemented
	based on MATLAB's ode45.

	The function to call is ODERK, a generic Runge-Kutta variable step integrator.

	The inputs of this function are:
		> scheme: Runge-Kutta scheme to use. Options are:
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

	Arnau Miro, Elena Terzic 2021
	Last rev: 2021
*/

#ifdef __cplusplus
extern "C" {
#endif

/*
	RUNGE-KUTTA SCHEMES

	Schemes for the Runge-Kutta integrator.
*/
typedef enum _RK_SCHEME
{
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
} RK_SCHEME;

const char *RKS2str(const RK_SCHEME rks);
RK_SCHEME   str2RKS(const char *str);


/*
	RUNGE-KUTTA PARAMETERS

	Parameters for the Runge-Kutta integrator.
*/
typedef struct _RK_PARAM
{
    double h0;  // Initial step for the interval
    double eps; // Tolerance to meet
    double epsevf;
    double minstep;
    double secfact;
    double secfact_max;
    double secfact_min;
    int (*eventfcn)(double, double *, int, double *, int *); // Event function must return continue or stop
    int (*outputfcn)(double, double *, int);                 // Output function must return continue or stop
} RK_PARAM;

RK_PARAM rkparams(const double xspan[2]);


/*
	RUNGE-KUTTA OUTPUT

	Output for the Runge-Kutta integrator.
*/
typedef struct _RK_OUT
{
    int retval, n;
    double err;
    double *x, *y, *dy;
} RK_OUT;

void freerkout(const RK_OUT *rko);
void writerkout(const char *outfile, const RK_OUT *rko, const int nvars);

/*
	RUNGE KUTTA METHOD

	Define which Runge-Kutta scheme to use
*/
typedef struct _RKMethod
{
    // Number of steps of the method
    int nstep;
    // Order of the method
    double alpha;
    // Coefficients of the Runge-Kutta method
    double *C, *A, *B, *Bhat, *Bp, *Bphat;

    // Check whether the memory is allocated
    int alloc;
} RKMethod;

RKMethod RKMethodSelection(const RK_SCHEME rks);
void     RKMethodFree(RKMethod *rkm);
int      CheckTableau(RKMethod *rkm);

// List of implemented Runge-Kutta methods
void EulerHeun12(RKMethod *rkm);
void BogackiShampine23(RKMethod *rkm);
void DormandPrince34A(RKMethod *rkm);
void Fehlberg45(RKMethod *rkm);
void CashKarp45(RKMethod *rkm);
void DormandPrince45(RKMethod *rkm);
void DormandPrince45A(RKMethod *rkm);
void Calvo56(RKMethod *rkm);
void DormandPrince78(RKMethod *rkm);
void Curtis810(RKMethod *rkm);
void Hiroshi912(RKMethod *rkm);

// List of implemented Runge-Kutta-Nystrom methods
void RungeKuttaNystrom34(RKMethod *rkm);
void RungeKuttaNystrom46(RKMethod *rkm);
void RungeKuttaNystrom68(RKMethod *rkm);
void RungeKuttaNystrom1012(RKMethod *rkm);


/*
	FUNCTIONS

	To be called in the code
*/

RK_OUT odeRK(const char *scheme, void (*odefun)(double, double *, int, double *),
              double xspan[2], double *y0, const int n, const RK_PARAM *rkp);
RK_OUT odeRKN(const char *scheme, void (*odefun)(double, double *, int, double *),
               double xspan[2], double *y0, double *dy0, const int n, const RK_PARAM *rkp);
int check_tableau(const char *scheme);

#ifdef __cplusplus
}
#endif