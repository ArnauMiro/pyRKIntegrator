/*
	RUNGE-KUTTA Integration

	Library to perform numerical integration using Runge-Kutta methods, implemented
	based on MATLAB's ode45.

	The function to call is ODERK, a generic Runge-Kutta variable step integrator.

	The inputs of this function are:
		> odefun: a function so that
			dydx = odefun(x,y,n)
		where n is the number of y variables to integrate.
		> xspan: integration start and end point.
		> y0: initial conditions (must be size n).
		> n: number of initial conditions.

		An additional parameter can be passed to the integrator, which contains
		parameters for the integrator:
			> h0:     Initial step for the interval
			> eps:    Tolerance to meet.
			> scheme: Runge-Kutta scheme to use. Options are:
				* Euler-Heun 1(2) (eulerheun12)
				* Bogacki-Shampine 2(3) (bogackishampine23)
				* Fehlberg 4(5) (fehlberg45)
				* Cash-Karp 4(5) (cashkarp45)
				* Dormand-Prince 4-5 (dormandprince45)
				* Calvo 5(6) (calvo56)
				* Dormand-Prince 7(8) (dormandprince78)
				* Curtis 8(10) (curtis810)
				* Hiroshi 9(12) (hiroshi912)
			> eventfcn: Event function.
			> outputfcn: Output function.

	The function will return a structure containing the following information:
		> retval: Return value. Will be negative in case of errors or positive if successful.
			* 1: indicates a successful run.
		> n: Number of steps taken.
		> err: Maximum error achieved.
		> x: solution x values of size n.
		> y: solution y values of size n per each variable.

	Arnau Miro, 2018
*/

#ifndef RK_H
#define RK_H

#include <vector>

/*
	TYPES
*/
typedef struct _RK_PARAM {
	char scheme[256];
	double h0;
	double eps;
	double epsevf;
	double minstep;
	int (*eventfcn)(double,double*,int,double*,int*); // Event function must return continue or stop
	int (*outputfcn)(double,double*,int);             // Output function must return continue or stop
} RK_PARAM;

typedef struct _RK_OUT {
	int retval, n;
	double err;
	double *x, *y, *dy;
} RK_OUT;

/*
	RUNGE KUTTA METHOD CLASS

*/
class RKMethod {
	public:
		RKMethod(const char scheme[]);
		~RKMethod();

		// Number of steps of the method
		int nstep;
		// Order of the method
		double alpha;
		// Coefficients of the Runge-Kutta method
		std::vector<double> C, A, B, Bhat, Bp, Bphat;

		int Aind(int ii, int jj) {return( this->nstep*ii + jj );}

		int CheckTableau();

	private:
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

extern "C" RK_OUT odeRK(void (*odefun)(double,double*,int,double*),
	double xspan[2], double y0[], const int n, RK_PARAM rkp);
extern "C" RK_OUT odeRKN(void (*odefun)(double,double*,int,double*),
	double xspan[2], double y0[], double dy0[], const int n, RK_PARAM rkp);
extern "C" void freerkout(RK_OUT rko);
RK_PARAM rkdefaults(double xspan[2]);
RK_PARAM rkndefaults(double xspan[2]);

#endif
