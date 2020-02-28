#!/bin/env python
#
# Example 3: Two-Body problem
#
# Arnau Miro, 2018

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import RungeKutta as rk

# Parameters
G  = 6.674e-11*1.e-9 # km3/kg s2
M1 = 5.97219e24      # kg
M2 = 1.9891e30       # kg

R0 = 1.5e8 # km
V0 = np.sqrt(G*M2/R0)
T = 2*np.pi / np.sqrt(G*M2/R0/R0/R0)

# Define the function to integrate
def TwoBody(t,var,n,varp):
	'''
	Body 1: Perturbated
		var[0] = rx   var[3] = vx
		var[1] = ry   var[4] = vy
		var[2] = rz   var[5] = vz
	Body 2: Perturber
		var[6] = rx   var[9]  = vx
		var[7] = ry   var[10] = vy
		var[8] = rz   var[11] = vz
	'''

	r = np.sqrt( (var[6]-var[0])*(var[6]-var[0]) + \
		         (var[7]-var[1])*(var[7]-var[1]) + \
		         (var[8]-var[2])*(var[8]-var[2]) )

	# Perturbated
	varp[0] = var[3]
	varp[1] = var[4]
	varp[2] = var[5]
	varp[3] = ( G*M2/r/r/r ) * (var[6]-var[0])
	varp[4] = ( G*M2/r/r/r ) * (var[7]-var[1])
	varp[5] = ( G*M2/r/r/r ) * (var[8]-var[2])

	# Perturber
	varp[6]  = var[9]
	varp[7]  = var[10]
	varp[8]  = var[11]
	varp[9]  = ( G*M1/r/r/r ) * (var[0]-var[6])
	varp[10] = ( G*M1/r/r/r ) * (var[1]-var[7])
	varp[11] = ( G*M1/r/r/r ) * (var[2]-var[8])

# Define an output function to print the time
def PrintTime(t,y,n):
	'''
	'''
	# Convert to years and days
	year = 0.
	while t > T:
		t -= T
		year += 1.
	print "Time: %.0f years %.2f days" % (year,t/24/3600)
	return 1

# Set span and initial solution
tspan = np.array([0., 10.*T], np.double)
y0 = np.array([R0,0,0,0,V0,0,0,0,0,0,0,0], np.double)

# Generate odeset structure
odeset = rk.odeset(h0=24*3600,eps=1e-14)#,outputfcn=PrintTime)

# Create plot figures
plt.figure(num=1,figsize=(8,6),dpi=100,facecolor='w',edgecolor='k')
ax1 = plt.gca(projection='3d')
ax1.set_title('Orbit')
plt.figure(num=2,figsize=(8,6),dpi=100,facecolor='w',edgecolor='k')
ax2 = plt.gca()
ax2.set_title('Energy')
ax2.set_xlabel('time (days)')
ax2.set_ylabel('Ek')

# Mechanic energy that must be conserved
Ek0 = 0.5*V0*V0 + G*M1/R0

# Loop all the schemes
for scheme in odeset.schemes:
	# Set integration scheme
	odeset.set_scheme(scheme)
	try:
		# Launch the integrator
		t,y,err = rk.odeRK(TwoBody,tspan,y0,odeset)
		print "scheme %s, error = %.2e with %d steps" % (scheme,err,len(t))
		# Plot results
		ax1.plot(y[:,0],y[:,1],y[:,2],label=scheme)
#		ax1.plot(y[:,6],y[:,7],y[:,8],label=scheme)
		Ek = 0.5*np.linalg.norm(y[:,3:5],axis=1)**2 + G*M1/np.linalg.norm(y[:,0:2],axis=1)
		ax2.plot(t/3600/34,Ek-Ek0,label=scheme)
	except Exception as e:
		print "scheme %s, failed!" % (scheme)

# Show the plot
ax2.legend(loc='lower right',fontsize='x-small',ncol=3)
plt.show()