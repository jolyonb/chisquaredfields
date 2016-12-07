#! /usr/bin/python

import numpy as np
import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})
import vegas
import integrand as INT
import time

import common
from math import pi

def number_density(parameters, numsamples):	
	N, gamma, nu, sigma0, sigma1 = parameters

	domain=[[-pi/2,pi/2],[-pi/2,pi/2],[-pi/2,pi/2],[0,pi/2],[0,pi/2],[0,pi/2],[-pi/2,pi/2],[-pi/2,pi/2],[-pi/2,pi/2]]

	f=INT.f_cython(dim=9,N=N,nu=nu,gamma=gamma)
		
	integ = vegas.Integrator(domain, nhcube_batch=1000)
	
	integ(f, nitn=10, neval=numsamples)
	vecresult = integ(f, nitn=10, neval=numsamples)

	scale = common.scale(*parameters)
	V_N = common.V_N(N,gamma)
	prefactor = scale/V_N

	integrals = np.zeros(5)
	errors = np.zeros(5)

	for i in range(1,vecresult.shape[0]):
		integrals[i-1] = prefactor*vecresult[i].mean
		errors[i-1] = prefactor*vecresult[i].sdev

	integrals[4] = prefactor*vecresult[0].mean		
	errors[4] = prefactor*vecresult[0].sdev	

	return integrals, errors, common.signed_exact(*parameters), 0.0


if __name__ == "__main__":
	# parameters = [N, gamma, nu, sigma0, sigma1]
	parameters = [4, 0.8, 1.0, 1.0, 1.0]
	numsamples = 1e5

	print "*" * 60
	print "Performing test run"
	print "Computing number densities with", int(numsamples), "samples..."
	print "N       = ", parameters[0]
	print "nu      = ", parameters[2]
	print "gamma   = ", parameters[1]
	print "sigma_0 = ", parameters[3]
	print "sigma_1 = ", parameters[4]

	start = time.time()
	integrals, errors, exact, acceptance = number_density(parameters, numsamples)
	end = time.time()

	print "Finished in", round(end - start, 4), "s"

	print "Minima:	  ", integrals[0], "+-", errors[0]
	print "Saddle (++-):", integrals[1], "+-", errors[1]
	print "Saddle (+--):", integrals[2], "+-", errors[2]
	print "Maxima:	  ", integrals[3], "+-", errors[3]
	print "Signed:      ", integrals[4], "+-", errors[4]
	print "Signed exact:", exact
	print "Signed error:", abs(integrals[4] - exact)

