#! /usr/bin/python
"""
Supporting functions for computing the number density of stationary points
"""

from scipy.special import gamma as Gamma
from math import pi, exp, sqrt

def scale(N, gamma, nu, sigma0, sigma1) :
    """Compute alpha dP/dnu"""
    if nu > 0 :
        alpha = 1.0 / (6 * pi)**1.5 * (sigma1 / sigma0 / nu)**3
        dpdnu = nu ** (N - 1) * exp(-nu*nu / 2) / 2**(float(N)/2-1) / Gamma(float(N)/2)
    elif nu == 0.0 :
        if N > 4 :
            return 0.0
        elif N == 4 :        
            return 1.0 / (6 * pi)**1.5 * (sigma1 / sigma0)**3 * exp(-nu*nu / 2) / 2**(float(N)/2-1) / Gamma(float(N)/2)
        elif N < 4 :
            raise ValueError("Scale diverges for N < 4 at nu = 0")
    else :
        raise ValueError("Scale is not defined for nu < 0")
    return alpha * dpdnu

def signed_exact(N, gamma, nu, sigma0, sigma1) :
    """Compute the exact signed number density"""
    val = (N-1)*(N-2)*(N-3) - 3*(N-1)*(N-1)*nu*nu + 3*N*nu**4 - nu**6
    return val * scale(N, gamma, nu, sigma0, sigma1)

def lowgamma_extrema(N, gamma, nuval, sigma0, sigma1):
    nufact = 3.0*nuval/gamma
    return scale(N, gamma, nuval, sigma0, sigma1)*(nufact**3)*(29.0*sqrt(2*pi) - 12.0*sqrt(3.0*pi))/(4.0*pi*(3.0**3)*(5.0**(3.0/2.0)))

def lowgamma_saddles(N, gamma, nuval, sigma0, sigma1):
    nufact = 3.0*nuval/gamma
    return scale(N, gamma, nuval, sigma0, sigma1)*(nufact**3)*(29.0*sqrt(2*pi) + 12.0*sqrt(3.0*pi))/(4.0*pi*(3.0**3)*(5.0**(3.0/2.0)))

def V_N(N, gamma) :
    """Compute V_N, the normalization factor for the MCMC integral"""
    return 2.0**(1.5*(N-1)) / 5**2.5 / 27 * pi * sqrt(1 - gamma*gamma) * gamma3(float(N-1)/2)

def gamma3(x) :
    """Compute Gamma_3(x), the multivariate gamma function"""
    return pi**1.5 * Gamma(x) * Gamma(x - 0.5) * Gamma(x - 1.0)

if __name__ == "__main__":
    print "This is a supporting library. It is not intended to be executed."
