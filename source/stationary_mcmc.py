#! /usr/bin/python
"""
Compute the number density of each type of stationary point by Monte Carlo integration
"""

import numpy as np
import time
from mcintegrate import MCIntegrate
from math import pi, sin, cos, exp, sqrt
from common import scale, signed_exact

def integrand(values, parameters) :
    """Computes the integrand"""
    l1, l2, l3, a, b, c, d, e, f = values
    N, gamma, nu, sigma0, sigma1, gammanu, kappa, nufact = parameters

    # Construct the determinant
    term1 = a*a*b*b*c*c
    term2 = l1*l2*l3
    term3 = (c*c + e*e + f*f)*l1*l2 + (b*b + d*d)*l1*l3 + a*a*l2*l3
    term4 = -2.0*b*d*e*f*l1 + (c*c + e*e)*(d*d*l1 + a*a*l2) + b*b*(c*c*l1 + f*f*l1 + a*a*l3)
    # Scale each level of term by the number of factors of 3 nu / gamma it needs
    term2 *= nufact**3
    term3 *= nufact**2
    term4 *= nufact
    # Sum all of the terms
    det = term1 + term2 + term3 + term4

    # Compute the other two 
    minor1 = nufact * l1 + a*a
    minor2 = minor1 * (nufact * l2 + b*b + d*d) - a*a * d*d

    # Find the type of stationary point
    # Contributions to minima, ++-, +--, maxima, signed number density
    result = np.array([0.0, 0.0, 0.0, 0.0, det])

    if minor1 < 0 and minor2 > 0 and det < 0 :
        # Three negative eigenvalues (maxima)
        result[3] = abs(det)
    elif minor1 > 0 and minor2 > 0 and det > 0 :
        # Three positive eigenvalues (minima)
        result[0] = abs(det)
    elif det > 0 :
        # Two negative, one positive eigenvalues
        result[2] = abs(det)
    else :
        # One negative, two positive eigenvalues
        result[1] = abs(det)

    # Return the result
    return result

def probability(values, parameters) :
    """Returns the (relative) probability density at the given values"""
    l1, l2, l3, a, b, c, d, e, f = values
    N, gamma, nu, sigma0, sigma1, gammanu, kappa, nufact = parameters

    # Compute Q
    trZ = l1 + l2 + l3
    trZ2 = l1*l1 + l2*l2 + l3*l3
    Q = a*a + b*b + c*c + d*d + e*e + f*f + kappa * (gammanu + trZ)**2 + 2.5*(3*trZ2 - trZ*trZ)

    # Construct the probability
    prob = (a * b * c)**(N-4) * a*a * b
    prob *= abs((l1 - l2) * (l2 - l3) * (l3 - l1))
    prob *= exp(-Q/2)

    # Return the result
    return prob

def number_density(parameters, numsamples) :
    """Computes the number density of stationary points"""

    # Construct the full set of parameters we'll need
    N, gamma, nu, sigma0, sigma1 = parameters
    gammanu = gamma * nu
    kappa = 1.0 / (1 - gamma * gamma)
    nufact = 3.0 * nu / gamma
    fullparams = [N, gamma, nu, sigma0, sigma1, gammanu, kappa, nufact]

    # Define the variables of integration and their ranges
    # "n" is standard normal, good enough for initialization for everything
    # "pn" is positive normal
    jump = 0.5
    jump2 = 0.5
    jump3 = 0.5
    variables = [
        ["l1", "n", jump], ["l2", "n", jump], ["l3", "n", jump],
        ["a", "pn", jump2], ["b", "pn", jump2], ["c", "pn", jump2],
        ["d", "n", jump3], ["e", "n", jump3], ["f", "n", jump3]
    ]

    # Perform the Monte Carlo integration
    integrals, errors, acceptance = MCIntegrate(variables, integrand, probability, numsamples, parameters=fullparams)

    # Scale everything by the appropriate coefficient
    scaleval = scale(*parameters)
    integrals *= scaleval
    errors *= scaleval

    # Return the results
    return integrals, errors, signed_exact(*parameters), acceptance

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
    print "Acceptance rate:", acceptance * 100 , "%"

    print "Minima:      ", integrals[0], "+-", errors[0]
    print "Saddle (++-):", integrals[1], "+-", errors[1]
    print "Saddle (+--):", integrals[2], "+-", errors[2]
    print "Maxima:      ", integrals[3], "+-", errors[3]
    print "Signed:      ", integrals[4], "+-", errors[4]
    print "Signed exact:", exact
    print "Signed error:", abs(integrals[4] - exact)
