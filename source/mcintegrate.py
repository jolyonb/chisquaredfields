#! /usr/bin/python
"""
Metropolis-Hastings Monte Carlo integration library
"""

import numpy as np
from math import sqrt
from random import normalvariate, uniform

def MCIntegrate(variables, integrand, probability, numsamples, parameters={}, burnin=1000) :
    """Perform a Metropolis-Hastings Monte Carlo integral for a 5D vector integrand"""

    # Initialize data collection
    samples = -burnin
    numsamples = int(numsamples)
    results = np.zeros([numsamples, 5])
    accepted = 0

    # Initialize values for each variable
    values = [0.0] * len(variables)
    lastprob = init_values(variables, values, probability, parameters)

    # Start integrating
    while samples < numsamples :
        # Jump to a new point
        values, lastprob, jumped = update_values(variables, values, probability, parameters, lastprob)

        # Are we done burning in?
        if samples >= 0 :
            # Count the number of iterations to compute the acceptance ratio
            if jumped : accepted += 1
            # Compute the integrand
            intval = integrand(values, parameters)
            # Add it to the results
            results[samples] = intval

        # Increment samples
        samples += 1

    # Compute integrals for the four different results (different types of stationary points)
    vals = [0] * 5
    vals[0], vals[1], vals[2], vals[3], vals[4] = np.transpose(results)
    integrals = np.zeros(5)
    errors = np.zeros(5)
    variances = np.zeros(5)
    for i in range(5) :
        integrals[i] = np.sum(vals[i]) / numsamples
        delta = vals[i] - integrals[i]
        variances[i] = np.sum(delta * delta) / (numsamples - 1)
        errors[i] = sqrt(variances[i] / numsamples)

    # Return integrals, errors and acceptance rate
    return integrals, errors, float(accepted) / numsamples

def update_values(variables, values, probability, parameters, lastprob) :
    """
    Jump to a new point in the domain of integration using the Metropolis-Hastings approach
    """

    newvalues = [0.0] * len(variables)

    # Generate a distance to jump for each variable
    for i, var in enumerate(variables) :
        vartype = var[1]
        varjump = var[2]
        if vartype == "n" :
            newvalues[i] = normalvariate(values[i], varjump)
        elif vartype == "pn" :
            newvalues[i] = abs(normalvariate(values[i], varjump))
    
    # Calculate the new probability
    newprob = probability(newvalues, parameters)

    # Accept or reject?
    # Return the new values, new probability and whether or not a jump was made
    if newprob >= lastprob or uniform(0, 1) < newprob / lastprob :
        # Accept
        return newvalues, newprob, True
    else :
        # Reject
        return values, lastprob, False

def init_values(variables, values, probability, parameters) :
    """Initialize variables for integration"""

    # Generate values for each variable
    for i, var in enumerate(variables) :
        vartype = var[1]
        if vartype == "n" :
            values[i] = normalvariate(0, 1)
        elif vartype == "pn" :
            values[i] = abs(normalvariate(0, 1))
    
    # Return the probability where we've landed
    return probability(values, parameters)

if __name__ == "__main__":
    print "This is a supporting library. It is not intended to be executed."
