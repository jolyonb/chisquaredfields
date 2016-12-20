Number Density of Stationary Points for Chi-Squared Fields
==========================================================

v1.0 by Jolyon Bloomfield and Zander Moss, December 2016

This python code computes the number density of stationary points for chi-squared fields. See arXiv:1612.03890 (https://arxiv.org/abs/1612.03890) for details.

This code requires the following python libraries to be installed:
* vegas
* cython
* matplotlib

Source files:
* compute.py computes the number density of stationary points across a range of nu. See in-line help (./compute.py -h) for usage information.
* plot.py plots the results of a computation by compute.py, showing both the number density of stationary points and the signed number density. See in-line help (./plot.py -h) for usage information.
* stationary_mcmc.py is a library that runs an MCMC computation for the signed number density. It is called from compute.py. It can also be run stand-alone to perform a single integration.
* mcintegrate.py is a library used by stationary_mcmc.py to perform the MCMC integration.
* stationary_vegas.py is a library that runs a VEGAS computation for the signed number density. It is called from compute.py. It can also be run stand-alone to perform a single integration. (VEGAS tends to be much more efficient than the MCMC computation.)
* integrand.pyx is the Cython implementation of the integrand for VEGAS.
* integrand.pyxbld contains compilation information for integrand.pyx.
