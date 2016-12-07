#! /usr/bin/python
"""Number Density of Stationary Points for Chi-Squared Fields
v1.0 by Jolyon Bloomfield and Zander Moss, December 2016
See arXiv:1612.????? for details
Plotting Module"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
from common import signed_exact, lowgamma_extrema, lowgamma_saddles

#
# Deal with command line arguments
#
parser = argparse.ArgumentParser(description="Plot the number density of stationary points in chi^2 fields", epilog="By default, all plots and curves are shown.")

# Input file
parser.add_argument("-i", help="Input file (default 'data.dat')", 
    type=argparse.FileType('r'), default="data.dat", dest="file")

# Plot signed number density?
parser.add_argument("-ns", help="Do not plot signed number density", 
    dest="signed", action="store_false", default=True)

# Plot individual number densities?
parser.add_argument("-ni", help="Do not plot individual number densities", 
    dest="individual", action="store_false", default=True)

# Which lines to plot?
parser.add_argument("-nx", help="Individual plot: Do not plot maxima", 
    dest="max", action="store_false", default=True)
parser.add_argument("-nn", help="Individual plot: Do not plot minima", 
    dest="min", action="store_false", default=True)
parser.add_argument("-nppm", help="Individual plot: Do not plot (++-) saddles", 
    dest="saddleppm", action="store_false", default=True)
parser.add_argument("-npmm", help="Individual plot: Do not plot (+--) saddles", 
    dest="saddlepmm", action="store_false", default=True)

# Plot low gamma analytic approximations
parser.add_argument("-l", help="Plot low-gamma analytic approximations (default off)", 
    dest="lowgamma", action="store_true", default=False)

# Error scaling
def pos_float(value):
    try:
        ivalue = float(value)
    except ValueError :
        raise argparse.ArgumentTypeError("must be a number. You supplied: %s" % value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("must be > 0. You supplied: %s" % value)
    return ivalue

parser.add_argument("-e", help="Error scaling (default 1.0)", 
    dest="errscale", default=1.0, type=pos_float)

# Quiet mode?
parser.add_argument("-q", help="Quiet mode", action="store_true", dest="quiet", default=False)

# Parse the command line
args = parser.parse_args()
# Print the header
if not args.quiet : print __doc__

#
# Processing starts here
#

# Read in the data
data = args.file.readlines()
args.file.close()

# Process the data
results = [map(float, x.split(", ")) for x in data]
N, gamma, sigma0, sigma1, numsamples = results[0]
data = np.transpose(results[1:])
nuvals = data[0]
nucount = 1000
nuvalsexact = np.linspace(nuvals[0], nuvals[-1], nucount)
integrals = data[1:6]
errors = data[6:11]
errors *= args.errscale
# 11 is the exact signed number density at that nu
# 12 is the acceptance rate from the MCMC (vegas will return zero)

def plot_signed(integrals, errors, nuvals, nuvalsexact, exactvals) :
    """Plot the signed number density, along with the exact solution"""

    # Plot the data with errorbars and the exact solution
    fig, ax = plt.subplots()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax.errorbar(nuvals, integrals[4], yerr=errors[4], fmt='ro')
    ax.plot(nuvalsexact, exactvals, 'b-')

    ax.set_xlabel(r'$\bar{\nu}$',fontsize=16)
    ax.set_ylabel(r'$\displaystyle\left\langle \frac{d{\cal N}^{\mathrm{signed}}}{d\bar{\nu}} \right\rangle$',fontsize=16)
    ax.set_title(r'Signed Number Density', fontsize=20)
    fig.subplots_adjust(left=0.16)

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    samplenum = "{:.0e}".format(numsamples)
    samplenum = samplenum.replace("e+0", "\\times 10^{") + "}"
    strings = [r'$N=' + str(int(N)) + '$',
               r'$\gamma=%.2f$'%(gamma),
               r'$\sigma_0=' + str(sigma0) + '$',
               r'$\sigma_1=' + str(sigma1) + '$',
               r'$' + samplenum + '\ \mathrm{Samples}$'
    ]

    # place a text box in upper left in axes coords
    ax.text(0.75, 0.95, "\n".join(strings), transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

def plot_all(integrals, errors, nuvals, curves, nuvalsexact, extrema_vals, saddle_vals, lowgamma) :
    """Plot the individual number densities"""

    # Plot the data with errorbars and the exact solution
    fig, ax = plt.subplots()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    if curves[0] : ax.errorbar(nuvals, integrals[0], yerr=errors[0], fmt='ro-', label="Minima")
    if curves[1] : ax.errorbar(nuvals, integrals[1], yerr=errors[1], fmt='bo-', label="Saddle (+,+,--)")
    if curves[2] : ax.errorbar(nuvals, integrals[2], yerr=errors[2], fmt='go-', label="Saddle (+,--,--)")
    if curves[3] : ax.errorbar(nuvals, integrals[3], yerr=errors[3], fmt='ko-', label="Maxima")

	# Plotting low gamma approximations
    if lowgamma and (curves[3] or curves[0]) : ax.plot(nuvalsexact, extrema_vals, 'r-')
    if lowgamma and (curves[2] or curves[1]) : ax.plot(nuvalsexact, saddle_vals, 'b-')

    ax.legend()

    ax.set_xlabel(r'$\bar{\nu}$',fontsize=16)
    ax.set_ylabel(r'$\displaystyle\left\langle \frac{d{\cal N}}{d\bar{\nu}} \right\rangle$',fontsize=16)
    ax.set_title(r'Number Density of Stationary Points', fontsize=20)
    fig.subplots_adjust(left=0.16)

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    samplenum = "{:.0e}".format(numsamples)
    samplenum = samplenum.replace("e+0", "\\times 10^{") + "}"
    strings = [r'$N=' + str(int(N)) + '$',
               r'$\gamma=%.2f$'%(gamma),
               r'$\sigma_0=' + str(sigma0) + '$',
               r'$\sigma_1=' + str(sigma1) + '$',
               r'$' + samplenum + '\ \mathrm{Samples}$'
    ]

    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, "\n".join(strings), transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

if args.signed :
    # Compute the exact solution for the signed number density
    exactvals = np.zeros(nucount)
    for i, nuval in enumerate(nuvalsexact) :
        exactvals[i] = signed_exact(N, gamma, nuval, sigma0, sigma1)
    # Plot the signed number density
    plot_signed(integrals, errors, nuvals, nuvalsexact, exactvals)

if args.individual :
    if args.lowgamma :
    	#Calculating the low gamma approximation for saddles and extrema.
        extrema_vals = np.zeros(nucount)
        saddle_vals = np.zeros(nucount)
        for i, nuval in enumerate(nuvalsexact) :
            extrema_vals[i] = lowgamma_extrema(N, gamma, nuval, sigma0, sigma1)
            saddle_vals[i] = lowgamma_saddles(N, gamma, nuval, sigma0, sigma1)
    else :
        extrema_vals = False
        saddle_vals = False
        nuvalsexact = False

    # Plot the individual number densities
    curves = [args.min, args.saddleppm, args.saddlepmm, args.max]
    if any(curves) :
        plot_all(integrals, errors, nuvals, curves, nuvalsexact, extrema_vals, saddle_vals, args.lowgamma)
    else :
        print "Error: No curves requested in individual number densities!"

if args.signed or args.individual :
    plt.show()
else :
    print "Error: No plots requested!"
