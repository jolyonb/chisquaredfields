#! /usr/bin/python
"""Number Density of Stationary Points for Chi-Squared Fields
v1.0 by Jolyon Bloomfield and Zander Moss, December 2016
See arXiv:1612.03890 for details
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

    # Global marker size msize
    msize = 4

    # Plot the data with errorbars and the exact solution
    fig, ax = plt.subplots()
    ax.errorbar(nuvals, integrals[4], yerr=errors[4], fmt='ro', markersize=msize, label="Numeric")
    ax.plot(nuvalsexact, exactvals, 'b-', label="Analytic")

    # Labels
    ax.set_title(r'Signed Number Density', fontsize=20)
    ax.set_xlabel(r'$\bar{\nu}$',fontsize=20)
    ax.set_ylabel(r'$\left\langle \frac{d{\cal N}^\mathrm{signed}}{d\bar{\nu}} \right\rangle$',fontsize=24,labelpad=-8)
    fig.subplots_adjust(left=0.16)

    # Legend
    h1, l1 = ax.get_legend_handles_labels()
    legend = ax.legend(h1,l1,loc='upper right',shadow=False,fancybox=True)

    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    frame = legend.get_frame()
    frame.set_facecolor('1.0')

    # Set sizes for the legend
    for label in legend.get_texts():
        label.set_fontsize(14)
    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width

    # Tick marks
    ax.tick_params(axis='x', which='major', labelsize=16)
    ax.tick_params(axis='y', which='major', labelsize=16)

    # Text box of parameters
    textlist=[
    r'$\sigma_0=%.2f$'%(sigma0)+'\n',
    r'$\sigma_1=%.2f$'%(sigma1)+'\n',
    r'$\gamma \,\, =  %.2f$'%(gamma)+'\n',
    r'$N\, = %1d$'%(N)]
    textstr=''.join(textlist)
    props = dict(boxstyle='round', facecolor='white', edgecolor='black')
    ax.text(0.8, 0.8, textstr, transform=ax.transAxes, fontsize=16,
        verticalalignment='top', bbox=props)

def plot_all(integrals, errors, nuvals, curves, nuvalsexact, extrema_vals, saddle_vals, lowgamma) :
    """Plot the individual number densities"""

    # A pretty purple!
    prettypurple = "#DE00FF"
    # Global marker size msize
    msize = 4

    # Plot the data with errorbars and the exact solution
    fig, ax = plt.subplots()
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='sans-serif')
    if curves[0] : ax.errorbar(nuvals, integrals[0], yerr=errors[0], markersize=msize, fmt='ro-', label="Minima")
    if curves[1] : ax.errorbar(nuvals, integrals[1], yerr=errors[1], markersize=msize, fmt='bo-', label="Saddle (+,+,--)")
    if curves[2] : ax.errorbar(nuvals, integrals[2], yerr=errors[2], markersize=msize, fmt='go-', label="Saddle (+,--,--)")
    if curves[3] : ax.errorbar(nuvals, integrals[3], yerr=errors[3], markersize=msize, color=prettypurple, fmt='o-', label="Maxima")

    # Plotting low gamma approximations
    if lowgamma and (curves[3] or curves[0]) : ax.plot(nuvalsexact, extrema_vals, 'r-')
    if lowgamma and (curves[2] or curves[1]) : ax.plot(nuvalsexact, saddle_vals, 'b-')

    # Labels
    ax.set_title(r'Number Density of Stationary Points', fontsize=20)
    ax.set_xlabel(r'$\bar{\nu}$',fontsize=20)
    ax.set_ylabel(r'$\left\langle \frac{d{\cal N}^\mathrm{stationary}}{d\bar{\nu}} \right\rangle$',fontsize=24,labelpad=-8)
    fig.subplots_adjust(left=0.16)

    # Legend
    h1, l1 = ax.get_legend_handles_labels()
    legend = ax.legend(h1,l1,loc='upper right',shadow=False,fancybox=True)

    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    frame = legend.get_frame()
    frame.set_facecolor('1.0')

    # Set sizes for the legend
    for label in legend.get_texts():
        label.set_fontsize(14)
    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width

    # Tick marks
    ax.tick_params(axis='x', which='major', labelsize=16)
    ax.tick_params(axis='y', which='major', labelsize=16)

    # Text box of parameters
    textlist=[
    r'$\sigma_0=%.2f$'%(sigma0)+'\n',
    r'$\sigma_1=%.2f$'%(sigma1)+'\n',
    r'$\gamma \,\, =  %.2f$'%(gamma)+'\n',
    r'$N\, = %1d$'%(N)]
    textstr=''.join(textlist)
    props = dict(boxstyle='round', facecolor='white', edgecolor='black')
    ax.text(0.8, 0.65, textstr, transform=ax.transAxes, fontsize=16,
        verticalalignment='top', bbox=props)

# Set some plotting parameters
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')

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
