"""

Created by Shane Bussmann

A number of visualization tools are included here to aid the user in evaluating
the:

    convergence of lnprob
    the posterior PDFs
    the evolution of the PDFs for each parameter of the model
    the covariance matrix for the posterior PDFs
    the best-fit model
    a number of randomly drawn realizations from the posterior PDFs

"""

from __future__ import print_function

import os
from astropy.io import fits
import visualutil
#import sys
#cwd = os.getcwd()
#sys.path.append(cwd)
#import config
import yaml


configloc = 'config.yaml'
configfile = open(configloc)
config = yaml.load(configfile)

def convergence(bestfitloc='posteriorpdf.fits'):

    """

    Plot the convergence profile.  I.e., Max(lnprob) - lnprob as a function of
    iteration number.

    """

    import numpy
    import matplotlib.pyplot as plt
    from pylab import savefig


    print("Reading burnin results from {0:s}".format(bestfitloc))
    pdf = fits.getdata(bestfitloc)
    keyname = 'lnprob'
    lnprob = pdf[keyname]

    lnprob = numpy.array(lnprob)
    lnprob = lnprob.max() - lnprob
    lnprob = numpy.abs(lnprob)

    plt.clf()
    plt.plot(lnprob, ',', alpha=0.5)
    plt.xlabel('iteration')
    plt.ylabel('max(lnprob) - lnprob')
    tmpcwd = os.getcwd()
    startindx = tmpcwd.find('ModelFits') + 10
    endindx = tmpcwd.find('uvfit') + 7
    objname = tmpcwd[startindx:endindx]
    plt.title(objname)
    plt.semilogy()

    outfile = 'convergence'
    savefig(outfile)

    try:
        import plotutils.autocorr as ac
        import plotutils.plotutils as pu

        from astropy.table import Table
        fitKeys = Table.read(bestfitloc).keys()

        nwalkers = config['Nwalkers']
        nsteps = len(pdf)/nwalkers
        ndim = len(fitKeys)
        assert isinstance(nsteps, int), 'the total number of sameples should be nsteps x nwalkers'

        chains = numpy.empty([nwalkers, nsteps, ndim])
        for ii, param in enumerate(fitKeys):
            these_chains = pdf[param]
            # first reshape chains in posteriorpdf.fits to
            # extract info for each walker, because
            # chains are flattened before saving.
            # currently each walker of the same iteration
            # are followed row by row.
            # hopefully this will work, assuming the order of the walkers
            # doesn't change in each iteration
            for i in range(nwalkers):
                chains[i, :, ii] = these_chains[ii::nwalkers]

        # If you leave off mean=False, then the function first averages the locations of all the walkers together, and plots the motion of this centroid over the course of the run
        pu.plot_emcee_chains(chains, mean=False)
        savefig('trace')

        # should fall off to zero after some time
        plt.clf()
        ac.plot_emcee_chain_autocorrelation_functions(chains)
        savefig('ACF')

        # calc ACF: about the # steps needed for these AC to die off
        print("ACF: {}".format(ac.emcee_chain_autocorrelation_lengths(chains)))

        # remove correlated samples
        thin_chain = ac.emcee_thinned_chain(chains)
        try:
            print(thin_chain.shape)
        except AttributeError:
            print("Oh no... cannot find uncorrelated sample..")
    except ImportError:
        pass


def walker(bestfitloc='posteriorpdf.fits', Ngood=5000):
    """
    Plot traces for chains. Modifed from Adrian Price-Whelan's code.

    - visual analysis using trace plots
    - must be produced for all parameters, not just those of interest
    - if reached stationary: mean and variance of the trace should be relatively constant

    """

    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    matplotlib.rcParams['font.family'] = "sans-serif"
    font_color = "#dddddd"
    tick_color = "#cdcdcd"

    fitresults = fits.getdata(bestfitloc)

    nwalkers = config['Nwalkers']
    nsteps = len(fitresults)/nwalkers
    assert isinstance(nsteps, int), 'the total number of sameples should be nsteps x nwalkers'

    from astropy.table import Table
    fitKeys = Table.read(bestfitloc).keys()

    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_pdf import PdfPages
    # Create the PdfPages object to which we will save the pages:
    # The with statement makes sure that the PdfPages object is closed properly at
    # the end of the block, even if an Exception occurs.

    with PdfPages('walkers.pdf') as pdf:

        numPanel = 5   # save plots for 5 parameters on each page
        # For each parameter, plot each walker on left panel, and a histogram
        # of all links from all walkers past prune_idx steps
        for ii, param in enumerate(fitKeys):
            # print(" plotting for {:} in panel {:}".format(param, ii % numPanel))
            these_chains = fitresults[param]

            if ii % numPanel == 0:
                fig = plt.figure(figsize=(16, 20.6))
                # two columns, left for trace plot; right for histogram
                gs = gridspec.GridSpec(numPanel, 3)
                counter_gs = 0

            # 5000 is the value we ways use for Ngood sample
            # plot the last total 5000 steps of all walkers
            prune_idx = (len(fitresults) - Ngood)/nwalkers
            # first reshape chains in posteriorpdf.fits to
            # extract info for each walker, because
            # chains are flattened before saving.
            # currently each walker of the same iteration
            # are followed row by row.
            # hopefully this will work, assuming the order of the walkers
            # doesn't change in each iteration
            chains = np.empty([nwalkers, nsteps])
            # for ll in np.arange(nsteps):
            #      for jj, samp in enumerate(these_chains):
            #         chains[jj%nwalkers, ll] = samp
            for i in range(nwalkers):
                chains[i, :] = these_chains[ii::nwalkers]
            # color walkers by their variance past prune_idx
            # so here, compute the maximum variance to scale the others to 0-1
            max_var = max(np.var(chains[:, prune_idx:], axis=1))

            totalwidth = these_chains.max() - these_chains.min()
            rms = np.std(these_chains[-Ngood:])
            nbins = totalwidth/rms

            ax1 = plt.subplot(gs[counter_gs, :2])
            ax1.set_axis_bgcolor("#333333")
            ax1.axvline(0,
                        color="#67A9CF",
                        alpha=0.7,
                        linewidth=2)
            for walker in chains:
                ax1.plot(np.arange(len(walker))-prune_idx, walker,
                         drawstyle="steps",
                         color=cm.bone_r(np.var(walker[prune_idx:]) / max_var),
                         alpha=0.5)
            ax1.set_ylabel(param,
                           fontsize=22,
                           labelpad=18,
                           rotation="horizontal",
                           color=font_color)
            # Don't show ticks on the y-axis
            ax1.yaxis.set_ticks([])
            # For the last plot on the bottom, add x-axis label.
            # Hide all others
            if counter_gs == numPanel - 1 or ii == len(fitKeys) - 1:
                ax1.set_xlabel("step number", fontsize=24,
                               labelpad=18, color=font_color)
            else:
                ax1.xaxis.set_visible(False)

            # histograms
            ax2 = plt.subplot(gs[counter_gs, 2])
            ax2.set_axis_bgcolor("#555555")
            # Create a histogram of all values past prune_idx. Make 100 bins
            #   between the y-axis bounds defined by the 'walkers' plot.
            ax2.hist(np.ravel(chains[:, prune_idx:]),
                     bins=int(np.min([nbins, 35])),
                     orientation='horizontal',
                     facecolor="#67A9CF",
                     edgecolor="none")

            # Same y-bounds as the walkers plot, so they line up
            ax1.set_ylim(np.min(chains[:, :]), np.max(chains[:, :]))
            ax2.set_ylim(ax1.get_ylim())
            ax2.xaxis.set_visible(False)
            ax2.yaxis.tick_right()
            # For the first plot, add titles and shift them up a bit
            if ii == 0:
                t = ax1.set_title("Walkers", fontsize=30, color=font_color)
                t.set_y(1.01)
                t = ax2.set_title("Posterior", fontsize=30, color=font_color)
                t.set_y(1.01)
            if "EinsteinRadius" in param or "Delta" in param:
                ax2.set_ylabel("arcsec",
                               fontsize=20,
                               rotation="horizontal",
                               color=font_color,
                               labelpad=16)
            ax2.yaxis.set_label_position("right")
            # Adjust axis ticks, e.g. make them appear
            # outside of the plots and change the padding / color.
            ax1.tick_params(axis='x', pad=2, direction='out',
                            colors=tick_color, labelsize=14)
            ax2.tick_params(axis='y', pad=2, direction='out',
                            colors=tick_color, labelsize=14)
            # Removes the top tick marks
            ax1.get_xaxis().tick_bottom()
            # this removed the first and last tick labels
            # so I can squash the plots right up against each other
            if param == "phi":
                ax2.set_yticks(ax2.get_yticks()[1:-2])
            else:
                ax2.set_yticks(ax2.get_yticks()[1:-1])
            fig.subplots_adjust(hspace=0.0, wspace=0.0, bottom=0.075,
                                top=0.95, left=0.12, right=0.88)
            if counter_gs == numPanel - 1 or ii == len(fitKeys) - 1:
                pdf.savefig(fig, facecolor='#222222')
                plt.close()
            counter_gs += 1
    return None


def posteriorPDF(bestfitloc='posteriorpdf.fits'):

    """

    Plot the posterior PDF of each parameter of the model.

    """

    # read posterior PDF
    print("Reading output from emcee")
    fitresults = fits.getdata(bestfitloc)
    tag = 'posterior'
    visualutil.plotPDF(fitresults, tag, Ngood=5000, axes='auto')


def evolvePDF(bestfitloc='posteriorpdf.fits', stepsize=50000):

    """

    Plot the evolution of the PDF of each parameter of the model.

    """

    import setuputil


    # Get upper and lower limits on the parameters to set the plot limits
    paramData = setuputil.loadParams(config)
    p_u = paramData['p_u']
    p_l = paramData['p_l']
    limits = [p_l, p_u]

    # read posterior PDF
    fitresults = fits.getdata(bestfitloc)
    nresults = len(fitresults)
    print("Output from emcee has = " + str(nresults) + " iterations.")
    start = 0
    for iresult in range(0, nresults, stepsize):

        strstep = str(stepsize)
        nchar = len(str(nresults))
        striresult = str(iresult).zfill(nchar)
        tag = 'evolution' + strstep + '.' + striresult + '.'
        trimresults = fitresults[start:start + stepsize]
        start += stepsize
        visualutil.plotPDF(trimresults, tag, limits=limits, Ngood=1000,
                axes='initial')

def covariance(bestfitloc='posteriorpdf.fits'):

    """

    Plot the covariance matrix for the parameters of the model.

    """

    import matplotlib.pyplot as plt
    import numpy
    from pylab import savefig
    import modifypdf
    from astropy.table import Table
    from matplotlib import rc


    # plotting parameters
    rc('font',**{'family':'sans-serif', 'sans-serif':['Arial Narrow'],
        'size':'6'})

    posteriorpdf = Table.read(bestfitloc)
    posteriorpdf = posteriorpdf[-5000:]

    # remove columns where the values are not changing
    posteriorpdfclean = modifypdf.cleanColumns(posteriorpdf)

    posteriorpdfgood = modifypdf.prune(posteriorpdfclean)

    headers = posteriorpdf.colnames
    ncol = len(headers)
    k = 0
    xsize = ncol * 2
    ysize = ncol * 1.5
    fig = plt.figure(figsize=(xsize, ysize))
    plt.subplots_adjust(left=0.020, bottom=0.02, right=0.99, top=0.97,
        wspace=0.5, hspace=0.5)

    #for i in numpy.arange(ncol):
    #    ax = plt.subplot(ncol, ncol, i + 1)
    #    namex = 'mu_aper'
    #    namey = headers[i]
    #    datax = mupdfgood[namex]
    #    datay = posteriorpdfgood[namey]
    #    if namex == 'lnprob':
    #        datax = datax.max() - datax
    #    if namey == 'lnprob':
    #        datay = datay.max() - datay
    #    lnprob = posteriorpdfgood['lnprob'].max() - posteriorpdfgood['lnprob']
    #    plt.hexbin(datax, datay, C = lnprob)
    #    plt.xlabel(namex)
    #    plt.ylabel(namey)


    for i in numpy.arange(ncol):

        for j in numpy.arange(ncol - i - 1) + i + 1:

            plotspot = ncol * i + j
            ax = plt.subplot(ncol, ncol, plotspot)

            namex = headers[i]
            namey = headers[j]
            #datax = posteriorpdforig[namex]
            #datay = posteriorpdforig[namey]
            #lnprob = posteriorpdforig['lnprob']

            #plt.hexbin(datax, datay, C = lnprob, color='black')

            datax = posteriorpdfgood[namex]
            datay = posteriorpdfgood[namey]
            if namex == 'lnprob':
                datax = datax.max() - datax
            if namey == 'lnprob':
                datay = datay.max() - datay
            lnprob = posteriorpdfgood['lnprob'].max() - posteriorpdfgood['lnprob']

            plt.hexbin(datax, datay, C = lnprob)
            plt.xlabel(namex)
            plt.ylabel(namey)
            print(i, j, plotspot, namex, namey)
            k += 1

    #plt.suptitle(iau_address, x=0.5, y=0.987, fontsize='xx-large')
    savefig('covariance.pdf')
    plt.clf()


def printFitParam(fitresult, fitKeys, mag=False):
    """
    Print parameters for this model
    mag: Boolean
        if True, print magnification factors as well
    """

    if not mag:
        fitresult = fitresult[:-4]
        fitKeys = fitKeys[:-4]

    print("Found the following parameters for this fit:")
    for k, v in zip(fitKeys, fitresult):
        print("%s : %.4f" %(k,v))


def bestFit(bestfitloc='posteriorpdf.fits', showOptical=False, cleanup=True,
        interactive=True, threshold=1.2):

    """

    Read posterior PDF and identify best-fit parameters.  Plot the best-fit
    model and compare to the data.  Also plot the residuals obtained after
    subtracting the best-fit model from the data and compare to the data.
    Optionally plot the best available optical image and compare to the data.

    Parameters
    ----------
    threshold: float
        in mJy, cleaning threshold

    """


    # read the posterior PDFs
    print("Found posterior PDF file: {:s}".format(bestfitloc))
    fitresults = fits.getdata(bestfitloc)

    from astropy.table import Table
    fitKeys = Table.read(bestfitloc).keys()

    # identify best-fit model
    minchi2 = fitresults['lnprob'].max()
    index = fitresults['lnprob'] == minchi2
    bestfit = fitresults[index][0]
    tag = 'bestfit'

    printFitParam(bestfit, fitKeys)
    visualutil.plotFit(config, bestfit, threshold, tag=tag, cleanup=cleanup,
            showOptical=showOptical, interactive=interactive)


def goodFits(bestfitloc='posteriorpdf.fits', Nfits=12, Ngood=5000,
        cleanup=True, interactive=True, showOptical=False, threshold=1.2):

    """

    Read posterior PDF and draw Nfits realizations from the final Ngood models
    at random.  Plot the model from each realization and compare to the data.
    Also plot the residuals obtained after subtracting the model from the data
    and compare to the data.  By default: Nfits = 12, Ngood=5000.

    Parameters
    ----------
    threshold: float
        in mJy, cleaning threshold


    """

    import modifypdf
    import numpy


    # read the posterior PDFs
    print("Found posterior PDF file: {:s}".format(bestfitloc))
    fitresults = fits.getdata(bestfitloc)
    fitresults = fitresults[-Ngood:]
    fitresults = modifypdf.prune(fitresults)

    # get keys
    from astropy.table import Table
    fitKeys = Table.read(bestfitloc).keys()

    # select the random realizations model
    Nunprune = len(fitresults)
    realids = numpy.floor(numpy.random.uniform(0, Nunprune, Nfits))

    for ifit in range(Nfits):
        realid = numpy.int(realids[ifit])
        fitresult = fitresults[realid]
        tag = 'goodfit' + str(realid).zfill(4)
        printFitParam(fitresult, fitKeys)
        visualutil.plotFit(config, fitresult, threshold, tag=tag, showOptical=showOptical,
                cleanup=cleanup, interactive=interactive)
