"""

Authors: Shane Bussmann & Daisy Leung

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
import cPickle as pickle

configloc = 'config.yaml'
configfile = open(configloc)
config = yaml.load(configfile)


def check_and_thin_chain(chainFile='chain.pkl'):
    '''

    Parameters
    ----------
    chainFile: str
        pickle file containing unflattened chain in shape (nwalkers, nsteps, nparams)

    Returns
    -------
    thin_chain: array
        chain after removing correlated samples

    '''

    try:
        import plotutils.autocorr as ac
        import plotutils.plotutils as pu
    except ImportError:
        print("Please install plotutils and re-run this function.")
    import matplotlib.pyplot as plt

    import os.path
    if not os.path.exists(chainFile):
        raise IOError(chainFile+" does not exist.")
    with open(chainFile) as f:
        chain = pickle.load(f)

    walkers, steps, dim = chain.shape

    # If you leave off mean=False, then the function first averages the locations of all the walkers together, and plots the motion of this centroid over the course of the run
    plt.clf()
    plt.figure(figsize=(20.6, 16))
    pu.plot_emcee_chains(chain, mean=False)
    plt.savefig('trace_unflattened_chain')

    # should fall off to zero after some time
    plt.clf()
    plt.figure(figsize=(20.6, 16))
    ac.plot_emcee_chain_autocorrelation_functions(chain)
    plt.savefig('ACF_unflattened_chain')

    # calc ACF: about the # steps needed for these AC to die off
    print("Auto-correlation of each parameters: {}".format(ac.emcee_chain_autocorrelation_lengths(chain)))
    print("Iterations carried out: {:d}".format(steps))
    print("Should have many more iterations than ACF for all parameters.\n")

    # remove correlated samples
    thin_chain = ac.emcee_thinned_chain(chain)
    try:
        print("Number of iterations after thinning: {}".format(thin_chain.shape[1]))
        return thin_chain
    except AttributeError:
        print("Oh no... cannot find uncorrelated sample..")


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


def walker_reconstructed(bestfitloc='posteriorpdf.fits', chainFile='chain_reconstructed.pkl', converged_idx=0):

    """
    Plot traces for reconstructed chains. Modifed from Adrian Price-Whelan's code.
    For each parameter, plot at most 10 walkers on left, and a histogram from *all* walkers past converged_idx steps

    Test convergence:
    - visual analysis using trace plots
    - must be produced for all parameters, not just those of interest
    - if reached stationary: mean and variance of the trace should be relatively constant

    Do not run in CASA


    Parameters
    ----------
    bestfitloc: str
        file name for flattened chain; use to extract parameter names
    chainFile: str
        chain file reconstructed from posteriorpdf.fits obtained from visualutil.reconstruct_chain()

    converged_idx: ind
        index of iteration steps --> threshold for plotting posterior.

    """

    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    import cPickle as pickle

    matplotlib.rcParams['font.family'] = "sans-serif"
    font_color = "#dddddd"
    tick_color = "#cdcdcd"

    # get parameter names
    from astropy.table import Table
    fitKeys = Table.read(bestfitloc).keys()

    with open(chainFile) as f:
        chain = pickle.load(f)

    if converged_idx is None:
        converged_idx = visualutil.get_autocor(chainFile) * 5

    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_pdf import PdfPages
    # Create the PdfPages object to which we will save the pages:
    # The with statement makes sure that the PdfPages object is closed properly at
    # the end of the block, even if an Exception occurs.

    with PdfPages('walkers_reconstructed.pdf') as pdf:

        numPanel = 5   # save plots for 5 parameters on each page

        # For each parameter, plot each walker on left panel, and a histogram
        # of all links from all walkers past converged_idx steps
        for ii, param in enumerate(fitKeys):
            # print(" plotting for {:} in panel {:}".format(param, ii % numPanel))
            these_chains = chain[:, :, ii]

            if ii % numPanel == 0:
                fig = plt.figure(figsize=(16, 20.6))
                # two columns, left for trace plot; right for histogram
                gs = gridspec.GridSpec(numPanel, 3)
                counter_gs = 0

            # color walkers by their variance past converged_idx
            # so here, compute the maximum variance to scale the others to 0-1
            max_var = max(np.var(these_chains[:, converged_idx:], axis=1))

            totalwidth = these_chains.max() - these_chains.min()
            rms = np.std(these_chains[:, converged_idx:])
            nbins = totalwidth/rms * 5
            nbins = nbins if nbins > 0 else 10

            ax1 = plt.subplot(gs[counter_gs, :2])
            ax1.set_axis_bgcolor("#333333")
            ax1.axvline(0,
                        color="#67A9CF",
                        alpha=0.7,
                        linewidth=2)

            # plot trace for nw walkers
            if these_chains.shape[0] > 5:
                nw = 10
            else:
                nw = these_chains.shape[0]

            for walker in these_chains[np.random.choice(these_chains.shape[0], nw, replace=False), :]:
                ax1.plot(np.arange(len(walker))-converged_idx, walker,
                         drawstyle="steps",
                         color=cm.bone_r(np.var(walker[converged_idx:]) / max_var),
                         alpha=0.5)
            ax1.set_ylabel(param,
                           fontsize=16,
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
            # Create a histogram of all values past converged_idx. Make 100 bins
            #   between the y-axis bounds defined by the 'walkers' plot.
            ax2.hist(np.ravel(these_chains[:, converged_idx:]),
                     bins=int(np.min([nbins, 35])),
                     orientation='horizontal',
                     facecolor="#67A9CF",
                     edgecolor="none")

            # Same y-bounds as the walkers plot, so they line up
            ax1.set_ylim(np.min(these_chains[:, :]), np.max(these_chains[:, :]))
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
                               labelpad=20)
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
            ax2.set_yticks(ax2.get_yticks()[1:-1])

            fig.subplots_adjust(hspace=0.0, wspace=0.0, bottom=0.075,
                                top=0.95, left=0.12, right=0.88)
            if counter_gs == numPanel - 1 or ii == len(fitKeys) - 1:
                pdf.savefig(fig, facecolor='#222222')
                plt.close()
            counter_gs += 1
    return None


def quality(bestfitloc='posteriorpdf.fits', Ngood=5000, plot=True):
    '''
    Ad-hoc way to compare models of different setup, should really be likelihood ratio * Ockham factor; using Bayes Evidence
    here we just treat chi2 as -2 * lnprob; but model is non-linear.

    Also compute DIC, but need manual update of avg parameter values to yaml file

    '''

    import modifypdf
    from astropy.io import fits
    import numpy as np

    print("Reading output from posteriorpdf.fits")
    fitresults = fits.getdata(bestfitloc)

    # grab the last Ngood fits
    fitresults = fitresults[-Ngood:]
    # identify the good fits
    fitresultsgood = modifypdf.prune(fitresults)

    lnprob_med = np.median(fitresultsgood['lnprob'])

    import uvutil
    visfileloc = config['UVData']
    data_complex, data_wgt = uvutil.visload(visfileloc)
    npos = data_complex.size
    print("total number of vis: {}".format(npos))

    npos_rmflagged = data_complex[data_wgt > 0].size
    print("number of vis after removing data with negative or zero weights: {}".format(npos_rmflagged))

    nvis = np.min([npos, npos_rmflagged])

    from astropy.table import Table
    fitKeys = Table.read(bestfitloc).keys()
    nparams = len(fitKeys)

    DOF = nvis - nparams

    print("median lnprob/DOF: {}\n".format(lnprob_med/DOF))


    # find the average values across all parameters
    thetaAvg_dict = posteriorPDF(bestfitloc)

    for k, v in thetaAvg_dict.iteritems():
        if isinstance(v, float):
            print("{:s}: {:.2f}".format(k, v))
        else:
            print(k, v)

    import sandbox
    configFile_avg = 'averageParam.yaml'
    import os
    if not os.path.isfile(configFile_avg):
        os.system('cp config.yaml ' + configFile_avg)

    raw_input("Press enter after updating " + configFile_avg)

    # compute the loglike of that
    thetaAvg_Loglike = sandbox.plot(configloc=configFile_avg, plot=plot, tag=configFile_avg[:configFile_avg.find('.yaml')])

    DIC = -4. * np.mean(fitresultsgood['lnprob']) - 2. * thetaAvg_Loglike
    return DIC


def posteriorPDF(bestfitloc='posteriorpdf.fits'):

    """

    Plot the posterior PDF of each parameter of the model.

    Returns
    -------
    avg_dic: dict
        key = names of the model parameters, value = average value from the last Ngood samples
    """

    # read posterior PDF
    print("Reading output from emcee")
    fitresults = fits.getdata(bestfitloc)
    tag = 'posterior'
    avgParam_dict = visualutil.plotPDF(fitresults, tag, Ngood=5000, axes='auto')
    return avgParam_dict


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
