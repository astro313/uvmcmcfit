#!/usr/bin/env python
"""
 Author: Shane Bussmann

 Last modified: 2014 February 26

 Note: This is experimental software that is in a very active stage of
 development.  If you are interested in using this for your research, please
 contact me first at sbussmann@astro.cornell.edu!  Thanks.

 Purpose: Fit a parametric model to interferometric data using Dan
 Foreman-Mackey's emcee routine.  Gravitationally lensed sources are accounted
 for using ray-tracing routines based on Adam Bolton's lensdemo_script and
 lensdemo_func python scripts.  Here is the copyright license from
 lensdemo_script.py:

 Copyright 2009 by Adam S. Bolton
 Creative Commons Attribution-Noncommercial-ShareAlike 3.0 license applies:
 http://creativecommons.org/licenses/by-nc-sa/3.0/
 All redistributions, modified or otherwise, must include this
 original copyright notice, licensing statement, and disclaimer.
 DISCLAIMER: ABSOLUTELY NO WARRANTY EXPRESS OR IMPLIED.
 AUTHOR ASSUMES NO LIABILITY IN CONNECTION WITH THIS COMPUTER CODE.

--------------------------
 USAGE

 python $PYSRC/uvmcmcfit.py

--------------------------
 SETUP PROCEDURES

 1. Establish a directory that contains data for the specific target for which
 you wish to measure a lens model.  This is the directory from which you will
 run the software.

 I call this "uvfit00" for the first run on a given dataset, "uvfit01" for
 the second, etc.

 2. Inside this directory, you must ensure the following files are present:

 - "config.py": This is the configuration file that describes where the source
 of interest is located, what type of model to use for the lens and source, the
 name of the image of the target from your interferometric data, the name of
 the uvfits files containing the interferometric visibilities, and a few
 important processing options as well.  Syntax is python.

 - Image of the target from your interferometric data.  The spatial resolution
 of this image (arcseconds per pixel), modified by an optional oversampling
 parameter, defines the spatial resolution in both the unlensed and lensed
 surface brightness maps.

 - interferometric visibilities for every combination of array configuration,
 sideband, and date observed that you want to model.  

 3. More info about the constraints and priors input files.

 - Lenses: The lenses are assumed to have singular isothermal ellipsoid
 profiles.  

 - Sources: Sources are represented by Gaussian profiles.  Source positions are
 always defined relative to the primary lens, unless there is no lens, in which
 case they are defined relative to the emission centroid defined in
 "config.py."

--------
 OUTPUTS

 "posteriorpdf.hdf5": model parameters for every MCMC iteration, in hdf5
 format.  Google search for hdf5 view if you want a tool to inspect the hdf5
 files directly.

"""

from __future__ import print_function

# import the required modules
import os
import os.path
import sys
from astropy.io import fits
from astropy.io.misc import hdf5
import numpy
from astropy.table import Table
#import emcee
from emcee import PTSampler
#from emcee import EnsembleSampler
#import pyximport
#pyximport.install(setup_args={"include_dirs":numpy.get_include()})
import sample_vis
import lensutil
import uvutil
import setuputil


cwd = os.getcwd()
sys.path.append(cwd)
import config

def logp(pzero_regions):

    """

    Function that computes the log prior probabilities of the model parameters.

    """

    # ensure all parameters are finite
    if (pzero_regions * 0 != 0).any():
        priorln = -numpy.inf

    # Uniform priors
    uniform_regions = paramSetup['prior_shape'] == 'Uniform'
    if uniform_regions.any():
        p_l_regions = paramSetup['p_l'][uniform_regions]
        p_u_regions = paramSetup['p_u'][uniform_regions]
        pzero_uniform = pzero_regions[uniform_regions]
        priorln = 0
        #mu = 1
        if (pzero_uniform < p_l_regions).any():
            priorln = -numpy.inf
        if (pzero_uniform > p_u_regions).any():
            priorln = -numpy.inf

    # Gaussian priors
    gaussian_regions = paramSetup['prior_shape'] == 'Gaussian'
    if gaussian_regions.any():
    #ngaussian = paramSetup['prior_shape'][gaussian_regions].size
    #for ipar in range(ngaussian):
        mean_regions = paramSetup['p_l'][gaussian_regions]
        rms_regions = paramSetup['p_u'][gaussian_regions]
        #part1 = numpy.log(2 * numpy.pi * rms_regions ** 2)
        parameter = pzero_regions[gaussian_regions]
        #print(parameter - mean_regions, (parameter - mean_regions)/rms_regions)
        part2 = (parameter - mean_regions) ** 2 / rms_regions ** 2
        priorln = -2.5 * (part2).sum()
        #priorln += priorln_param

    return priorln#, mu


def logl(pzero_regions):

    """ 
    
    Function that computes the log likelihood of the data.
    
    """

    # search poff_models for parameters fixed relative to other parameters
    fixed = (numpy.where(fixindx >= 0))[0]
    nfixed = fixindx[fixed].size
    p_u_regions = paramSetup['p_u']
    poff_regions = p_u_regions.copy()
    poff_regions[:] = 0.
    #for ifix in range(nfixed):
    #    poff_regions[fixed[ifix]] = pzero_regions[fixindx[fixed[ifix]]]
    for ifix in range(nfixed):
        ifixed = fixed[ifix]
        subindx = fixindx[ifixed]
        par0 = 0
        if fixindx[subindx] > 0:
            par0 = pzero_regions[fixindx[subindx]]
        poff_regions[ifixed] = pzero_regions[subindx] + par0

    parameters_regions = pzero_regions + poff_regions

    model_real = 0.
    model_imag = 0.
    npar_previous = 0

    amp = []  # Will contain the 'blobs' we compute

    nregions = paramSetup['nregions']
    for regioni in range(nregions):

        # get the model info for this model
        x = paramSetup['x'][regioni]
        y = paramSetup['y'][regioni]
        headmod = paramSetup['modelheader'][regioni]
        nlens = paramSetup['nlens_regions'][regioni]
        nsource = paramSetup['nsource_regions'][regioni]
        model_types = paramSetup['model_types'][regioni]

        # get pzero, p_u, and p_l for this specific model
        nparlens = 5 * nlens
        nparsource = 6 * nsource
        npar = nparlens + nparsource + npar_previous
        parameters = parameters_regions[npar_previous:npar]
        npar_previous = npar

        #-----------------------------------------------------------------
        # Create a surface brightness map of lensed emission for the given set
        # of foreground lens(es) and background source parameters.
        #-----------------------------------------------------------------

        g_image, g_lensimage, e_image, e_lensimage, amp_tot, amp_mask = \
                lensutil.sbmap(x, y, nlens, nsource, parameters, model_types)
        amp.extend(amp_tot)
        amp.extend(amp_mask)

        #----------------------------------------------------------------------
        # Python version of UVMODEL:
        # "Observe" the lensed emission with the interferometer
        #----------------------------------------------------------------------

        if nlens > 0:
            # Evaluate amplification for each region
            lensmask = e_lensimage != 0
            mask = e_image != 0
            numer = g_lensimage[lensmask].sum()
            denom = g_image[mask].sum()
            amp_mask = numer / denom
            numer = g_lensimage.sum()
            denom = g_image.sum()
            amp_tot = numer / denom
            if amp_tot > 1e2:
                amp_tot = 1e2
            if amp_mask > 1e2:
                amp_mask = 1e2
            amp.extend([amp_tot])
            amp.extend([amp_mask])

        model_complex = sample_vis.uvmodel(g_lensimage, headmod, uuu, vvv, pcd)
        model_real += numpy.real(model_complex)
        model_imag += numpy.imag(model_complex)

        #fits.writeto('g_lensimage.fits', g_lensimage, headmod, clobber=True)
        #import matplotlib.pyplot as plt
        #print(pzero_regions)
        #plt.imshow(g_lensimage, origin='lower')
        #plt.colorbar()
        #plt.show()
        #plt.imshow(g_image, origin='lower')
        #plt.colorbar()
        #plt.show()
        #import pdb; pdb.set_trace()

    # use all visibilities
    goodvis = (real * 0 == 0)

    # calculate chi^2 assuming natural weighting
    #fnuisance = 0.0
    modvariance_real = 1 / wgt #+ fnuisance ** 2 * model_real ** 2
    modvariance_imag = 1 / wgt #+ fnuisance ** 2 * model_imag ** 2
    #wgt = wgt / 4.
    chi2_real_all = (real - model_real) ** 2. / modvariance_real
    chi2_imag_all = (imag - model_imag) ** 2. / modvariance_imag
    chi2_all = numpy.append(chi2_real_all, chi2_imag_all)

    # compute the sigma term
    sigmaterm_real = numpy.log(2 * numpy.pi * modvariance_real)
    sigmaterm_imag = numpy.log(2 * numpy.pi * modvariance_imag)
    sigmaterm_all = numpy.append(sigmaterm_real, sigmaterm_imag)

    # compute the ln likelihood
    lnlikemethod = paramSetup['lnlikemethod']
    if lnlikemethod == 'chi2':
        lnlike = chi2_all
    else:
        lnlike = chi2_all + sigmaterm_all

    # compute number of degrees of freedom
    #nmeasure = lnlike.size
    #nparam = (pzero != 0).size
    #ndof = nmeasure - nparam

    # assert that lnlike is equal to -1 * maximum likelihood estimate
    likeln = -0.5 * lnlike[goodvis].sum()
    if likeln * 0 != 0:
        likeln = -numpy.inf

    return likeln#, amp

#def lnprob(pzero_regions):
#
#    """
#
#    Computes ln probabilities via ln prior + ln likelihood
#
#    """
#
#    lp, mu = lnprior(pzero_regions)
#
#    if not numpy.isfinite(lp):
#        probln = -numpy.inf
#        mu = 1
#        return probln, mu
#
#    ll, mu = lnlike(pzero_regions)
#
#    normalization = 1.0#2 * real.size
#    probln = lp * normalization + ll
#    #print(probln, lp*normalization, ll)
#    
#    return probln, mu

# Determine parallel processing options
mpi = config.ParallelProcessingMode

# Single processor with Nthreads cores
if mpi != 'MPI':

    # set the number of threads to use for parallel processing
    Nthreads = config.Nthreads

# multiple processors on a cluster using MPI
else:
    from emcee.utils import MPIPool

    # One thread per slot
    Nthreads = 1

    # Initialize the pool object
    pool = MPIPool()

    # If this process is not running as master, wait for instructions, then exit
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

#--------------------------------------------------------------------------
# Read in ALMA image and beam
im = fits.getdata(config.ImageName)
im = im[0, 0, :, :].copy()
headim = fits.getheader(config.ImageName)

# get resolution in ALMA image
#celldata = numpy.abs(headim['CDELT1'] * 3600)

#--------------------------------------------------------------------------
# read in visibilities
fitsfiles = config.FitsFiles
nfiles = len(fitsfiles)
nvis = []

# read in the observed visibilities
uuu = []
vvv = []
real = []
imag = []
wgt = []
for file in fitsfiles:
    print("Doing file {:s}".format(file))
    vis_data = fits.open(file)

    uu, vv = uvutil.uvload(vis_data)
    pcd = uvutil.pcdload(vis_data)
    real_raw, imag_raw, wgt_raw = uvutil.visload(vis_data)
    uuu.extend(uu)
    vvv.extend(vv)
    real.extend(real_raw)
    imag.extend(imag_raw)
    wgt.extend(wgt_raw)

# convert the list to an array
real = numpy.array(real)
imag = numpy.array(imag)
wgt = numpy.array(wgt)
uuu = numpy.array(uuu)
vvv = numpy.array(vvv)
#www = numpy.array(www)

# remove the data points with zero or negative weight
positive_definite = wgt > 0
real = real[positive_definite]
imag = imag[positive_definite]
wgt = wgt[positive_definite]
uuu = uuu[positive_definite]
vvv = vvv[positive_definite]
#www = www[positive_definite]

npos = wgt.size

#----------------------------------------------------------------------------
# Load input parameters
paramSetup = setuputil.loadParams(config)
Nwalkers = paramSetup['Nwalkers']
Ntemps = paramSetup['Ntemps']
nregions = paramSetup['nregions']
nparams = paramSetup['nparams']
pname = paramSetup['pname']
nsource_regions = paramSetup['nsource_regions']

# Use an intermediate posterior PDF to initialize the walkers if it exists
posteriorloc = 'posteriorpdf.hdf5'
if os.path.exists(posteriorloc):

    # read the latest posterior PDFs
    print("Found existing posterior PDF file: {:s}".format(posteriorloc))
    posteriordat = hdf5.read_table_hdf5(posteriorloc)
    if len(posteriordat) > 1:

        # assign values to pzero
        nlnprob = 1
        pzero = numpy.zeros((Ntemps, Nwalkers, nparams))
        startindx = nlnprob
        import pdb; pdb.set_trace()
        for itemp in range(Ntemps):
            for j in range(nparams):
                namej = posteriordat.colnames[j + startindx]
                pzero[itemp, :, j] = posteriordat[namej][itemp, -Nwalkers:]

        # number of mu measurements
        #nmu = len(posteriordat.colnames) - nparams - nlnprob

        # output name is based on most recent burnin file name
        realpdf = True
    else:
        realpdf = False
else:
    realpdf = False

if not realpdf:
    extendedpname = ['lnprob']
    extendedpname.extend(pname)
    #nmu = 0
    #for regioni in range(nregions):
    #    ri = str(regioni)
    #    if paramSetup['nlens_regions'][regioni] > 0:
    #        nsource = nsource_regions[regioni]
    #        for i in range(nsource):
    #            si = '.Source' + str(i) + '.Region' + ri
    #            extendedpname.append('mu_tot' + si) 
    #            extendedpname.append('mu_aper' + si) 
    #            nmu += 2
    #        extendedpname.append('mu_tot.Region' + ri)
    #        extendedpname.append('mu_aper.Region' + ri) 
    #        nmu += 2
    posteriordat = Table(names = extendedpname)
    pzero = numpy.array(paramSetup['pzero'])

# make sure no parts of pzero exceed p_u or p_l
#arrayp_u = numpy.array(p_u)
#arrayp_l = numpy.array(p_l)
#for j in range(Nwalkers):
#    exceed = arraypzero[j] >= arrayp_u
#    arraypzero[j, exceed] = 2 * arrayp_u[exceed] - arraypzero[j, exceed]
#    exceed = arraypzero[j] <= arrayp_l
#    arraypzero[j, exceed] = 2 * arrayp_l[exceed] - arraypzero[j, exceed]
#pzero = arraypzero
#p_u = arrayp_u
#p_l = arrayp_l

# determine the indices for fixed parameters
fixindx = setuputil.fixParams(paramSetup)

# Initialize the sampler with the chosen specs.
if mpi != 'MPI':
    # Single processor with Nthreads cores
    sampler = PTSampler(Ntemps, Nwalkers, nparams, logl, logp, threads=Nthreads)
    #sampler = EnsembleSampler(Nwalkers, nparams, lnprob, 
    #        threads=Nthreads)
else:
    # Multiple processors using MPI
    sampler = PTSampler(Ntemps, Nwalkers, nparams, logl, logp, pool=pool)
    #sampler = EnsembleSampler(Nwalkers, nparams, lnprob, pool=pool)

# Sample, outputting to a file
os.system('date')

# pos is the position of the sampler
# prob the Ln probability
# state the random number generator state
# amp the metadata 'blobs' associated with the current positoni
#for pos, prob, state, amp in sampler.sample(pzero, iterations=10000):
for pos, prob, like in sampler.sample(pzero, iterations=10000):

    print("Mean acceptance fraction: {:f}".
          format(numpy.mean(sampler.acceptance_fraction)))
    os.system('date')
    #ff.write(str(prob))
    #superpos = numpy.zeros(1 + nparams + nmu)
    superpos = numpy.zeros(1 + nparams)

    for ti in range(Ntemps):
        for wi in range(Nwalkers):
            superpos[0] = prob[ti, wi]
            superpos[1:nparams + 1] = pos[ti, wi]
            #superpos[nparams + 1:nparams + nmu + 1] = amp[ti, wi]
            posteriordat.add_row(superpos)
    posteriordat.write('posteriorpdf.hdf5', 
            path = '/posteriorpdf', overwrite=True, compression=True)
    #posteriordat.write('posteriorpdf.txt', format='ascii')
