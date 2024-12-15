# Insert below the path of the developer version of pyBBarolo (= the git repo)
import sys
sys.path.insert(0,"/Users/blanca/Documents/TESIS/software/Bbarolo-1.7")


import numpy as np
import multiprocessing
from pyBBarolo import Param, Rings, FitsCube
from pyBBarolo.BB_interface import libBB


from dynesty import plotting as dyplot
from dynesty import DynamicNestedSampler
import matplotlib.pyplot as plt
import pickle
from astropy.table import Table


###################################################################
######## USER INPUTS ##############################################
###################################################################
fitsname = "./models/model1.fits"
freepar  = ['vrot','vdisp_single','inc_single','phi_single']
#            'z0_single','xpos_single','ypos_single','vsys_single']


# Rings
radii = np.arange(30,240,60)
rings = dict(xpos=25.5,ypos=25.5,vsys=0.0,\
             vrot=120,vdisp=8,vrad=0,\
             z0=30,inc=60,phi=30,dens=1.E20,\
             vvert=0.,dvdz=0.,zcyl=0.)


# Additional parameters for BB
params = dict(mask="SMOOTH&SEARCH",linear=0, bweight=0)


# Define flat prior boundaries for EACH free parameter
bounds = dict(vrot  = [0,250],
              vdisp = [0,40],
              inc   = [50,70],
              phi   = [20,40],
              z0    = [0,60],
              xpos  = [20,30],
              ypos  = [20,30],
              vsys  = [-20,20]
)
###################################################################
###################################################################


multiprocessing.set_start_method('spawn', force=True)
threads  = multiprocessing.cpu_count()


# Creating the needed C++ objects: Rings(), Params() and FitsCube()
opts = Param(fitsfile=fitsname)
opts.add_params(verbose=False,**params)
opts.make_object()


inri = Rings(len(radii))
inri.set_rings(radii=radii,**rings)


inp = FitsCube(fitsname)


mod = libBB.Galfit_new_par(inp._cube,inri._rings,opts._params)

print(mod)
a = plt.plot(mod)
plt.show()


# Determining the number of parameters to fit and indexes for theta
freepar_idx = {}
ndim = 0
freepar_names = []


for p in freepar:
    s = p.split('_')
    if s[0] not in inri.r:
        raise ValueError(f"ERROR! The requested free parameter is unknown: {s[0]}")
    if len(s)==2 and 'single' in p:
        freepar_idx[s[0]] = np.array([ndim])
        ndim += 1
        freepar_names.append(s[0])
    elif len(s)==1:
        freepar_idx[s[0]] = np.arange(ndim,ndim+inri.nr,dtype='int')
        ndim += inri.nr
        for i in range(inri.nr):
            freepar_names.append(f'{s[0]}{i+1}')




def log_likelihood(theta):


    rings = inri
    for key in freepar_idx:
        pvalue = theta[freepar_idx[key]]
        if len(pvalue)==1: pvalue = pvalue[0]
        rings.modify_parameter(key,pvalue)
    rings.make_object()
    
    res = libBB.Galfit_calcresiduals(mod,rings._rings)
        
    return -1000*res




def prior_tranform(u):


    p = np.zeros_like(u)
    for key in freepar_idx:
        p_min,p_max = bounds[key]
        p[freepar_idx[key]] = p_min + u[freepar_idx[key]]*(p_max-p_min)
    return p




if __name__ == "__main__":
    
    with multiprocessing.Pool(threads) as pool:
        
        sampler = DynamicNestedSampler(log_likelihood, prior_tranform, ndim=ndim,\
                                       bound='multi',pool=pool,queue_size=pool._processes)
        sampler.run_nested()
        results = sampler.results
    
    # Extract the best-fit parameters
    samples = results.samples  # Posterior samples
    weights = np.exp(results.logwt - results.logz[-1])
    params = np.average(samples, axis=0, weights=weights)
        
    #print (params)
    
    # Save results with pickle
    with open("dynesty_results.pkl", "wb") as f:
        pickle.dump(results, f)
        
    # To load it back later
    #with open("dynesty_results.pkl", "rb") as f:
    #    loaded_results = pickle.load(f)
    
    
    pp, err_up, err_low = np.zeros(shape=(3,len(params)))
    for i in range(len(params)):
        mcmc = np.percentile(samples[:, i], [15.865, 50, 84.135])
        q = np.diff(mcmc)
        txt = "%10s = %10.3f %+10.3f %+10.3f"%(freepar_names[i],mcmc[1],-q[0],q[1])
        print (txt)
        pp[i] = mcmc[1]
        err_low[i] = q[0]
        err_up[i] = q[1]


    quantiles = [0.16,0.50,0.84]
    cfig, caxes = dyplot.cornerplot(results,show_titles=True,title_quantiles=quantiles,
                                    quantiles=quantiles, color='blue',max_n_ticks=5, labels=freepar_names, \
                                    label_kwargs=dict(fontsize=20))
    cfig.savefig('corner.pdf',bbox_inches='tight')
    
    
    # Compare with "real" input parameters
    t = Table.read(fitsname[:-5]+'_params.txt',format='ascii')


    nrows = int(np.ceil(len(freepar)/2))
    fig,ax = plt.subplots(nrows=nrows,ncols=2)
    ax = np.ravel(ax)
    for a in ax:
        a.set_xlabel("radius (arcsec)")


    rad_mc = inri.r['radii']
    
    for i in range (len(freepar)):
        fp = freepar[i]
        isOne = False
        if '_single' in fp:
            fp = fp.replace('_single','') 
            isOne = True
        aa = fp.upper()
        if aa=="PHI": aa="PA"
        idx = freepar_idx[fp]
        ax[i].set_ylabel(fp)
        ax[i].plot(t['RADII'],t[aa],label='input')
        
        if isOne:
            ax[i].axhspan(pp[idx][0]-err_low[idx][0],pp[idx][0]+err_up[idx][0],alpha=0.2,color='orange')
            ax[i].axhline(pp[idx][0],ls='--',color='orange')
            
        else:
            ax[i].errorbar(rad_mc,pp[idx],fmt='o',yerr=[err_low[idx],err_up[idx]],label='recovered')
        
        if i==0:
            ax[i].legend(loc='lower right')


    '''
    idx = freepar_idx['vrot']
    ax[0].plot(t['RADII'],t['VROT'],label='input')
    ax[0].errorbar(rad_mc,pp[idx],fmt='o',yerr=[err_low[idx],err_up[idx]],label='recovered')
    ax[0].set_ylabel("vrot (km/s)")
    ax[0].legend(loc='lower right')


    idx = freepar_idx['vdisp']
    ax[1].plot(t['RADII'],t['VDISP'])
    ax[1].errorbar(rad_mc,pp[idx],fmt='o',yerr=[err_low[idx],err_up[idx]])
    ax[1].set_ylabel("vdisp (km/s)")


    idx = freepar_idx['inc']
    ax[2].plot(t['RADII'],t['INC'])
    ax[2].axhspan(pp[idx][0]-err_low[idx][0],pp[idx][0]+err_up[idx][0],alpha=0.2,color='orange')
    ax[2].axhline(pp[idx][0],ls='--',color='orange')
    ax[2].set_ylabel("inc (deg)")
    ax[2].set_ylim(t['INC'][0]-20,t['INC'][0]+20)


    idx = freepar_idx['phi']
    ax[3].plot(t['RADII'],t['PA'])
    ax[3].axhspan(pp[idx][0]-err_low[idx][0],pp[idx][0]+err_up[idx][0],alpha=0.2,color='orange')
    ax[3].axhline(pp[idx][0],ls='--',color='orange')
    ax[3].set_ylabel("phi (deg)")
    ax[3].set_ylim(t['PA'][0]-10,t['PA'][0]+10)


    idx = freepar_idx['z0']
    ax[4].plot(t['RADII'],t['Z0'])
    ax[4].axhspan(pp[idx][0]-err_low[idx][0],pp[idx][0]+err_up[idx][0],alpha=0.2,color='orange')
    ax[4].axhline(pp[idx][0],ls='--',color='orange')
    ax[4].set_ylabel("z0 (arcs)")
    ax[4].set_ylim(t['Z0'][0]-30,t['Z0'][0]+30)


    idx = freepar_idx['vsys']
    ax[5].plot(t['RADII'],t['VSYS'])
    ax[5].axhspan(pp[idx][0]-err_low[idx][0],pp[idx][0]+err_up[idx][0],alpha=0.2,color='orange')
    ax[5].axhline(pp[idx][0],ls='--',color='orange')
    ax[5].set_ylabel("vsys (km/s)")
    ax[5].set_ylim(t['VSYS'][0]-30,t['VSYS'][0]+30)


    idx = freepar_idx['xpos']
    ax[6].plot(t['RADII'],t['XPOS'])
    ax[6].axhspan(pp[idx][0]-err_low[idx][0],pp[idx][0]+err_up[idx][0],alpha=0.2,color='orange')
    ax[6].axhline(pp[idx][0],ls='--',color='orange')
    ax[6].set_ylabel("xpos (pix)")
    ax[6].set_ylim(t['XPOS'][0]-5,t['XPOS'][0]+5)


    idx = freepar_idx['ypos']
    ax[7].plot(t['RADII'],t['YPOS'])
    ax[7].axhspan(pp[idx][0]-err_low[idx][0],pp[idx][0]+err_up[idx][0],alpha=0.2,color='orange')
    ax[7].axhline(pp[idx][0],ls='--',color='orange')
    ax[7].set_ylabel("ypos (pix)")
    ax[7].set_ylim(t['YPOS'][0]-5,t['YPOS'][0]+5)
    '''
    fig.savefig('parameters.pdf',bbox_inches='tight')
    
    
        
    
