
# Insert below the path of the developer version of pyBBarolo (= the git repo)
# This way, we can override any other installed version and 
# modify pyBBarolo without the need of reinstalling it every time
import sys
sys.path.insert(0,'/Users/blanca/Documents/TESIS/software/Bbarolo/BBarolo')

# This should give version 1.3.3dev!
from pyBBarolo import version
print (version)

from contextlib import redirect_stdout, redirect_stderr, ExitStack
from pyBBarolo.bayesian import BayesianBBarolo
from dynesty import plotting as dyplot
import matplotlib.pyplot as plt
import numpy as np
from pyBBarolo.BB_interface import libBB
import ctypes, time, copy
from pyBBarolo import Param, Rings, FitMod3D, reshapePointer, vprint, isIterable
import os
import gc
import corner

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self):
        for f in self.files:
            f.flush()


def my_norm(model,data):
    nrm = np.nansum(data)/np.nansum(model)
    return nrm*model

#Res collection
def res_Gaussian(data,model,noise,mask,multiplier=1):
    """
    Gaussian residuals with noise
    """

    data *= mask
    model = my_norm(model,data)
    res = np.nansum((data-model)**2)

    return multiplier*res/(noise*noise)

def res_abs(data,model,noise,mask,multiplier=1):
    """
    Use absolute difference
    """
    
    data *= mask
    model = my_norm(model,data)
    res = np.nansum(np.abs(data-model))

    return multiplier*res/noise





##HERE WE MODIFY THE MAIN Byesian Barolo  implementation
class BayesianBBaroloMod(BayesianBBarolo):

    def __init__(self,fitsname,**kwargs):
        """ Initialize the BayesianBBarolo class.
    
        Parameters
        ----------
        fitsname : str
            The name of the fits file with the datacube to fit.
        **kwargs : dict
            Any other parameter to be passed to the BBarolo's library.
        """

        super(BayesianBBaroloMod,self).__init__(fitsname,**kwargs)
        self.noise=None
        self.Npix=None

    
    def _log_likelihood(self,theta):
        """ Likelihood function for the fit """
        
        rings = self._update_rings(self._inri,theta)

        # Calculate likelihood 
        if self.noise is None: self.noise=np.nanstd(self.data[0,:,:])
        if self.Npix is None:  self.Npix=np.prod(self.data.shape)

        if self.useBBres:
            # Calculating residuals through BBarolo directly
            res = libBB.Galfit_calcresiduals(self._galfit,rings._rings)
        else: 
            # Calculating residuals manually            

            # Recompute the density profile along the current rings and update the rings
            if self.useNorm and self.update_prof:
                self._update_profile(rings)

            # Calculate the model and the boundaries
            mod, bhi, blo, galmod = self._calculate_model(rings)
            
            # Calculate the residuals
            mask = self.mask[:,blo[1]:bhi[1],blo[0]:bhi[0]]
            data = self.data[:,blo[1]:bhi[1],blo[0]:bhi[0]]
            res  = self._calculate_residuals(mod,data,mask)

            libBB.Galmod_delete(galmod)
            
        return -res

    #Uncomment
    #Gaussian 
    def _calculate_residuals(self,model,data,mask=None):
        
        #Option A: Standard absolute residuals: no noise, residual muplitied by 1000 as done before
        #res=res_abs(model=model, data=data, noise=1, mask=mask, multiplier=1000)

        #Option B Standard absolute residuals: cube noise, residual muplitied by 1000 as done before
        #res=res_abs(model=model, data=data, noise=self.noise, mask=mask, multiplier=1000)

        #Option C Standard Gaussian residuals: cube noise,
        #res=res_Gaussian(model=model, data=data, noise=self.noise, mask=mask, multiplier=1)

        #Option D Gaussian residuals: no noise
        res=res_Gaussian(model=model, data=data, noise=1, mask=mask, multiplier=1)

        #Option E Gaussian residuals: no noise, multiplied by 1000
        #res=res_Gaussian(model=model, data=data, noise=1, mask=mask, multiplier=1000)

        #Option F Gaussian residuals: cube noise, multiplied by 1000    
        #res=res_Gaussian(model=model, data=data, noise=self.noise, mask=mask, multiplier=1000)


        return res
    


# Name of the FITS file to be fitted
model = "CGal_5_70_0.01_D_3000"
centre = 38.5
fitsname = f"/home/user/THESIS/models/A_MODELS_new/new_PA/CGal_5_70_0.01/CGal_5_70_0.01.fits"
#freepar = ['inc_single']
#freepar = ['vrot','vdisp']
freepar = ['vrot','vdisp','inc_single','phi_single']
#Uncomment to fit the density
#freepar = ['vrot','vdisp','dens','inc_single','phi_single']
output = "/home/user/THESIS/tests_all_resolution/newPA2"

# Creating an object for bayesian barolo
f3d = BayesianBBaroloMod(fitsname)

# Initializing rings. 
f3d.init(radii=np.arange(24,240,48),xpos=centre,ypos=centre,vsys=0.0,\
         vrot=100,vdisp=10,vrad=0,z0=30,inc=70,phi=0)

# Here it is possible to give any other BBarolo parameter, for example to control
# the mask, linear, bweight, cdens, wfunc, etc...
f3d.set_options(mask="SEARCH",linear=0,outfolder=f"{output}/{model}",plotmask=True, threads=8)
#To remove the mask
#f3d.set_options(mask="NONE",linear=0,outfolder=f"output/{model}",plotmask=True)

f3d.show_options()

# Default priors are uniform and the default boundaries for the fit are in f3d.bounds.
""" f3d.bounds['vrot']  = [0,250]
f3d.bounds['vdisp'] = [1,40]
f3d.bounds['inc']   = [20,80]
f3d.bounds['phi']   = [-20,20]
f3d.bounds['z0']    = [0,60]
f3d.bounds['xpos']  = [20,30]
f3d.bounds['ypos']  = [20,30]
f3d.bounds['vsys']  = [-20,20]
f3d.bounds['dens']  = [1,30] """




""" f3d.priors['inc']["loc"]= [10]
f3d.priors['inc']["scale"]= [80]
f3d.priors['phi']["loc"]= [10]
f3d.priors['phi']["scale"]= [300]   """   


# Keywords to be passed to the sample run
run_kwargs = dict()
# Keywords to be passed to the 
sample_kwargs = dict()

# Running the fit with dynesty.
f3d.compute(threads=8,useBBres=False,method='dynesty',dynamic=True,
            freepar=freepar,run_kwargs=run_kwargs,sample_kwargs=sample_kwargs)

print (f3d.params,f3d._log_likelihood(f3d.params))

output_file_path = os.path.join(output, model, "results_summary.txt")
with open(output_file_path, 'w') as f:
    tee = Tee(sys.stdout, f)
    with ExitStack() as stack:
        stack.enter_context(redirect_stdout(tee))
        stack.enter_context(redirect_stderr(tee))

        # Writing best model and plots (experimental, to be checked)
        f3d.write_bestmodel()

        # Print some statistics of the sample
        f3d.print_stats()

        # Print summary of results
        f3d.results.summary()

#corner.corner(
#    f3d.samples, weights=f3d.weights, labels=f3d.freepar_names, color='purple',
#    plot_datapoints=False, label_kwargs=dict(fontsize=20))


# Plot the 2-D marginalized posteriors.
quantiles = [0.16,0.50,0.84]
cfig, caxes = dyplot.cornerplot(f3d.results,show_titles=True,truth_color='black',title_quantiles=quantiles,
                                quantiles=quantiles, color='purple',max_n_ticks=5, \
                                labels=f3d.freepar_names, label_kwargs=dict(fontsize=20))
cfig.savefig(f'{output}/{model}/{model}_corner.pdf',bbox_inches='tight')

tfig, axes = dyplot.traceplot(f3d.results,
                             truth_color='black', show_titles=True,
                             trace_cmap='viridis', connect=True,
                             connect_highlight=range(5))
tfig.savefig(f'{output}/{model}/{model}_trace.pdf',bbox_inches='tight')
# Saving samples
np.save("dynesty_samples.npy", f3d.results.samples)

del f3d
gc.collect()
