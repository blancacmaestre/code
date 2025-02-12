#!/usr/bin/env python3

# Insert below the path of the developer version of pyBBarolo (= the git repo)
# This way, we can override any other installed version and 
# modify pyBBarolo without the need of reinstalling it every time
import sys
sys.path.insert(0,'/Users/edt/Dropbox/Codex/BBarolo')

# This should give version 1.3.3dev!
from pyBBarolo import version
print (version)

from pyBBarolo.bayesian import BayesianBBarolo
from dynesty import plotting as dyplot
import matplotlib.pyplot as plt
import numpy as np

# Name of the FITS file to be fitted
model = "model4"
threads = 10
fitsname = f"./models/{model}.fits"
freepar = ['vrot','vdisp','inc_single','phi_single']


# Creating an object for bayesian barolo
f3d = BayesianBBarolo(fitsname)

# Initializing rings. 
f3d.init(radii=np.arange(30,240,60),xpos=25.5,ypos=25.5,vsys=0.0,\
         vrot=100,vdisp=10,vrad=0,z0=30,inc=60,phi=0)

# Here it is possible to give any other BBarolo parameter, for example to control
# the mask, linear, bweight, cdens, wfunc, etc...
f3d.set_options(mask="SEARCH",linear=0,outfolder=f"output/{model}",plotmask=True)
f3d.show_options()

# Default priors are uniform and the default boundaries for the fit are in f3d.bounds.
f3d.bounds['vrot']  = [0,250]
f3d.bounds['vdisp'] = [1,40]
f3d.bounds['inc']   = [60,80]
f3d.bounds['phi']   = [-20,20]
f3d.bounds['z0']    = [0,60]
f3d.bounds['xpos']  = [20,30]
f3d.bounds['ypos']  = [20,30]
f3d.bounds['vsys']  = [-20,20]
f3d.bounds['dens']  = [1,10]

# As an example, here I am overriding the default normalization and residual functions
def my_norm(model,data):
    nrm = np.nansum(data)/np.nansum(model)
    return nrm*model

def my_calcres(model,data,mask):
    data *= mask
    model = my_norm(model,data)
    res = np.nansum((data-model)**2)
    return res

f3d._calculate_residuals = my_calcres
f3d._normalize_model = my_norm

# Keywords to be passed to the sample run
run_kwargs = dict()
# Keywords to be passed to the 
sample_kwargs = dict()

# Running the fit with dynesty.
f3d.compute(threads=threads,useBBres=False,method='dynesty',dynamic=True,
            freepar=freepar,run_kwargs=run_kwargs,sample_kwargs=sample_kwargs)

print (f3d.params,f3d._log_likelihood(f3d.params))

# Writing best model and plots (experimental, to be checked)
f3d.write_bestmodel()

# Print some statistics of the sample
f3d.print_stats()

# Plot the 2-D marginalized posteriors.
quantiles = [0.16,0.50,0.84]
cfig, caxes = dyplot.cornerplot(f3d.results,show_titles=True,title_quantiles=quantiles,
                                quantiles=quantiles, color='blue',max_n_ticks=5, \
                                labels=f3d.freepar_names, label_kwargs=dict(fontsize=20))
cfig.savefig(f'output/{model}/{model}_corner.pdf',bbox_inches='tight')

# Saving samples
np.save("dynesty_samples.npy", f3d.results.samples)
