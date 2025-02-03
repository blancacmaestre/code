###########################################################################
#This script simulates emission-line datacubes using pyBBarolo
#NB: models and model parameters are written in a 'models' directory 
###########################################################################
import os, subprocess
import numpy as np
from pyBBarolo.utils import SimulatedGalaxyCube

# General parameters for datacube
xysize, vsize = 51, 64          # Number of pixels/channels
pixsize   = 20                  # Size of pixels (arcsec)
chwidth   = 10                  # Channel width (km/s)
beamFWHM  = 60                  # Beam size (arcsec)
modname   = 'model4'            # Name of the model
noiserms  = 0.01                # RMS noise in Jy/beam

# This the main BBarolo executable
BBmain = "/Users/edt/Dropbox/Codex/BBarolo/BBarolo"

# Basic parameters of the model
radmax  = 240
radii   = np.arange(0,radmax,pixsize)
#dens    = 20*np.exp(-radii/100-50/(0.5*radii+50))
#vrot    = 2./np.pi*150*np.arctan(radii/30.)
dens     = np.full(len(radii),10)
vrot     = np.full(len(radii),100)
vdisp   = np.full(len(radii),10.)
pa      = np.full(len(radii),0.)
inc     = np.full(len(radii),70.)
z0      = np.full(len(radii),30.)

# Below if we wanna define a warp or a flare (these are linear for example)
#rstart = 150
#pa[radii>rstart] = -(radii[radii>rstart]-rstart)/20.+pa[0]
#inc[radii>rstart] = -(radii[radii>rstart]-rstart)/80.+inc[0]
#z0[radii>rstart] = (radii[radii>rstart]-rstart)/4. + z0[0]

# Initializing a SimulatedGalaxyCube instance
s = SimulatedGalaxyCube(axisDim=[xysize, xysize, vsize],\
                        cdelts=[-pixsize/3600., pixsize/3600., chwidth],\
                        beam=beamFWHM/3600., bunit='JY/BEAM', obj=modname)

# Setting up galaxy parameters (if parameters are not given, they are random!)
s.define_galaxy(radii=radii,vsys=0,vdisp=vdisp,inc=inc,pa=pa,\
                dens=dens,z0=z0,vrot=vrot,warpinc=False,warppa=False)

# Running BB and creating a model
print (f"Simulating {modname}...",flush=True,end='')
s.run(exe=BBmain,stdout='null',smooth=True,noise=noiserms,linear=0)
print (f"Done! ")

# Cleaning working directory 
if not os.path.isdir('models'): os.mkdir('models')
mf = f'{modname}mod_noise.fits' if noiserms>0 else f'{modname}mod_nonorm.fits'
subprocess.call([f'{BBmain}','--modhead',f'{modname}/{mf}','OBJECT',f'{modname}'],stdout=subprocess.DEVNULL)
subprocess.call([f'mv',f'{modname}/{mf}',f'models/{modname}.fits'],stdout=subprocess.DEVNULL)
subprocess.call([f'mv',f'{modname}/{modname}_params.txt',f'models/'],stdout=subprocess.DEVNULL)
subprocess.call([f'rm','-rf',f'{modname}','emptycube.fits','galaxy_params.txt'],stdout=subprocess.DEVNULL)

print (f"Model written in models/ directory. See you soon!")


