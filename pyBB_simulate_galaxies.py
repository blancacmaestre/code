###########################################################################
#This script simulates emission-line datacubes using pyBBarolo
#NB: models and model parameters are written in a 'models' directory 
###########################################################################
import os, subprocess
import numpy as np
from pyBBarolo.utils import SimulatedGalaxyCube

# General parameters for datacube
xysize, vsize = 150, 62        # Number of pixels/channels   #Value in model1 = 51,128
pixsize   = 20                  # Size of pixels (arcsec)     #Value in model1 = 20
chwidth   = 5                   # Channel width (km/s)        #Value in model1 = 5
beamFWHM  = 60                  # Beam size (arcsec)          #Value in model1 = 60
modname   = 'model1_4'            # Name of the model           
noiserms  = 0.01                # RMS noise in Jy/beam        #Value in model1 = 0.01

# This the main BBarolo executable
BBmain = "/Users/blanca/Documents/TESIS/software/Bbarolo-1.7/BBarolo"

# Basic parameters of the model
radmax  = 1216.240                        #Value in model1 = 240
radii   = np.arange(0,radmax,pixsize) 
dens    = 50*np.exp(-radii/400-100/(0.5*radii+100))
vrot    = 2./np.pi*200.*np.arctan(radii/10.)
vdisp   = np.full(len(radii),10.)         #Value in model1 = 10
pa      = np.full(len(radii),30.)         #Value in model1 = 30
inc     = np.full(len(radii),60.)         #Value in model1 = 60
z0      = np.full(len(radii),10.)         #Value in model1 = 30
vsys    = np.full(len(radii),132.8)       #Value in model1 = 30

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
s.define_galaxy(radii=radii,vsys=vsys,vdisp=vdisp,inc=inc,pa=pa,\
                dens=dens,z0=z0,vrot=vrot,warpinc=False,warppa=False)
print(s)

# Running BB and creating a model
print (f"Simulating {modname}...",flush=True,end='')
s.run(exe=BBmain,stdout='null',smooth=True,noise=noiserms)
print (f"Done! ")

# Cleaning working directory 
if not os.path.isdir('models'): os.mkdir('models')
mf = f'{modname}mod_noise.fits' if noiserms>0 else f'{modname}mod_nonorm.fits'
subprocess.call([f'{BBmain}','--modhead',f'{modname}/{mf}','OBJECT',f'{modname}'],stdout=subprocess.DEVNULL)
subprocess.call([f'mv',f'{modname}/{mf}',f'models/{modname}.fits'],stdout=subprocess.DEVNULL)
subprocess.call([f'mv',f'{modname}/{modname}_params.txt',f'models/'],stdout=subprocess.DEVNULL)
subprocess.call([f'rm','-rf',f'{modname}','emptycube.fits','galaxy_params.txt'],stdout=subprocess.DEVNULL)

print (f"Model written in models/ directory. See you soon!")

