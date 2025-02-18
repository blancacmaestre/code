###########################################################################
#This script simulates emission-line datacubes using pyBBarolo
#NB: models_new and model parameters are written in a 'models' directory 
###########################################################################
import os, subprocess, sys
import numpy as np
from pyBBarolo.utils import SimulatedGalaxyCube
from astropy.table import Table 
import matplotlib.pyplot as plt
from pyBBarolo.wrapper import PVSlice
from tools import calculate_SNR


BBmain = "/home/user/Bbarolo/BBarolo"
output = "/home/user/THESIS/models/A_MODELS_new/new_attempt/"

# General parameters for datacube
xysize, vsize = 51,64   # Number of pixels/channels   #Value in model1 = 51,128  #value in ngc2403 75, 62
pixsize   = 20  # Size of pixels (arcsec)             #Value in model1 = 20      #value in ngc2403 32 
chwidth   =  10   # Channel width (km/s)                #Value in model1 = 5       #value in ngc2403 -5.12 
beamFWHM  = 60  # Beam size (arcsec)                  #Value in model1 = 60      #value in ngc2403 180 or 360
modname   = "CGal_4_80_0.01"  # Name of the model           
noiserms  = 0.01   # RMS noise in Jy/beam            #Value in model1 = 0.01     #value in ngc2403 0.0015


# Basic parameters of the model

radmax  = 240
radii   = np.arange(0,radmax,pixsize)
#dens   = 20*np.exp(-radii/100-50/(0.5*radii+50))
#vrot   = 2./np.pi*150*np.arctan(radii/30.)
dens    = np.full(len(radii),10)
vrot    = np.full(len(radii),100)
vdisp   = np.full(len(radii),10.)
pa      = np.full(len(radii),0.)
inc     = np.full(len(radii),80.)
z0      = np.full(len(radii),30.)
vsys    = np.full(len(radii),0)

dens_shape = "constant"
vrot_shape = "constant"
vdisp_shape = "constant"

""" radmax  = 300   #Value in model1 = 240
radii   = np.arange(0,radmax,pixsize) 
dens   = 20*np.exp(-radii/100-50/(0.5*radii+100))
dens_shape = "exponential"
vrot    = 2./np.pi*300*np.arctan(radii/50.) #this is the value of model1
vrot_shape = "arctan"
vdisp    = 10 + 10*np.exp(-radii/50.)
vdisp_shape = "exponential"
pa   = np.full(len(radii),0)       #Value in model1 = 30 # actual angle of ngc2405 is 123.7
inc  = np.full(len(radii),45)         #Value in model1 = 60 value in ngc2403 62.9
z0   = np.full(len(radii),10)         #Value in model1 = 30
vsys = np.full(len(radii),0)       #Value in model1 = 0  value in ngc2403 132.8
xpos = 25.5
ypos = 25.5 """

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
s.run(exe=BBmain,stdout='null',smooth=True,noise=noiserms) #HERE IS WHERE THE NOISE IS ADDED
print (f"Done! ")

# Cleaning working directory 
if not os.path.isdir(f'{output}{modname}'):
    os.mkdir(f'{output}{modname}')
mf = f'{modname}mod_noise.fits' if noiserms>0 else f'{modname}mod_nonorm.fits'
subprocess.call([f'{BBmain}','--modhead',f'{modname}/{mf}','OBJECT',f'{modname}'],stdout=subprocess.DEVNULL)
subprocess.call([f'mv',f'{modname}/{mf}',f'{output}{modname}/{modname}.fits'],stdout=subprocess.DEVNULL)
subprocess.call([f'mv',f'{modname}/{modname}_params.txt',f'{output}{modname}/'],stdout=subprocess.DEVNULL)
subprocess.call([f'rm','-rf',f'{modname}','emptycube.fits','galaxy_params.txt'],stdout=subprocess.DEVNULL)

print (f"Model written in models_new/ directory. See you soon!")

SNR, total_SNR = calculate_SNR(f"{output}{modname}/{modname}.fits")
print("SNR: ", total_SNR)   

with open(os.path.join(f"{output}{modname}", f"{modname}_input.txt"), 'w') as file:
    file.write(f"Name of the model = {modname}\n")
    file.write("\n")
    file.write("CUBE PARAMETERS\n")
    file.write("\n")
    file.write(f"Number of pixels in x/y = {xysize}\n")
    file.write(f"Size of pixels (arcsec)= {pixsize}\n")
    file.write(f"Number of channels = {vsize}\n")
    file.write(f"Channel width = {chwidth}\n")
    file.write(f"Beam width = {beamFWHM}\n")
    file.write(f"RMS noise in Jy/beam = {noiserms}\n")
    file.write("\n")
    file.write("GALAXY PARAMETERS\n")
    file.write("\n")
    file.write(f"SNR = {total_SNR}\n")
    file.write(f"Maximum radius of galaxy = {radmax}\n")
    file.write(f"Beam width = {beamFWHM}\n")
    file.write(f"Number of radii = {len(radii)}\n")
    file.write(f"Density profile initial value = {dens[0]}\n")
    file.write(f"Density ptofile shape = {dens_shape}\n")
    file.write("\n")
    file.write("VELOCITIES\n")
    file.write(f"Rotational velocity initial value = {vrot[0]}\n")
    file.write(f"Rotational velocity shape = {vrot_shape}\n")
    file.write(f"Velocity dispersion initial value= {vdisp[0]}\n")
    file.write(f"Velocity dispersion shape = {vdisp_shape}\n")
    file.write("\n")
    file.write("GEOMETRY\n")
    file.write(f"Position angle = {pa[0]}\n")
    file.write(f"Inclination = {inc[0]}\n")
    file.write(f"Thickness = {z0[0]}\n")
    file.write(f"Systemic velocity (km/s) = {vsys[0]}\n")


v = Table.read(f"{output}{modname}/{modname}_params.txt",format='ascii')   
    
major_GAL = PVSlice( fitsname=f"{output}{modname}/{modname}.fits", XPOS_PV = np.mean(v['XPOS']), YPOS_PV = np.mean(v['YPOS']),  PA_PV = np.mean(v['PA']), OUTFOLDER = f"{output}{modname}/slices")              
major_GAL.run(BBmain)
minor_GAL = PVSlice( fitsname=f"{output}{modname}/{modname}.fits", XPOS_PV = np.mean(v['XPOS']), YPOS_PV = np.mean(v['YPOS']),  PA_PV = np.mean(v['PA']+90), OUTFOLDER= f"{output}{modname}/slices")              
minor_GAL.run(BBmain)

print(f"{modname}_initial_params.txt created with default parameters.")
