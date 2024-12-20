###########################################################################
#This script simulates emission-line datacubes using pyBBarolo
#NB: models and model parameters are written in a 'models' directory 
###########################################################################
import os, subprocess
import numpy as np
from pyBBarolo.utils import SimulatedGalaxyCube
from astropy.table import Table 
import matplotlib.pyplot as plt
from pyBBarolo.wrapper import PVSlice
BBmain = "/Users/blanca/Documents/TESIS/software/Bbarolo-1.7/BBarolo"

# General parameters for datacube
xysize, vsize = 51,128   # Number of pixels/channels   #Value in model1 = 51,128  #value in ngc2403 75, 62
pixsize   = 20   # Size of pixels (arcsec)             #Value in model1 = 20      #value in ngc2403 32 
chwidth   =  5   # Channel width (km/s)                #Value in model1 = 5       #value in ngc2403 -5.12 
beamFWHM  = 60   # Beam size (arcsec)                  #Value in model1 = 60      #value in ngc2403 180 or 360
modname   = 'velset_inc75'   # Name of the model           
noiserms  = 0.001   # RMS noise in Jy/beam            #Value in model1 = 0.01     #value in ngc2403 0.0015


ringfile = "/Users/blanca/Documents/TESIS/software/code/models/ngc2403_180/rings_final1.txt"
t = Table.read(ringfile,format='ascii')

# Basic parameters of the model
radmax  = 300   #Value in model1 = 240
radii   = np.arange(0,radmax,pixsize) 
#when i am using the data from another galaxy i needto match this to the data 
#when i am creating a model i can put from 0 and say that the jump is pxsize

#dens    = np.full(len(radii),1.)
#dens_shape = "constant at 1"
dens   = 50*np.exp(-radii/400-100/(0.5*radii+100))
dens_shape = "enrico"

#vrot   = t['VROT(km/s)']
#vrot_shape = "ngc 2403"
#vrot   = np.full(len(radii),200.)
vrot = np.linspace(100, 250, len(radii)).tolist()
vrot_shape = "linearly decreasing from 200km/s to 100km/s"
#vrot    = 200 + 2./np.pi*200*np.arctan(2 *radii/270.) #this is the value of model1
#vrot_shape = "arctan (enrico)"

#vdisp   = np.linspace(30, 10, len(radii)).tolist()
vdisp   = np.full(len(radii),20.)        #Value in model1 = 10
#vdisp    = 10 + 30*np.exp(-radii/180.)
vdisp_shape = "constant at 20km/s"

pa   = np.full(len(radii),80)       #Value in model1 = 30 # actual angle of ngc2405 is 123.7
inc  = np.full(len(radii),75)         #Value in model1 = 60 value in ngc2403 62.9
z0   = np.full(len(radii),0)         #Value in model1 = 30
vsys = np.full(len(radii),30)       #Value in model1 = 0  value in ngc2403 132.8
xpos = 25.5
ypos = 25.5

# Below if we wanna define a warp or a flare (these are linear for example)
#rstart = 150
#pa[radii>rstart] = -(radii[radii>rstart]-rstart)/20.+pa[0]
#inc[radii>rstart] = -(radii[radii>rstart]-rstart)/80.+inc[0]
#z0[radii>rstart] = (radii[radii>rstart]-rstart)/4. + z0[0]
#BPA= -79.4

# Initializing a SimulatedGalaxyCube instance

#crpixs=[38.,38., 1]
#crvals=[114.22, 65.58, 289.61]

#This is the base datacube GALMOD needs
s = SimulatedGalaxyCube(axisDim=[xysize, xysize, vsize],\
                        cdelts=[-pixsize/3600., pixsize/3600., chwidth],\
                        #crpixs=crpixs,\
                        #crvals=crvals,\
                        beam=beamFWHM/3600., bunit='JY/BEAM', obj=modname)

# Setting up galaxy parameters (if parameters are not given, they are random!)
s.define_galaxy(xpos=xpos, ypos=ypos, radii=radii,vsys=vsys,vdisp=vdisp,inc=inc,pa=pa,\
                dens=dens,z0=z0,vrot=vrot,warpinc=False,warppa=False)
print(s)

# Running BB and creating a model
print (f"Simulating {modname}...",flush=True,end='')
s.run(exe=BBmain,stdout='null',smooth=True,noise=noiserms) #HERE IS WHERE THE NOISE IS ADDED
#this is where the galaxy is actually created
print (f"Done! ")

# Cleaning working directory 
if not os.path.isdir(f'models/{modname}'): os.mkdir(f'models/{modname}')
mf = f'{modname}mod_noise.fits' if noiserms>0 else f'{modname}mod_nonorm.fits'
subprocess.call([f'{BBmain}','--modhead',f'{modname}/{mf}','OBJECT',f'{modname}'],stdout=subprocess.DEVNULL)
subprocess.call([f'mv',f'{modname}/{mf}',f'models/{modname}/{modname}.fits'],stdout=subprocess.DEVNULL)
subprocess.call([f'mv',f'{modname}/{modname}_params.txt',f'models/{modname}/'],stdout=subprocess.DEVNULL)
subprocess.call([f'rm','-rf',f'{modname}','emptycube.fits','galaxy_params.txt'],stdout=subprocess.DEVNULL)


print (f"Model written in models/ directory. See you soon!")
 

with open(os.path.join(f"models/{modname}", f"{modname}_input.txt"), 'w') as file:
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
    file.write(f"xpos and ypos = {xpos},{ypos}\n")
    #file.write(f"crvals = {crvals}\n")
    #file.write(f"crpixs = {crpixs}\n")
    file.write("\n")
    file.write("GALAXY PARAMETERS\n")
    file.write("\n")
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

v = Table.read(f"models/{modname}/{modname}_params.txt",format='ascii')   
    
major_GAL = PVSlice( fitsname=f"models/{modname}/{modname}.fits", XPOS_PV = np.mean(v['XPOS']), YPOS_PV = np.mean(v['YPOS']),  PA_PV = np.mean(v['PA']), OUTFOLDER = f"models/{modname}/slices")              
major_GAL.run(BBmain)
minor_GAL = PVSlice( fitsname=f"models/{modname}/{modname}.fits", XPOS_PV = np.mean(v['XPOS']), YPOS_PV = np.mean(v['YPOS']),  PA_PV = np.mean(v['PA']+90), OUTFOLDER= f"models/{modname}/slices")              
minor_GAL.run(BBmain)


print(f"{modname}_initial_params.txt created with default parameters.")
