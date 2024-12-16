from pyBBarolo.wrapper import PVSlice
from astropy.table import Table
import numpy as np
from astropy.io import fits
BBmain = "/Users/blanca/Documents/TESIS/software/Bbarolo-1.7/BBarolo" 

##########################################################################################################################################################

#THIS IS A CODE TO GET THE PV SLICES

def getslice(ringfile, fitsfile, output):
    v = Table.read(ringfile,format='ascii') 
    major_MOD = PVSlice( fitsname=fitsfile, XPOS_PV = np.mean(v['XPOS']), YPOS_PV = np.mean(v['YPOS']),  PA_PV = np.mean(v['PA']), OUTFOLDER = output)              
    major_MOD.run(BBmain)
    minor_MOD = PVSlice( fitsname=fitsfile, XPOS_PV = np.mean(v['XPOS']), YPOS_PV = np.mean(v['YPOS']),  PA_PV = np.mean(v['PA']+90), OUTFOLDER= output)              
    minor_MOD.run(BBmain)
    print(f"The slices are completed")

##########################################################################################################################################################

#THIS IS A CODE TO CALCULATE THE VELOCITY AT A CERTAIN CHANNEL

def getthevelocity(cube_file,channel_index):
    hdul = fits.open(cube_file)
    header = hdul[0].header

    CRVAL3 = header['CRVAL3']  
    CDELT3 = header['CDELT3']  
    CRPIX3 = header['CRPIX3']  
    velocity = CRVAL3 + (channel_index - CRPIX3) * CDELT3
    print(f"Velocity at channel {channel_index}: {velocity} km/s")
    return velocity

#THIS IS A CODE TO CALCULATE THE CHANNEL AT A CERTAIN VELOCITY

def getthechannel(cube_file,vel):
    hdul = fits.open(cube_file)
    header = hdul[0].header

    crpix3 = header['CRPIX3']  
    crval3 = header['CRVAL3']  
    cdelt3 = header['CDELT3']  
    channel = crpix3 + (vel - crval3) / cdelt3
    print(f"The channel for velocity {vel} km/s is {channel}")
    return vel

##########################################################################################################################################################
