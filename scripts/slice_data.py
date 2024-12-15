from pyBBarolo.wrapper import PVSlice

BBmain = "/Users/blanca/Documents/TESIS/software/Bbarolo-1.7/BBarolo"

#fitsname = "/Users/blanca/Documents/TESIS/software/code/tests/first_try/barbamodel/barbamodelmod_nonorm.fits"

angle = 30

#pvslice_a = PVSlice( fitsname=fitsname, XPOS_PV=25.5, YPOS_PV=25.5,  PA_PV=angle, OUTFOLDER= "SLICES",  )              
#pvslice_a.run(BBmain)

#THIS IS A CODE TO CALCULATE THE VELOCITY AT A CERTAIN CHANNEL

import numpy as np
from astropy.io import fits

# Load the data cube (FITS file)
cube_file = "/Users/blanca/Documents/TESIS/software/code/OLD_TESTS/models_old/model1.fits"  # Replace with your file path
 # Replace with the actual file path
hdul = fits.open(cube_file)
header = hdul[0].header

# Extract necessary values from the header
CRVAL3 = header['CRVAL3']  # Reference velocity (km/s)
CDELT3 = header['CDELT3']  # Pixel scale along the velocity axis (km/s)
CRPIX3 = header['CRPIX3']  # Reference pixel for the velocity axis

# Example: Calculate the velocity for a specific channel
channel_index = 30  # Replace with the channel index you want (1-based index)

# Calculate the velocity for the given channel
velocity = CRVAL3 + (channel_index - CRPIX3) * CDELT3

# Print the result
print(f"Velocity at channel {channel_index}: {velocity} km/s")



