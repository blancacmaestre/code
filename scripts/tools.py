from pyBBarolo.wrapper import PVSlice
from astropy.table import Table
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from spectral_cube import SpectralCube as SC
from astropy import wcs
from astropy import units as u
import numpy.ma as ma
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
    velocity = int(CRVAL3 + (channel_index - CRPIX3) * CDELT3)
    print(f"Velocity at channel {channel_index}: {velocity} km/s")
    return velocity

#THIS IS A CODE TO CALCULATE THE CHANNEL AT A CERTAIN VELOCITY

def getthechannel(cube_file,vel):
    hdul = fits.open(cube_file)
    header = hdul[0].header

    crpix3 = header['CRPIX3']  
    crval3 = header['CRVAL3']  
    cdelt3 = header['CDELT3']  
    channel = int(crpix3 + (vel - crval3) / cdelt3)
    print(f"The channel for velocity {vel} km/s is {channel}")
    return channel

##########################################################################################################################################################

#THIS IS A CODE TO CALCULATE THE SNR

def calculate_SNR(fname):
    image = fits.getdata(fname) 
    SNR = []
    k_values = []
    k = 0
    while k < image.shape[0]:
        mean, median, std = sigma_clipped_stats(image[k], sigma=100)
        mx = ma.masked_array(image[k], mask=image[k] < mean + 3 * std)

        if mx.compressed().size == 0 or mx.data[mx.mask].size == 0:
            k += 1
            continue

        in_mask_mean = np.mean(mx.compressed())  # Data that i see
        out_mask_std = np.std(mx.data[mx.mask])  # Data eliminated by mask

        if np.isnan(in_mask_mean) or np.isnan(out_mask_std):
            if k > 1:
                SNR = SNR[:-2]  # Remove the last 2 values
                k_values = k_values[:-2]
            k += 2  # Skip the next 2 values
            continue

        SNR.append(in_mask_mean / out_mask_std)
        k_values.append(k)
        k += 1

    SNR = np.array(list(zip(k_values, SNR)))
    total_SNR = np.mean(SNR[:, 1])
    return SNR, total_SNR

##########################################################################################################################################################

#THIS IS A CODE TO PLOT THE CHANNELS

def plot_channels(fname, skip=3, with_contours=True):
    image = fits.getdata(fname)  # numpy array, my data
    # row number is y coordinate so it is (rows, columns), (y,x) this is for a 2d one, but if i take a 3d? (z,y,x)

    num_slices = image.shape[0] // skip  # Only take 1/skip of the slices
    num_cols = 3
    num_rows = (num_slices + num_cols - 1) // num_cols  # Calculate the number of rows needed

    fig, ax = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    ax = ax.flatten()  # Flatten the array to make indexing easier
    for k in range(0, image.shape[0], skip):  # Skip channels based on the skip parameter
        mean, median, std = sigma_clipped_stats(image[k], sigma=5.0)
        if with_contours:
            cont_level_k = [std * x for x in [-1, 1, 2, 4, 8, 16, 32, 64]]
            ax[k // skip].contour(image[k], levels=cont_level_k, cmap='viridis')
        ax[k // skip].imshow(image[k], cmap="Greys", origin="lower")

    # Hide any unused subplots
    for i in range((image.shape[0] + skip - 1) // skip, len(ax)):
        fig.delaxes(ax[i])

    plt.show()

##########################################################################################################################################################

#THIS IS A CODE TO PLOT THE PV SLICES

def plot_PV(fname1, fname2, ringfile1, ringfile2, output):
    c = Table.read(ringfile1, format='ascii')
    cube_maj = PVSlice(fitsname=fname1, XPOS_PV=np.mean(c['XPOS']), YPOS_PV=np.mean(c['YPOS']), PA_PV=np.mean(c['PA']), OUTFOLDER=output)
    cube_maj.run(BBmain)
    
    cube_min = PVSlice(fitsname=fname1, XPOS_PV=np.mean(c['XPOS']), YPOS_PV=np.mean(c['YPOS']), PA_PV=np.mean(c['PA'] + 90), OUTFOLDER=output)
    cube_min.run(BBmain)
    c_head = fits.open(fname1)
    name1 = c_head[0].header["OBJECT"]

    m = Table.read(ringfile2, format='ascii')
    model_maj = PVSlice(fitsname=fname2, XPOS_PV=np.mean(m['XPOS']), YPOS_PV=np.mean(m['YPOS']), PA_PV=np.mean(m['PA']), OUTFOLDER=output)
    model_maj.run(BBmain)

    model_min = PVSlice(fitsname=fname2, XPOS_PV=np.mean(m['XPOS']), YPOS_PV=np.mean(m['YPOS']), PA_PV=np.mean(m['PA'] + 90), OUTFOLDER=output)
    model_min.run(BBmain)
    m_head = fits.open(fname2)
    name2 = m_head[0].header["OBJECT"]

    im_cube_maj = fits.getdata(f'{output}{name1}pv_{int(np.mean(c['PA']))}.fits')
    im_cube_min = fits.getdata(f'{output}{name1}pv_{int(np.mean(c['PA'])+90)}.fits')
    im_model_maj = fits.getdata(f'{output}{name2}pv_{int(np.mean(c['PA']))}.fits')
    im_model_min = fits.getdata(f'{output}{name2}pv_{int(np.mean(c['PA'])+90)}.fits')

    fig, axs = plt.subplots(1, 2, figsize=(15, 12))
    mean1, meadian1, std1 = sigma_clipped_stats(im_cube_maj, sigma=5.0)
    cont_level_1 = [std1 * x for x in [-1, 1, 2, 4, 8, 16, 32, 64]]
    im1 = axs[0].imshow(im_cube_maj, cmap="Reds", origin="lower")
    axs[0].contour(im_cube_maj, levels=cont_level_1, cmap='Reds_r', linewidths=1)
    fig.colorbar(im1, ax=axs[0])

    mean2, meadian2, std2 = sigma_clipped_stats(im_model_maj, sigma=5.0)
    cont_level_2 = [std2 * x for x in [-1, 1, 2, 4, 8, 16, 32, 64]]
    axs[0].contour(im_model_maj, levels=cont_level_2, cmap='Blues_r', linewidths=1)
    axs[0].imshow(im_model_maj, cmap="Blues", origin="lower", alpha=0.5)
    axs[0].set_title("Slice at angle " + str(int(np.mean(c['PA']))) + "°")
    axs[0].set_xlabel('Offset(arcsec)')
    axs[0].set_ylabel('LOS velocity (km/s)')

    mean3, meadian3, std3 = sigma_clipped_stats(im_cube_min, sigma=5.0)
    cont_level_3 = [std3 * x for x in [-1, 1, 2, 4, 8, 16, 32, 64]]
    im2 = axs[1].imshow(im_cube_min, cmap='Reds', origin='lower')
    axs[1].contour(im_cube_min, levels=cont_level_3, cmap='Reds_r', linewidths=1)
    fig.colorbar(im2, ax=axs[1])

    mean4, meadian4, std4 = sigma_clipped_stats(im_model_min, sigma=5.0)   
    cont_level_4 = [std4 * x for x in [-1, 1, 2, 4, 8, 16, 32, 64]]
    axs[1].contour(im_model_min, levels=cont_level_4, cmap='Blues_r', linewidths=1)
    axs[1].imshow(im_model_min, cmap='Blues', origin='lower', alpha=0.5)
    axs[1].set_title("Slice at angle " + str(int(np.mean(c['PA']))+90) + "°")
    axs[1].set_xlabel('Offset(arcsec)')
    axs[1].set_ylabel('LOS velocity (km/s)')

    plt.savefig(f'{output}PV_plot.png')
    plt.show()