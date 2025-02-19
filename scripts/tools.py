from pyBBarolo.wrapper import PVSlice
from astropy.table import Table
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
#from spectral_cube import SpectralCube as SC
from astropy import wcs
from astropy import units as u
import numpy.ma as ma
import imageio
from ipywidgets import interact, FloatSlider, IntSlider
from GAstro.plot import ploth2
import os
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

    im_cube_maj = fits.getdata(f'{output}{name1}pv_{int(np.mean(c["PA"]))}.fits')
    im_cube_min = fits.getdata(f'{output}{name1}pv_{int(np.mean(c["PA"])+90)}.fits')
    im_model_maj = fits.getdata(f'{output}{name2}pv_{int(np.mean(c["PA"]))}.fits')
    im_model_min = fits.getdata(f'{output}{name2}pv_{int(np.mean(c["PA"])+90)}.fits')

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

##########################################################################################################################################################

#THIS IS A CODE TO PLOT THE CHANNELS ALL TOGETHER

def add_channels(fname, k_min=0, k_max=10):
    head = fits.getheader(fname)
    image = fits.getdata(fname) #numpy array, my data
    print("original shape of cube:" , image.shape)
    # row number is y coordinate so it is (rows, columns), (y,x) this is for a 2d one, but if i take a 3d? (z,y,x)

    all_channels = np.zeros((image.shape[1], image.shape[2]))
    choose_channels = np.zeros((image.shape[1], image.shape[2]))

    for k in range(image.shape[0]):
        all_channels = image[k] + all_channels
        if k_min < k < k_max:
            choose_channels = image[k] + choose_channels

    mean1, meadian1, std1 = sigma_clipped_stats(all_channels, sigma=2.0)
    mean2, meadian2, std2 = sigma_clipped_stats(choose_channels, sigma=2.0)

    plt.rcParams['figure.figsize'] = [10, 5]

    fig, axs = plt.subplots(1, 2)

    cont_level_all = [std1 * x for x in [-1, 1, 2, 4, 8, 16, 32, 64]]
    cax0 = axs[0].contour(all_channels, levels=cont_level_all, cmap='viridis')
    axs[0].imshow(all_channels, cmap="Greys", origin="lower")
    axs[0].set_title('All Channels')

    cont_level_choose = [std2 * x for x in [-1, 1, 2, 4, 8, 16, 32, 64]]
    cax1 = axs[1].contour(choose_channels, levels=cont_level_choose, cmap='viridis')
    axs[1].imshow(choose_channels, cmap="Greys", origin="lower")
    axs[1].set_title(f'Chosen Channels ({k_min} < k < {k_max})')

    plt.show()

##########################################################################################################################################################

#THIS IS A CODE TO CREATE A GIF WITH THE CHANNELS AND RESIDUALS

def chview_gif(data, data2=None, show_residuals=True, output_gif="channel_animation.gif", fps=5, figtitle=None):
    
    frames = []
    frames_filenames = []
    rms = np.nanstd(data[0, :, :])
    Hres = (data - data2) / rms
    for channel in range(data.shape[0]):  # Iterate over all channels
        if data2 is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            axl = [ax, None]
        elif show_residuals:
            fig, axl = plt.subplots(1, 3, figsize=(12, 4))
        else:
            fig, axl = plt.subplots(1, 2, figsize=(8, 4))

        # First data
        ax = axl[0]
        rms = np.nanstd(data[0, :, :])
        if rms == 0: rms = 0.001
        dmax = np.nanmax(data)
        levels = [-3 * rms,] + list(np.linspace(2 * rms, dmax, 6))
        vmin = 0.1 * np.min(data)
        vmax = 0.99 * np.max(data)

        _ = ploth2(ax=ax, H=data[channel, :, :], edges=(np.arange(0, data.shape[1]), np.arange(0, data.shape[2])),
                   cmap="plasma", gamma=0.5, vmin=vmin, vmax=vmax, vminmax_option="absolute",
                   levels=levels, levels_color="lime")

        ax.set_title(f"Data: Channel {channel}")
        
        if data2 is not None:
            ax = axl[1]
            _ = ploth2(ax=ax, H=data2[channel, :, :], edges=(np.arange(0, data2.shape[1]), np.arange(0, data2.shape[2])),
                       cmap="plasma", gamma=0.5, vmin=vmin, vmax=vmax, vminmax_option="absolute",
                       levels=levels, levels_color="lime")
            ax.set_title(f"Model:  Channel {channel}")
            
            if show_residuals:
                ax = axl[2]
                _ = ploth2(ax=ax, H=Hres[channel, :, :], edges=(np.arange(0, data2.shape[1]), np.arange(0, data2.shape[2])),
                           cmap="seismic", gamma=1, vmin=-10, vmax=10, vminmax_option="absolute", colorbar=True)
                ax.set_title(f"Residuals: Channel {channel}, Sum={np.nansum(Hres[channel,:,:]):4g}")

            if figtitle is not None:
                fig.suptitle(figtitle)

        fig.tight_layout()

        # Save frame
        frame_filename = f"frame_{channel}.png"
        plt.savefig(frame_filename)
        plt.close(fig)

        # Append frame to list
        frames_filenames.append(frame_filename)
        frames.append(imageio.imread(frame_filename))
 
    # Convert image list to animated GIF
    # Save frames as a GIF
    imageio.mimsave(output_gif, frames, fps=fps)
    print(f"GIF saved as {output_gif}")

    # Remove all PNG frames after creating the GIF
    for filename in frames_filenames:
        os.remove(filename)

##########################################################################################################################################################

#THIS IS A CODE TO PLOT THE CHANNELS INTERACTIVELY, WITH THE CURSOR THING

def chview(data,data2=None,show_residuals=True):

        
    def plot_channel(channel):

        if data2 is None: 
            fig,ax=plt.subplots(1,1,figsize=(4, 4))
            axl=[ax,None]
        elif show_residuals: 
            fig,axl=plt.subplots(1,3,figsize=(12, 4))
        else:
            fig,axl=plt.subplots(1,2,figsize=(8, 4))
        
        #First data
        ax=axl[0]
        rms=np.nanstd(data[0,:,:])
        print(rms)
        if rms==0: rms=0.001
        dmax=np.nanmax(data)
        levels=[-3*rms,]+list(np.linspace(2*rms,dmax,6))
        vmin=0.1*np.min(data)
        vmax=0.99*np.max(data)

        print(channel)
        
        _=ploth2(ax=ax,H=data[channel,:,:],edges=(np.arange(0,data.shape[1]),np.arange(0,data.shape[2])),
            cmap="plasma",gamma=0.5,vmin=vmin,vmax=vmax, vminmax_option="absolute",
            levels=levels,levels_color="lime")
        plt.sca(ax)

        if data2 is not None:
            ax=axl[1]
            _=ploth2(ax=ax,H=data2[channel,:,:],edges=(np.arange(0,data2.shape[1]),np.arange(0,data2.shape[2])),
            cmap="plasma",gamma=0.5,vmin=vmin,vmax=vmax, vminmax_option="absolute",
            levels=levels,levels_color="lime")
            plt.sca(ax)

            if show_residuals:
                Hres=(data-data2)/rms
                ax=axl[2]
                _=ploth2(ax=ax,H=Hres[channel,:,:],edges=(np.arange(0,data2.shape[1]),np.arange(0,data2.shape[2])),
                cmap="seismic",gamma=1,vmin=-10,vmax=10, vminmax_option="absolute",colorbar=True)
                plt.sca(ax)                
                
        fig.tight_layout()
        
    interact(plot_channel, channel=IntSlider(value=0, min=0, max=data.shape[0]-1, step=1))
    plt.show()
