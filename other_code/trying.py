import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import PowerStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import PercentileInterval

# Set plotting parameters
mpl.rc('xtick', direction='in')
mpl.rc('ytick', direction='in')
mpl.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('font', family='sans-serif', serif='Helvetica', size=10)
params = {'text.usetex': False, 'mathtext.fontset': 'cm', 'mathtext.default': 'regular'}
plt.rcParams.update(params)

# Define paths and filenames
galaxy_dir = "/Users/blanca/Documents/TESIS/software/Bbarolo-1.7/output/model2/"
model_dir = '/Users/blanca/Documents/TESIS/software/codes/last_model2/model_result_barba/'
fitsname = "/Users/blanca/Documents/TESIS/codes/models/model2.fits"
outfolder = '/Users/blanca/Documents/TESIS/software/codes/pv_plot'
gname = 'model2'

# Read model parameters
twostage = 1
model_params = np.genfromtxt(model_dir + 'model_result_barba_params.txt', usecols=(0, 1, 4, 5, 10), unpack=True)
rad, vrot, inc, pa, vsys = model_params

# Find model files
files_pva_mod, files_pvb_mod = [], []
for thisFile in sorted(os.listdir(model_dir + 'pvs/')):
    if 'pv_a' in thisFile: files_pva_mod.append(thisFile)
    if 'pv_b' in thisFile: files_pvb_mod.append(thisFile)

# Open the galaxy and model images
image_maj = fits.open(galaxy_dir + 'pvs/' + gname + '_pv_a.fits')
image_min = fits.open(galaxy_dir + 'pvs/' + gname + '_pv_b.fits')
image_mas_maj = fits.open(model_dir + 'pvs/' + "model_result_barbamask_pv_a.fits")
image_mas_min = fits.open(model_dir + 'pvs/' + "model_result_barbamask_pv_b.fits")

# Extract header information
head = [image_maj[0].header, image_min[0].header]
crpixpv = np.array([head[0]['CRPIX1'], head[1]['CRPIX1']])
cdeltpv = np.array([head[0]['CDELT1'], head[1]['CDELT1']])
crvalpv = np.array([head[0]['CRVAL1'], head[1]['CRVAL1']])

# Define x-axis bounds for PV diagrams
xminpv, xmaxpv = np.floor(crpixpv - 1 - 21), np.ceil(crpixpv - 1 + 21)
xminpv[xminpv < 0] = 0
xmaxpv[xmaxpv >= head[0]['NAXIS1']] = head[0]['NAXIS1'] - 1

# Define z-axis bounds
zmin, zmax = 21, 106

# Extract data from FITS files
data_maj = image_maj[0].data[zmin:zmax + 1, int(xminpv[0]):int(xmaxpv[0]) + 1]
data_min = image_min[0].data[zmin:zmax + 1, int(xminpv[1]):int(xmaxpv[1]) + 1]
data_mas_maj = image_mas_maj[0].data[zmin:zmax + 1, int(xminpv[0]):int(xmaxpv[0]) + 1]
data_mas_min = image_mas_min[0].data[zmin:zmax + 1, int(xminpv[1]):int(xmaxpv[1]) + 1]

# Convert pixel coordinates to WCS coordinates
xmin_wcs = ((xminpv + 1 - 0.5 - crpixpv) * cdeltpv + crvalpv) * 3600
xmax_wcs = ((xmaxpv + 1 + 0.5 - crpixpv) * cdeltpv + crvalpv) * 3600
zmin_wcs, zmax_wcs = -215, 215

# Contour levels
cont = 0.00259927
v = np.array([1, 2, 4, 8, 16, 32, 64]) * cont
v_neg = [-cont]
interval = PercentileInterval(99.5)
vmax = interval.get_limits(data_maj)[1]
if cont > vmax:
    vmin, vmax = vmax, cont

norm = ImageNormalize(vmin=cont, vmax=vmax, stretch=PowerStretch(0.5))


# Prepare for the plot
radius = np.concatenate((rad, -rad))
pamaj_av = 0.117532
pamin_av = 90.1175
costh = np.cos(np.deg2rad(np.abs(pa - pamaj_av)))
vlos1 = vsys + vrot * np.sin(np.deg2rad(inc)) * costh
vlos2 = vsys - vrot * np.sin(np.deg2rad(inc)) * costh
reverse = False
if reverse: vlos1, vlos2 = vlos2, vlos1
vlos = np.concatenate((vlos1, vlos2))
vsys_m = np.nanmean(vsys)

ext = [[xmin_wcs[0], xmax_wcs[0], zmin_wcs - vsys_m, zmax_wcs - vsys_m],
       [xmin_wcs[1], xmax_wcs[1], zmin_wcs - vsys_m, zmax_wcs - vsys_m]]

palab = [r'$\phi = $%i$^\circ$' % np.round(pamaj_av), r'$\phi = $%i$^\circ$' % np.round(pamin_av)]

# Define a function for plotting the PV diagrams
def plot_pv(ax, data, extent, contour_levels, contour_color, label):
    ax.imshow(data, origin='lower', cmap=mpl.cm.Greys, norm=norm, extent=extent, aspect='auto')
    ax.contour(data, contour_levels, origin='lower', linewidths=0.7, colors=contour_color, extent=extent)
    ax.axhline(y=0, color='black')
    ax.axvline(x=0, color='black')
    ax.grid(color='gray', linestyle='--', linewidth=0.3)
    ax.set_xlabel('Offset (arcsec)', fontsize=14)
    ax.set_ylabel(r'$\mathrm{\Delta V_{LOS}}$ (km/s)', fontsize=14)

# Begin plotting
for k in range(len(files_pva_mod)):
    # Open model PV diagrams
    image_mod_maj = fits.open(model_dir + 'pvs/' + files_pva_mod[k])
    image_mod_min = fits.open(model_dir + 'pvs/' + files_pvb_mod[k])

    # Extract data
    data_mod_maj = image_mod_maj[0].data[zmin:zmax + 1, int(xminpv[0]):int(xmaxpv[0]) + 1]
    data_mod_min = image_mod_min[0].data[zmin:zmax + 1, int(xminpv[1]):int(xmaxpv[1]) + 1]
    
    # Prepare data to plot
    toplot = [[data_maj, data_min], [data_mod_maj, data_mod_min], [data_mas_maj, data_mas_min]]

    # Create figure
    fig = plt.figure(figsize=(10, 10), dpi=150)
    ax = [fig.add_axes([0.1, 0.7, 0.6, 0.42]), fig.add_axes([0.1, 0.58, 0.6, 0.42])]
    
    for i in range(2):
        plot_pv(ax[i], toplot[0][i], ext[i], v, '#00008B', r'$\phi = $%i$^\circ' % np.round(pamaj_av + i * 90))

    # Save figure
    outfile = '%s_pv' % gname
    if 'azim' in files_pva_mod[k]: outfile += '_azim'
    elif 'local' in files_pva_mod[k]: outfile += '_local'
    
    fig.savefig(outfolder + outfile + '.pdf', bbox_inches='tight')

    # Close files
    image_mod_maj.close()
    image_mod_min.close()

# Close the image files
image_maj.close()
image_min.close()