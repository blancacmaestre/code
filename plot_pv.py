#################################################################################
#### This script writes a plot of position-velocity slices of model and data ####
#################################################################################
#i am importing stuff here
import numpy as np
import os 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from astropy.io import fits 
from astropy.visualization import PowerStretch 
from astropy.visualization.mpl_normalize import ImageNormalize 
from astropy.visualization import PercentileInterval 
from pyBBarolo.wrapper import PVSlice
from astropy.stats import mad_std

from tools import getthevelocity, getthechannel
BBmain = "/Users/blanca/Documents/TESIS/software/Bbarolo-1.7/BBarolo"

#I NEED TO SEE WHAT THIS IS!!!!!

mpl.rc('xtick',direction='in') 
mpl.rc('ytick',direction='in') 
mpl.rcParams['contour.negative_linestyle'] = 'solid' 
plt.rc('font',family='sans-serif',serif='Helvetica',size=10) 
params = {'text.usetex': False, 'mathtext.fontset': 'cm', 'mathtext.default': 'regular'} 
plt.rcParams.update(params) 

#THIS PART IS ALL THE IMPUTS 

#SLICES AND FITS FILE OF FITTING GALAXY
major_GAL = "/Users/blanca/Documents/TESIS/software/code/models/model1/model1pv_30.fits"
minor_GAL = "/Users/blanca/Documents/TESIS/software/code/models/model1/model1pv_120.fits"
fits_GAL = "/Users/blanca/Documents/TESIS/software/code/models/model1/model1.fits"

#SLICES AND PARAMETERS OF MODEL
major_MOD = '/Users/blanca/Documents/TESIS/software/code/TESTS/model1_enrico/slices/barbamodelpv_30.fits' 
minor_MOD = "/Users/blanca/Documents/TESIS/software/code/TESTS/model1_enrico/slices/barbamodelpv_120.fits"
model_par = "/Users/blanca/Documents/TESIS/software/code/TESTS/model1_enrico/barbamodel/barbamodel_params.txt"
fits_MOD = "/Users/blanca/Documents/TESIS/software/code/TESTS/model1_enrico/barbamodel/barbamodel.fits"

#SLICES OF MASK (PARAMETERS MATCH THAT OF THE GALAXY)
major_MASK = '/Users/blanca/Documents/TESIS/software/code/TESTS/model1_enrico/slices/model1pv_30.fits' 
minor_MASK = "/Users/blanca/Documents/TESIS/software/code/TESTS/model1_enrico/slices/model1pv_120.fits"

#OUTPUT FILE AND NAME
gname = 'model1_enrico'
outfile = f'{gname}_pv'
outfolder = '/Users/blanca/Documents/TESIS/software/code/pv_plot/' 

plotmask = 0  

#HERE I GET THE PARAMETERS OF MY MODEL

rad,vrot,inc,pa,vsys = np.genfromtxt(model_par,usecols=(0,1,4,5,10),unpack=True) 
print(vrot)

zmin,zmax = 30,110
#zmin = getthechannel(fits_MOD, vrot[0])-2
#print("zmin is",zmin)
#zmax = getthechannel(fits_MOD, vrot[len(vrot)-1])+5
#print("zmax is",zmax)

GAL_maj  = fits.open(major_GAL) #this is the real stuff
GAL_min  = fits.open(minor_GAL) 
MASK_maj = fits.open(major_MASK) 
MASK_min = fits.open(minor_MASK) 

#this part is calculating the bounds of the image, I should calulate them so that i can us this only putting the files

head = [GAL_maj[0].header,GAL_min[0].header] 
crpixpv = np.array([head[0]['CRPIX1'],head[1]['CRPIX1']]) 
cdeltpv = np.array([head[0]['CDELT1'],head[1]['CDELT1']]) 
crvalpv = np.array([head[0]['CRVAL1'],head[1]['CRVAL1']]) 
xminpv, xmaxpv = np.floor(crpixpv-1-21), np.ceil(crpixpv-1 +21) 
if xminpv[0]<0: xminpv[0]=0 
if xminpv[1]<0: xminpv[1]=0 
if xmaxpv[0]>=head[0]['NAXIS1']: xmaxpv[0]=head[0]['NAXIS1']-1 
if xmaxpv[1]>=head[1]['NAXIS1']: xmaxpv[1]=head[1]['NAXIS1']-1 

data_maj = GAL_maj[0].data[zmin:zmax+1,int(xminpv[0]):int(xmaxpv[0])+1] 
data_min = GAL_min[0].data[zmin:zmax+1,int(xminpv[1]):int(xmaxpv[1])+1] 
data_mas_maj = MASK_maj[0].data[zmin:zmax+1,int(xminpv[0]):int(xmaxpv[0])+1] 
data_mas_min = MASK_min[0].data[zmin:zmax+1,int(xminpv[1]):int(xmaxpv[1])+1] 
xmin_wcs = ((xminpv+1-0.5-crpixpv)*cdeltpv+crvalpv)*3600
xmax_wcs = ((xmaxpv+1+0.5-crpixpv)*cdeltpv+crvalpv)*3600
zmin_wcs, zmax_wcs = getthevelocity(fits_MOD,zmin), getthevelocity(fits_MOD, zmax) #I NEED TO GET THIS S0MEHOW and this is the velocity?

cont = np.std(fits_GAL)

v = np.array([1,2,4,8,16,32,64])*cont 
v_neg = [-cont] 
interval = PercentileInterval(99.5) 
vmax = interval.get_limits(data_maj)[1] 
norm = ImageNormalize(vmin=cont, vmax=vmax, stretch=PowerStretch(0.5)) 


radius = np.concatenate((rad,-rad)) 
pa_av = np.mean(pa) 
pa_min = np.mean(pa)+90 
costh = np.cos(np.deg2rad(np.abs(pa-pa_av))) 
vlos1 = vsys+vrot*np.sin(np.deg2rad(inc))*costh 
vlos2 = vsys-vrot*np.sin(np.deg2rad(inc))*costh

if 225>pa_av>=45: 
	vlos1, vlos2 = vlos2, vlos1 
vlos = np.concatenate((vlos1,vlos2)) 
vsys_m = np.nanmean(vsys) 
ext = [[xmin_wcs[0],xmax_wcs[0],zmin_wcs-vsys_m,zmax_wcs-vsys_m],\
       [xmin_wcs[1],xmax_wcs[1],zmin_wcs-vsys_m,zmax_wcs-vsys_m]] 
labsize = 14 
palab = [r'$\phi = $%i$^\circ$'%np.round(pa_av), r'$\phi = $%i$^\circ$'%np.round(pa_min)] 

# Beginning PV plot 

image_mod_maj = fits.open(major_MOD)
image_mod_min = fits.open(minor_MOD) 
data_mod_maj = image_mod_maj[0].data[zmin:zmax+1,int(xminpv[0]):int(xmaxpv[0])+1] 
data_mod_min = image_mod_min[0].data[zmin:zmax+1,int(xminpv[1]):int(xmaxpv[1])+1] 
toplot = [[data_maj,data_min],[data_mod_maj,data_mod_min],[data_mas_maj,data_mas_min]]
    

fig = plt.figure(figsize=(10,10), dpi=150) 
x_len, y_len, y_sep = 0.6, 0.42, 0.08 
ax, bottom_corner = [], [0.1,0.7] 
for i in range (2): 
	bottom_corner[0], axcol = 0.1, [] 
	ax.append(fig.add_axes([bottom_corner[0],bottom_corner[1],x_len,y_len])) 
	bottom_corner[1]-=(y_len+y_sep) 

for i in range (2): 
	axis = ax[i] 
	axis.tick_params(which='major',length=8, labelsize=labsize) 
	axis.set_xlabel('Offset (arcsec)',fontsize=labsize+2) 
	axis.set_ylabel(r'$\mathrm{\Delta V_{LOS}}$ (km/s)',fontsize=labsize+2) 
	axis.text(1, 1.02,palab[i],ha='right',transform=axis.transAxes,fontsize=labsize+4) 
	axis2 = axis.twinx() 
	axis2.set_xlim([ext[i][0],ext[i][1]]) 
	axis2.set_ylim([ext[i][2]+vsys_m,ext[i][3]+vsys_m]) 
	axis2.tick_params(which='major',length=8, labelsize=labsize) 
	axis2.set_ylabel(r'$\mathrm{V_{LOS}}$ (km/s)',fontsize=labsize+2) 
	axis.imshow(toplot[0][i],origin='lower',cmap = mpl.cm.Greys,norm=norm,extent=ext[i],aspect='auto') 
	axis.contour(toplot[0][i],v,origin='lower',linewidths=0.7,colors='#00008B',extent=ext[i]) 
	axis.contour(toplot[0][i],v_neg,origin='lower',linewidths=0.1,colors='gray',extent=ext[i]) 
	axis.contour(toplot[1][i],v,origin='lower',linewidths=1,colors='#B22222',extent=ext[i]) 
	axis.axhline(y=0,color='black') 
	axis.axvline(x=0,color='black') 
	axis.grid(color='gray', linestyle='--', linewidth=0.3) 
	axis.contour(toplot[2][i],levels=[0],origin='lower',linewidths=2,colors='k',extent=ext[i]) 
	if i==0 : 
		axis2.plot(radius,vlos,'yo') 
		axis.text(0, 1.1, gname, transform=axis.transAxes,fontsize=22) 


image_mod_maj.close() 
image_mod_min.close() 
fig.savefig(f"{outfolder}{outfile}.pdf", bbox_inches='tight') 

