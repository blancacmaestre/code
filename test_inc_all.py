#i am creating another file so that i can run a bunch of tests at once
import sys
sys.path.insert(0, "/Users/blanca/Documents/TESIS/software/Bbarolo-1.7")
from pyBBarolo import version
print (version)
import pyBBarolo as BB
import pyBBarolo.utils as ut
from pyBBarolo.bayesian import BayesianBBarolo
from dynesty import plotting as dyplot
from astropy.table import Table 
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import os, subprocess
from astropy.io import fits
import corner
from pyBBarolo.wrapper import PVSlice

BBmain = "/Users/blanca/Documents/TESIS/software/Bbarolo-1.7/BBarolo"

fitsname = "/Users/blanca/Documents/TESIS/software/code/models/model1/model1.fits"
ringfile = "/Users/blanca/Documents/TESIS/software/code/models/model1/model1_params.txt"
outputfile = "/Users/blanca/Documents/TESIS/software/code/TESTS/model1_enrico_NOR6_noz0/"

fitsnames = ["models/velcon_inc30/velcon_inc30.fits","models/velcon_inc30/velcon_inc30.fits",\
             "models/velcon_inc35/velcon_inc35.fits","models/velcon_inc35/velcon_inc35.fits",\
             "models/velcon_inc40/velcon_inc40.fits","models/velcon_inc40/velcon_inc40.fits",\
             "models/velcon_inc45/velcon_inc45.fits","models/velcon_inc45/velcon_inc45.fits",\
             "models/velcon_inc50/velcon_inc50.fits","models/velcon_inc50/velcon_inc50.fits",\
             "models/velcon_inc55/velcon_inc55.fits","models/velcon_inc55/velcon_inc55.fits",\
             "models/velcon_inc60/velcon_inc60.fits","models/velcon_inc60/velcon_inc60.fits",\
             "models/velcon_inc65/velcon_inc65.fits","models/velcon_inc65/velcon_inc65.fits",\
             "models/velcon_inc70/velcon_inc70.fits","models/velcon_inc70/velcon_inc70.fits",\
             "models/velcon_inc75/velcon_inc75.fits","models/velcon_inc75/velcon_inc75.fits"]

ringfiles = ["models/velcon_inc30/velcon_inc30_params.txt","models/velcon_inc30/velcon_inc30_params.txt",\
             "models/velcon_inc35/velcon_inc35_params.txt","models/velcon_inc35/velcon_inc35_params.txt",\
             "models/velcon_inc40/velcon_inc40_params.txt","models/velcon_inc40/velcon_inc40_params.txt",\
             "models/velcon_inc45/velcon_inc45_params.txt","models/velcon_inc45/velcon_inc45_params.txt",\
             "models/velcon_inc50/velcon_inc50_params.txt","models/velcon_inc50/velcon_inc50_params.txt",\
             "models/velcon_inc55/velcon_inc55_params.txt","models/velcon_inc55/velcon_inc55_params.txt",\
             "models/velcon_inc60/velcon_inc60_params.txt","models/velcon_inc60/velcon_inc60_params.txt",\
             "models/velcon_inc65/velcon_inc65_params.txt","models/velcon_inc65/velcon_inc65_params.txt",\
             "models/velcon_inc70/velcon_inc70_params.txt","models/velcon_inc70/velcon_inc70_params.txt",\
             "models/velcon_inc75/velcon_inc75_params.txt","models/velcon_inc75/velcon_inc75_params.txt"]

outputfiles = ["velcon_inc30","velcon_inc30_biginc",\
               "velcon_inc35","velcon_inc35_biginc",\
               "velcon_inc40","velcon_inc40_biginc",\
               "velcon_inc45","velcon_inc45_biginc",\
               "velcon_inc50","velcon_inc50_biginc",\
               "velcon_inc55","velcon_inc55_biginc",\
               "velcon_inc60","velcon_inc60_biginc",\
               "velcon_inc65","velcon_inc65_biginc",\
               "velcon_inc70","velcon_inc70_biginc",\
               "velcon_inc75","velcon_inc75_biginc"]

inci = [30,30,35,35,40,40,45,45,50,50,55,55,60,60,65,65,70,70]
inci_bound = [[10,50],[20,80],[15,55],[20,80],[20,60],[20,80],[25,65],[20,80],[30,70],[20,80],\
              [35,75],[20,80],[40,80],[20,80],[45,85],[20,80],[45,85],[20,80],[45,85],[20,80]]

##########################################################################################################################################################

for i in range(len(fitsname)):

##########################################################################################################################################################
    fitsname = fitsnames[i]
    ringfile = ringfiles[i]
    outputfile = outputfiles[i]


    method = "nautilus"
    f3d = BayesianBBarolo(fitsname)
    th  = multiprocessing.cpu_count() 

    rad = np.arange(30,240,60)   
    xpos=25.5  
    ypos=25.5  
    vsys=20  
    vrad=0
    z0=0
    inc= inci[i] 
    phi=100   

    f3d.init(radii=rad,xpos=xpos,ypos=ypos,vsys=vsys,\
            vrot=120,vdisp=8,vrad=vrad,z0=z0,inc=inc,phi=phi)


    f3d.set_options(mask="SEARCH",linear=0)  
    f3d.show_options()

    f3d.bounds['vrot']  = [0,400]
    f3d.bounds['vdisp'] = [0,100]
    f3d.bounds['inc']   = inci_bound[i]
    f3d.bounds['phi']   = [50,150]
    #f3d.bounds['z0']    = [0,60]
    f3d.bounds['xpos']  = [20,30]
    f3d.bounds['ypos']  = [20,30]
    f3d.bounds['vsys']  = [-10,50] 

    f3d.compute(threads=th, method=method,freepar=['vrot','vdisp','inc_single','phi_single','xpos_single','ypos_single',"vsys_single"])
    # ','z0_single'])

    ##########################################################################################################################################################

    print(f3d.freepar_idx)
    print(f3d.ndim)

    hdul = fits.open(fitsname)
    modname   = 'barbamodel' 
    fi = hdul[0].header
    #these are the sizes
    xaxis = fi['NAXIS1']
    yaxis = fi['NAXIS2']
    zaxis = fi['NAXIS3']
    #xaxis = fi['NAXIS1']
    #here the cdelts
    pix1 = fi['CDELT1']
    pix2 = fi['CDELT2']
    chan = fi['CDELT3']
    # the beam
    bea = fi['BMAJ']
    #central values
    cr1 = fi["CRVAL1"]
    cr2 = fi["CRVAL2"]
    cr3 = fi["CRVAL3"]
    #central pixels
    crp1 = fi["CRPIX1"]
    crp2 = fi["CRPIX2"]
    crp3 = fi["CRPIX3"]

    #creating the shape of the model
    sim = ut.SimulatedGalaxyCube(axisDim=[xaxis, yaxis, zaxis],\
                            cdelts=[pix1, pix2, chan],\
                            crvals=[cr1, cr2, cr3],\
                            crpixs=[crp1, crp2, crp3],\
                            beam=bea, bunit='JY/BEAM', obj=modname)

    #we set the parameters
    vel  = f3d.params[f3d.freepar_idx["vrot"]]

    disp = f3d.params[f3d.freepar_idx["vdisp"]]

    if "inc" in f3d.freepar_idx:
        inc  = np.full(len(rad),f3d.params[f3d.freepar_idx["inc"]])
    else:
        inc = np.full(len(rad), inc)   

    if "phi" in f3d.freepar_idx:
        phi  =  np.full(len(rad),f3d.params[f3d.freepar_idx["phi"]])
    else:
        phi = np.full(len(rad), phi)   

    z0d   = np.full(len(rad),0.000)

    if "xpos" in f3d.freepar_idx:
        xpos  = np.full(len(rad),f3d.params[f3d.freepar_idx["xpos"]])
    else:
        xpos = np.full(len(rad), xpos)

    if "ypos" in f3d.freepar_idx:
        ypos  = np.full(len(rad),f3d.params[f3d.freepar_idx["ypos"]])
    else:
        ypos = np.full(len(rad), ypos)    
        

    vsy = np.full(len(rad),f3d.params[f3d.freepar_idx["vsys"]])

    dens   = np.full(len(rad),1)

    sim.define_galaxy(xpos=xpos,ypos=ypos,radii=rad,vsys=vsy,vdisp=disp,inc=inc,pa=phi,\
                    dens=dens,z0=z0d,vrot=vel,warpinc=False,warppa=False)


    print (f"Simulating {modname}...",flush=True,end='')

    ##########################################################################################################################################################

    sim.run(exe=BBmain, stdout='null', outfolder=outputfile+modname, smooth=True)

    if not os.path.isdir(f'{outputfile}{modname}'):os.mkdir(f'{outputfile}{modname}')
    mf = f'{modname}mod_nonorm.fits'

    subprocess.call([BBmain, '--modhead', f'{outputfile}{modname}/{mf}', 'OBJECT', modname], stdout=subprocess.DEVNULL)
    subprocess.call(['mv', f'{outputfile}{modname}/{mf}', f'{outputfile}{modname}/{modname}.fits'], stdout=subprocess.DEVNULL)

    for folder in ['pvs', 'maps']:
        if os.path.isdir(os.path.join(outputfile, modname, folder)):
            subprocess.call(['rm', '-rf', os.path.join(outputfile, modname, folder)], stdout=subprocess.DEVNULL)

    os.remove(f'{outputfile}{modname}/mask.fits')
    os.remove(f"{outputfile}{modname}/rings_model.txt")

    for item in [modname, 'emptycube.fits', 'galaxy_params.txt' ]:
        if os.path.isfile(item):
            os.remove(item)
        elif os.path.isdir(item):
            subprocess.call(['rm', '-rf', item], stdout=subprocess.DEVNULL)

    subprocess.call([f'mv',f'/Users/blanca/Documents/TESIS/software/code/mask.fits',f'{outputfile}'],stdout=subprocess.DEVNULL)

    print (f"Cleaning outpu done! ")

    ##########################################################################################################################################################

    if method == "dynesty":
        results = f3d.results
        rfig, raxes = dyplot.runplot(results)
        rfig.savefig(outputfile+'output_run.pdf',bbox_inches='tight')

        tfig, taxes = dyplot.traceplot(results)
        tfig.savefig(outputfile+'posteriors.pdf',bbox_inches='tight')

        quantiles = [0.16,0.50,0.84] # are these the contours?
        cfig, caxes = dyplot.cornerplot(results,show_titles=True,title_quantiles=quantiles,quantiles=quantiles, color='blue',max_n_ticks=5, labels=f3d.freepar_names, label_kwargs=dict(fontsize=20))
        cfig.savefig(outputfile+'corner.pdf',bbox_inches='tight')

    if method == "nautilus":
        print(f3d.samples)
        quantiles = [0.16,0.50,0.84]
        cfig = corner.corner(f3d.samples, weights=f3d.weights, title_quantiles=quantiles,quantiles=quantiles,show_titles=True,
        title_kwargs={"fontsize": 12}, labels=f3d.freepar_names, color='purple',plot_datapoints=True, range=np.repeat(0.999,f3d.ndim))
        cfig.savefig(outputfile+'corner.pdf',bbox_inches='tight')

    print (f"Corner plot is done!")

    ##########################################################################################################################################################

    t = Table.read(ringfile,format='ascii') #imput params
    v = Table.read(f"{outputfile}{modname}/{modname}_params.txt",format='ascii')


    major_MOD = PVSlice( fitsname=f"{outputfile}{modname}/{modname}.fits", XPOS_PV = np.mean(v['XPOS']), YPOS_PV = np.mean(v['YPOS']),  PA_PV = np.mean(v['PA']), OUTFOLDER = outputfile+"slices")              
    major_MOD.run(BBmain)
    minor_MOD = PVSlice( fitsname=f"{outputfile}{modname}/{modname}.fits", XPOS_PV = np.mean(v['XPOS']), YPOS_PV = np.mean(v['YPOS']),  PA_PV = np.mean(v['PA']+90), OUTFOLDER= outputfile+"slices")              
    minor_MOD.run(BBmain)

    if 'RAD(arcs)' in t.colnames:
        major_MASK = PVSlice( fitsname=f"{outputfile}mask.fits", XPOS_PV=np.mean(t['YPOS(pix)']), YPOS_PV=np.mean(t['YPOS(pix)']),  PA_PV=np.mean(t['P.A.(deg)']), OUTFOLDER= outputfile+"slices")              
        major_MASK.run(BBmain)
        minor_MASK = PVSlice( fitsname=f"{outputfile}mask.fits", XPOS_PV=np.mean(t['XPOS(pix)']), YPOS_PV=np.mean(t['YPOS(pix)']),  PA_PV=np.mean(t['P.A.(deg)']+90), OUTFOLDER= outputfile+"slices")              
        minor_MASK.run(BBmain)
    if 'RADII' in t.colnames:        
        major_MASK = PVSlice( fitsname=f"{outputfile}mask.fits", XPOS_PV=np.mean(t['XPOS']), YPOS_PV=np.mean(t['YPOS']),  PA_PV=np.mean(t['PA']), OUTFOLDER= outputfile+"slices")              
        major_MASK.run(BBmain)
        minor_MASK = PVSlice( fitsname=f"{outputfile}mask.fits", XPOS_PV=np.mean(t['XPOS']), YPOS_PV=np.mean(t['YPOS']),  PA_PV=np.mean(t['PA']+90), OUTFOLDER= outputfile+"slices")              
        minor_MASK.run(BBmain)

    print (f"slices are done! ")

    ##########################################################################################################################################################

    if method == "dynesty":
        samples = f3d.results.samples #these are all the samples that dynesty got, the last array are the parameters
        weights = np.exp(f3d.results.logwt - f3d.results.logz[-1])
        params = np.average(samples, axis=0, weights=weights) #mean of values 
    if method == "nautilus":
        samples = f3d.samples
        weights = f3d.weights
        params = f3d.params

    rad_mc = f3d._inri.r['radii']  # Get the radii array
    labs = f3d.freepar_names

    ra, pp, err_up, err_low = np.zeros(shape=(4, len(params)))
    # Adjust how we index rad_mc for vrot and vdisp parameters, using modulo if needed
    for i in range(len(params)):
        mcmc = np.percentile(samples[:, i], [15.865, 50, 84.135])
        q = np.diff(mcmc)
        txt = "%10s = %10.3f %+10.3f %+10.3f" % (labs[i], mcmc[1], -q[0], q[1])
        
        pp[i] = mcmc[1]
        err_low[i] = q[0]
        err_up[i] = q[1]
        
        # For vrot and vdisp parameters, assign the radius values
        if labs[i].startswith("vrot") or labs[i].startswith("vdisp"):
            # Use modulo indexing to ensure we don't go out of bounds (repeat the radii if necessary)
            idx = i % len(rad_mc)  # Ensure the index is within bounds
            ra[i] = rad_mc[idx]
        else:
            ra[i] = None  # No radius for other parameters

    output_file = "errors.txt"

    with open(outputfile+output_file, "w") as file:
        file.write(f"#{'Parameter':<15}{'Median':<15}{'Error_Low':<15}{'Error_Up':<15}{'Radius':<15}\n")
        
        for i in range(len(params)):

            # Check if the parameter is vrot or vdisp and write the radius
            if labs[i].startswith("vrot") or labs[i].startswith("vdisp"):

                # If the radius is None, print 'N/A' for the radius
                radius_str = f"{ra[i]:<15.6f}" if ra[i] is not None else "N/A"
                file.write(f"{labs[i]:<15}{pp[i]:<15.6f}{err_low[i]:<15.6f}{err_up[i]:<15.6f}{radius_str}\n")
            else:
                # Write the parameters without radius (vsys, inc, pa, xpos, ypos, etc.)
                file.write(f"{labs[i]:<15}{pp[i]:<15.6f}{err_low[i]:<15.6f}{err_up[i]:<15.6f}{"N/A":<15}\n")

    with open(outputfile+output_file, "r") as file:
        content = file.read()
    print(content)

    print(f"Results of errors saved to {output_file}")

    ##########################################################################################################################################################

    t = Table.read(ringfile,format='ascii')

    if 'RAD(arcs)' in t.colnames:


        fig,ax = plt.subplots(nrows=4,ncols=2,figsize=(11,16))
        ax = np.ravel(ax)
        for a in ax:
            a.set_xlabel("radius (arcsec)")

        rad_mc = f3d._inri.r['radii']

        idx = f3d.freepar_idx['vrot']
        ax[0].plot(t['RAD(arcs)'],t['VROT(km/s)'],label='input')
        ax[0].errorbar(rad_mc,pp[idx],fmt='o',yerr=[err_low[idx],err_up[idx]],label='recovered')
        ax[0].set_ylabel("vrot (km/s)")
        ax[0].legend(loc='lower right')

        idx = f3d.freepar_idx['vdisp']
        ax[1].plot(t['RAD(arcs)'],t['DISP(km/s)'])
        ax[1].errorbar(rad_mc,pp[idx],fmt='o',yerr=[err_low[idx],err_up[idx]])
        ax[1].set_ylabel("vdisp (km/s)")

        if "inc" in f3d.freepar_idx:
            idx = f3d.freepar_idx['inc']
            ax[2].plot(t['RAD(arcs)'],t['INC(deg)'])
            ax[2].axhspan(pp[idx][0]-err_low[idx][0],pp[idx][0]+err_up[idx][0],alpha=0.2,color='orange')
            ax[2].axhline(pp[idx][0],ls='--',color='orange')
            ax[2].set_ylabel("inc (deg)")
            ax[2].set_ylim(t['INC(deg)'][0]-20,t['INC(deg)'][0]+20)

        if "phi" in f3d.freepar_idx:
            idx = f3d.freepar_idx['phi']
            ax[3].plot(t['RAD(arcs)'],t['P.A.(deg)'])
            ax[3].axhspan(pp[idx][0]-err_low[idx][0],pp[idx][0]+err_up[idx][0],alpha=0.2,color='orange')
            ax[3].axhline(pp[idx][0],ls='--',color='orange')
            ax[3].set_ylabel("phi (deg)")
            ax[3].set_ylim(t['P.A.(deg)'][0]-10,t['P.A.(deg)'][0]+10)

        if "z0" in f3d.freepar_idx:
            idx = f3d.freepar_idx['z0']
            ax[4].plot(t['RAD(arcs)'],t['Z0(arcs)'])
            ax[4].axhspan(pp[idx][0]-err_low[idx][0],pp[idx][0]+err_up[idx][0],alpha=0.2,color='orange')
            ax[4].axhline(pp[idx][0],ls='--',color='orange')
            ax[4].set_ylabel("z0 (arcs)")
            ax[4].set_ylim(t['Z0(arcs)'][0]-30,t['Z0(arcs)'][0]+30)

        if "vsys" in f3d.freepar_idx:
            idx = f3d.freepar_idx['vsys']
            ax[5].plot(t['RAD(arcs)'],t['VSYS(km/s)'])
            ax[5].axhspan(pp[idx][0]-err_low[idx][0],pp[idx][0]+err_up[idx][0],alpha=0.2,color='orange')
            ax[5].axhline(pp[idx][0],ls='--',color='orange')
            ax[5].set_ylabel("vsys (km/s)")
            ax[5].set_ylim(t['VSYS(km/s)'][0]-30,t['VSYS(km/s)'][0]+30)

        if "xpos" in f3d.freepar_idx:
            idx = f3d.freepar_idx['xpos']
            ax[6].plot(t['RAD(arcs)'],t['XPOS(pix)'])
            ax[6].axhspan(pp[idx][0]-err_low[idx][0],pp[idx][0]+err_up[idx][0],alpha=0.2,color='orange')
            ax[6].axhline(pp[idx][0],ls='--',color='orange')
            ax[6].set_ylabel("xpos (pix)")
            ax[6].set_ylim(t['XPOS(pix)'][0]-5,t['XPOS(pix)'][0]+5)

        if "ypos" in f3d.freepar_idx:
            idx = f3d.freepar_idx['ypos']
            ax[7].plot(t['RAD(arcs)'],t['YPOS(pix)'])
            ax[7].axhspan(pp[idx][0]-err_low[idx][0],pp[idx][0]+err_up[idx][0],alpha=0.2,color='orange')
            ax[7].axhline(pp[idx][0],ls='--',color='orange')
            ax[7].set_ylabel("ypos (pix)")
            ax[7].set_ylim(t['YPOS(pix)'][0]-5,t['YPOS(pix)'][0]+5)

        fig.savefig(outputfile+'parameters.pdf',bbox_inches='tight')    
    else:
        print("RAD(arcsec) is not in the table.")  


    if 'RADII' in t.colnames:

        fig,ax = plt.subplots(nrows=4,ncols=2,figsize=(11,16))
        ax = np.ravel(ax)
        for a in ax:
            a.set_xlabel("radius (arcsec)")

        rad_mc = f3d._inri.r['radii']

        idx = f3d.freepar_idx['vrot']

        ax[0].plot(t['RADII'],t['VROT'],label='input')
        ax[0].errorbar(rad_mc,pp[idx],fmt='o',yerr=[err_low[idx],err_up[idx]],label='recovered')
        ax[0].set_ylabel("vrot (km/s)")
        ax[0].legend(loc='lower right')

        idx = f3d.freepar_idx['vdisp']
        ax[1].plot(t['RADII'],t['VDISP'])
        ax[1].errorbar(rad_mc,pp[idx],fmt='o',yerr=[err_low[idx],err_up[idx]])
        ax[1].set_ylabel("vdisp (km/s)")

        if "inc" in f3d.freepar_idx:
            idx = f3d.freepar_idx['inc']
            ax[2].plot(t['RADII'],t['INC'])
            ax[2].axhspan(pp[idx][0]-err_low[idx][0],pp[idx][0]+err_up[idx][0],alpha=0.2,color='orange')
            ax[2].axhline(pp[idx][0],ls='--',color='orange')
            ax[2].set_ylabel("inc (deg)")
            ax[2].set_ylim(t['INC'][0]-20,t['INC'][0]+20)

        if "phi" in f3d.freepar_idx:
            idx = f3d.freepar_idx['phi']
            ax[3].plot(t['RADII'],t['PA'])
            ax[3].axhspan(pp[idx][0]-err_low[idx][0],pp[idx][0]+err_up[idx][0],alpha=0.2,color='orange')
            ax[3].axhline(pp[idx][0],ls='--',color='orange')
            ax[3].set_ylabel("phi (deg)")
            ax[3].set_ylim(t['PA'][0]-10,t['PA'][0]+10)

        if "z0" in f3d.freepar_idx:
            idx = f3d.freepar_idx['z0']
            ax[4].plot(t['RADII'],t['Z0'])
            ax[4].axhspan(pp[idx][0]-err_low[idx][0],pp[idx][0]+err_up[idx][0],alpha=0.2,color='orange')
            ax[4].axhline(pp[idx][0],ls='--',color='orange')
            ax[4].set_ylabel("z0 (arcs)")
            ax[4].set_ylim(t['Z0'][0]-30,t['Z0'][0]+30)

        if "vsys" in f3d.freepar_idx:
            idx = f3d.freepar_idx['vsys']
            ax[5].plot(t['RADII'],t['VSYS'])
            ax[5].axhspan(pp[idx][0]-err_low[idx][0],pp[idx][0]+err_up[idx][0],alpha=0.2,color='orange')
            ax[5].axhline(pp[idx][0],ls='--',color='orange')
            ax[5].set_ylabel("vsys (km/s)")
            ax[5].set_ylim(t['VSYS'][0]-30,t['VSYS'][0]+30)

        if "xpos" in f3d.freepar_idx:
            idx = f3d.freepar_idx['xpos']
            ax[6].plot(t['RADII'],t['XPOS'])
            ax[6].axhspan(pp[idx][0]-err_low[idx][0],pp[idx][0]+err_up[idx][0],alpha=0.2,color='orange')
            ax[6].axhline(pp[idx][0],ls='--',color='orange')
            ax[6].set_ylabel("xpos (pix)")
            ax[6].set_ylim(t['XPOS'][0]-5,t['XPOS'][0]+5)

        if "ypos" in f3d.freepar_idx:
            idx = f3d.freepar_idx['ypos']
            ax[7].plot(t['RADII'],t['YPOS'])
            ax[7].axhspan(pp[idx][0]-err_low[idx][0],pp[idx][0]+err_up[idx][0],alpha=0.2,color='orange')
            ax[7].axhline(pp[idx][0],ls='--',color='orange')
            ax[7].set_ylabel("ypos (pix)")
            ax[7].set_ylim(t['YPOS'][0]-5,t['YPOS'][0]+5)
        fig.savefig(outputfile+'parameters.pdf',bbox_inches='tight')
    else:
        print("RADII is not in the table.")

    fig.savefig(outputfile+'parameters.pdf',bbox_inches='tight')

    print("parameter plots are done!.")
