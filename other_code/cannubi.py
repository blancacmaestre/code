import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import subprocess
from astropy.io import fits
from scipy.stats import median_abs_deviation as MADFM
import sys
from tqdm import tqdm


print ()
print ("=====                                                                 =====")
print ("=====                            CANNUBI                              =====")
print ("=====  Centre and Angles via Numerical Nous Using Bayesian Inference  =====")
print ("=====                                                                 =====")
print ()


###                  ###
### INITIAL SETTINGS ### 
###                  ###


# BBarolo executable #
BBexec = "/Users/filippo/Software/Bbarolo-master/BBarolo"


# READING PARAMETER FROM CANNUBI input file #
if (len(sys.argv) == 1):
    cannubi_parFile = input("Parameter file name? (you can also write it as you call Cannubi)")
else:
    cannubi_parFile = str(sys.argv[1])
    
print ("Reading parameters from ", cannubi_parFile)


key = np.ndarray.tolist(np.genfromtxt(cannubi_parFile, dtype=str, comments='//', usecols=(0), case_sensitive=False, unpack=True))
val = np.ndarray.tolist(np.genfromtxt(cannubi_parFile, dtype=str, comments='//', usecols=(1), case_sensitive='lower', unpack=True))
key=[item.lower() for item in key]
val=[item.lower() for item in val]


### MAIN SETTINGS ###
reduceCube=False
if ("reducecube" in key):
    if (val[key.index("reducecube")] in ['true', 'y', 'yes']):
        reduceCube=True
        reduceFactor=2.0
        if (val[key.index("factor")]): reduceFactor=float(val[key.index("factor")])
useAllCube=False
if ("useallcube" in key):
    if (val[key.index("useallcube")] in ['true', 'y', 'yes']): useAllCube=True
else:
    print ('WARNING: useallcube not specified, assuming False')
normaliseMod=True
if (val[key.index("normalisemod")] in ['false', 'n', 'no']): normaliseMod=False
runmcmc=True
if (val[key.index("runmcmc")] in ['false', 'n', 'no']): runmcmc=False
useMaskTot=True
if (val[key.index("usemasktot")] in ['false', 'n', 'no']): useMaskTot=False
plotTotMask=False
if (val[key.index("plottotmask")] in ['true', 'y', 'yes']): plotTotMask=True
if ("usemaskmod" in key):
    useMaskMod=val[key.index("usemaskmod")]
    if (useMaskMod not in ['union', 'same', 'threshold']):
        print ('WARNING: Type of mask for model not specified, assuming union')
        useMaskMod='union'
    if (useMaskMod=='same' and not useMaskTot):
        print ("WARNING: useMaskMod = Same can be only used when useMaskTot is true - I use useMaskMod = Union")
        useMaskMod='union'
else:
    print ('WARNING: Type of mask for model not specified, assuming union')
    useMaskMod='union'
if ("residualtype" in key):
    residualType=val[key.index("residualtype")]
else:
    print ('WARNING: Type of residuals not specified, assuming abs')
    residualtype="abs"
if ("errortype" in key):
    errorType=val[key.index("errortype")]
else:
    print ('WARNING: Type of errors in residuals not specified, assuming marasco')
    errorType="marasco"
fitPA=False; fitZ0=False; fitSep=False
if (val[key.index("fitpa")] in ['true', 'y', 'yes']):
    fitPA=True
if (val[key.index("fitz0")] in ['true', 'y', 'yes']):
    fitZ0=True
if (val[key.index("fitsep")] in ['true', 'y', 'yes']):
    fitSep=True
finalModel=False


### READING KEYWORDS FROM DATACUBE's HEADER ###
filename=val[key.index("fitsfile")]
hdu_list = fits.open(filename)


if (reduceCube):
#    with fits.open(filename) as hdu_list:
    hdr = hdu_list[0].header
    data = hdu_list[0].data
    hdu_list.close()
    naxis = hdr['NAXIS']
    naxis1 = hdr['NAXIS1']
    naxis2 = hdr['NAXIS2']
    firstpix1=int(naxis1/2.)-int(naxis1/reduceFactor/2.)
    firstpix2=int(naxis2/2.)-int(naxis2/reduceFactor/2.)
    lastpix1=int(naxis1/2.)+int(naxis1/reduceFactor/2.)
    lastpix2=int(naxis2/2.)+int(naxis2/reduceFactor/2.)
    hdr['CRPIX1'] = hdr['CRPIX1']-firstpix1
    hdr['CRPIX2'] = hdr['CRPIX2']-firstpix2
    if (naxis == 3):
        hduReduced=fits.PrimaryHDU(data[:,firstpix1:lastpix1,firstpix2:lastpix2])
    if (naxis == 4):
        hduReduced=fits.PrimaryHDU(data[:,:,firstpix1:lastpix1,firstpix2:lastpix2])        
    filename=filename[:-5]+'_r.fits'
    hduReduced.header=hdr
    hduReduced.writeto(filename, overwrite=True)
    hdu_list = fits.open(filename)


print ('Working on file with ', end='')
print (hdu_list.info())


hdr = hdu_list[0].header
naxis = hdr['NAXIS']
if ('OBJECT' in hdr):
    galname=hdr['OBJECT']
    galname = galname.replace(' ', '_')
else: galname='NONE'
if ('BMAJ' in hdr): Bmaj=hdr['BMAJ']
else: Bmaj = input("Beam not in the Header; enter value: ")
if ('BMIN' in hdr): Bmin=hdr['BMIN']
if ('BUNIT' in hdr): bunit=hdr['BUNIT']
if ('CDELT2' in hdr): cdelt2=hdr['CDELT2']
if ('CDELT3' in hdr): cdelt3=hdr['CDELT3']
beam=np.sqrt(Bmaj*Bmin)


BBfitDir='./output/'+galname+'/'


### FIT PARAMETERS ###
cutMask=float(val[key.index("cutmask")])
fromTop=float(val[key.index("fromtop")])
thresholdAll=float(val[key.index("thresholdall")])
lowestContUser=float(val[key.index("lastconttot")])
noiseTimes=float(val[key.index("noisetimes")])


#print (np.nonzero(hdu_list[0].data))


if ("noise" in key):
    noise=float(val[key.index("noise")])
else:
    if (0 in hdu_list[0].data):
        noise=1.48*np.nanmedian(MADFM(np.nonzero(hdu_list[0].data)))
        print ("WARNING: Zeros present in data, ignored in calculation of r.m.s. noise")
    else:
        noise=1.48*np.nanmedian(MADFM(hdu_list[0].data))
    print ("WARNING: RMS noise per channel not specified, I estimate : {:.6f}".format(noise))


hdu_list.close()


### Barolo SETTINGS ###
if ("nradii" in key): nradii=float(val[key.index("nradii")])
else:
    print ("ERROR: NRADII not found in the input file")
    nradii=input("nradii=?")
if ("radsep" in key): radsep=float(val[key.index("radsep")])
else:
    print ("ERROR: RADSEP not found in the input file")
    radsep=input("radsep=?")
if ("vsys" in key): vsys=float(val[key.index("vsys")])
else:
    print ("WARNING: VSYS not found in the input file - 3D-Barolo will estimate it")
    vsys=-1
if ("vrot" in key): vrot=float(val[key.index("vrot")])
else: vrot=-1
if ("vdisp" in key): vdisp=float(val[key.index("vdisp")])
else: vdisp=-1
if ("vrad" in key): vrad=float(val[key.index("vrad")])
else: vrad=0
if ("inc" in key): inc=float(val[key.index("inc")])
else: 
    print ("WARNING: INCL not found in the input file - assuming 60")
    inc=60
if ("pa" in key): pa=float(val[key.index("pa")])
else:
    print ("WARNING: PA not found in the input file.")
    print ("Input it here or run 3D-Barolo to find its value or use trustGuess=false")
    pa=float(input("PA=?"))
if ("z0" in key): z0=float(val[key.index("z0")])
else:
    print ("WARNING: Z0 not found in the input file - assuming 0")
    z0=0
if ("cdens" in key): cdens=float(val[key.index("cdens")])
else: cdens=10
if ("redshift" in key): redshift=float(val[key.index("redshift")])
else: redshift=False
if ("restwave" in key): restwave=float(val[key.index("restwave")])
else: restwave=False
if ("restfreq" in key): restfreq=float(val[key.index("restfreq")])
else: restfreq=False


fitKin=False
if (val[key.index("fitkin")] in ['true', 'y', 'yes']): fitKin=True


#$%
if (useAllCube):
    if (not fitKin):
        fitKin=True
        print ("WARNING: fitKin has been set to true. When you use the cube and not the total flux map to calculate the residuals fitkin needs to be true.")


trustGuess=False
if (val[key.index("trustguess")] in ['true', 'y', 'yes']): trustGuess=True


if ("mask" in key):
    tmask=(val[key.index("mask")])
    if (tmask[:4] == 'file'):
        print ("Using mask provided by the user: ", tmask)
    mask=tmask
else: mask='smooth&search'


if ("factor" in key): factor=float(val[key.index("factor")])
else: factor=2
if ("blankcut" in key): blankcut=float(val[key.index("blankcut")])
else: blankcut=3
if ("snrcut" in key): snrcut=float(val[key.index("snrcut")])
else: snrcut=3
if ("growthcut" in key): growthcut=float(val[key.index("growthcut")])
else: growthcut=2.5
if ("linear" in key): linear=float(val[key.index("linear")])
else:
    print ("WARNING: LINEAR not found in the input file - assuming 0.42")
    linear=0.42


if ("distance" in key):
    distance=float(val[key.index("distance")])
else:
    print ("WARNING: distance not found in the input file.")    
    distance=input("distance=?")
lengthConv=1./(distance*1e3/(3600.*180/np.pi))


if ("xpos" in key):
    xpos=float(val[key.index("xpos")])
    if (reduceCube): xpos-=firstpix1
else:
    print ("WARNING: X-position of the centre not found in input file.")
    print ("Input it here or run 3D-Barolo to find its value or use trustGuess=false")
    xpos=input("XPOS=?")
if ("ypos" in key):
    ypos=float(val[key.index("ypos")])
    if (reduceCube): ypos-=firstpix2
else:
    print ("WARNING: Y-position of the centre not found in input file.")
    print ("Input it here or run 3D-Barolo to find its value or use trustGuess=false")
    ypos=input("YPOS=?")


### SPACEPAR PARAMETERS ###
nincl=int(val[key.index("nincl")])
secondPar=val[key.index("secondpar")]
nSecondPar=int(val[key.index("nsecondpar")])


### SETTING PARAMETERS MCMC ###
nwalkers=int(val[key.index("nwalkers")])
runs=val[key.index("runs")]
if (runmcmc):
    if (xpos == 0): xpos=np.random.rand(1)[0]
    if (ypos == 0): ypos=np.random.rand(1)[0]
    if (fitPA):
        if (pa == 0): pa=4*np.random.rand(1)[0]-2
        if (fitZ0):
            if (z0 < abs(0.2*lengthConv)):
                z0=abs(0.2*lengthConv)
                print ("WARNING: Z0 too small in input file, assuming 200 pc = {:.2f}".format(z0), "arcsec")
            if (fitSep):
                toFit = ['Incl', 'x0', 'y0', 'pa', 'Z0', 'radSep']
                par_ini=[inc, xpos, ypos, pa, z0, radsep]
            else:
                toFit = ['Incl', 'x0', 'y0', 'pa', 'Z0']
                par_ini=[inc, xpos, ypos, pa, z0]
        else:
            if (fitSep):
                toFit = ['Incl', 'x0', 'y0', 'pa', 'radSep']
                par_ini=[inc, xpos, ypos, pa, radsep]
            else:                
                toFit = ['Incl', 'x0', 'y0', 'pa']
                par_ini=[inc, xpos, ypos, pa]
    else:
        if (fitZ0):
            if (z0 < abs(0.2*lengthConv)):
                z0=abs(0.2*lengthConv)
                print ("WARNING: Z0 too small in input file, assuming 200 pc = {:.2f}".format(z0), "arcsec")
            if (fitSep):
                toFit = ['Incl', 'x0', 'y0', 'Z0', 'radSep']
                par_ini=[inc, xpos, ypos, z0, radsep]
            else:
                toFit = ['Incl', 'x0', 'y0', 'Z0']
                par_ini=[inc, xpos, ypos, z0]
        else:
            if (fitSep):
                toFit = ['Incl', 'x0', 'y0', 'radSep']
                par_ini=[inc, xpos, ypos, radsep]
            else:
                toFit = ['Incl', 'x0', 'y0'] 
                par_ini=[inc, xpos, ypos]
    ndim=len(toFit)
else:
    if (secondPar == 'x0'): par_ini=[inc, xpos]
    if (secondPar == 'y0'): par_ini=[inc, ypos]
    if (secondPar == 'radsep'): par_ini=[inc, radsep]
    if (secondPar == 'z0'): par_ini=[inc, z0]
    
### SETTING PARAMETER SPACE ###
deltaCentre=float(val[key.index("deltacentre")])
inclMin=float(val[key.index("inclmin")])
inclMax=float(val[key.index("inclmax")])
if (runmcmc):
    x0min=par_ini[1]-deltaCentre; x0max=par_ini[1]+deltaCentre
    y0min=par_ini[2]-deltaCentre; y0max=par_ini[2]+deltaCentre
    if (ndim>3):
        paMin=pa-90; paMax=pa+90
        if (useAllCube): paMin=pa-180; paMax=pa+180
        z0min = float(val[key.index("z0min")])
        z0max = float(val[key.index("z0max")])
        sepMin = float(val[key.index("radsepmin")])*radsep
        sepMax = float(val[key.index("radsepmax")])*radsep
else:
    if (secondPar == 'x0' or secondPar == 'y0'):
        secondMin = par_ini[1]-deltaCentre
        secondMax = par_ini[1]+deltaCentre
    if (secondPar == 'radsep'):
        sepMin = float(val[key.index("radsepmin")])*radsep
        sepMax = float(val[key.index("radsepmax")])*radsep
        secondMin = sepMin
        secondMax = sepMax
    if (secondPar == 'z0'):
        z0min = float(val[key.index("z0min")])
        z0max = float(val[key.index("z0max")])
        secondMin = z0min
        secondMax = z0max
        
def makeModel(_toFit):
    _pa=pa
    _radsep=radsep
    _z0=z0
    if (runmcmc):
        _xpos=_toFit[1]; _ypos=_toFit[2]
        if (len(_toFit)==4):
            if (fitPA): _pa=_toFit[3]
            if (fitZ0): _z0=_toFit[3]
            if (fitSep): _radsep=_toFit[3]
        if (len(_toFit)==5):
            if (fitPA and fitZ0): _pa=_toFit[3]; _z0=_toFit[4]
            if (fitPA and fitSep): _pa=_toFit[3]; _radsep=_toFit[4]
            if (fitSep and fitZ0): _z0=_toFit[3]; _radsep=_toFit[4]
        if (len(_toFit)==6): _pa=_toFit[3]; _z0=_toFit[4]; _radsep=_toFit[5]
    else:
        if (secondPar == 'x0'):
            _xpos=_toFit[1]
            _ypos=ypos
        if (secondPar == 'y0'):
            _ypos=_toFit[1]
            _xpos=xpos
        if (secondPar == 'radsep'):
            _xpos=xpos; _ypos=ypos
            _radsep=_toFit[1]
        if (secondPar == 'z0'):
            _xpos=xpos; _ypos=ypos
            _z0=_toFit[1]
    # General options
    opts = ['FITSFILE='+str(filename),
            'NRADII='+str(nradii),
            'RADSEP='+str(_radsep),
            'STARTRAD=0',
            'XPOS='+str(_xpos),
            'YPOS='+str(_ypos),
            'VSYS='+str(vsys),
            'VRAD='+str(vrad),
            'Z0='+str(_z0),
            'INC='+str(_toFit[0]),
            'PA='+str(_pa),
            'CDENS='+str(cdens),
            'NORM=AZIM',
            'MASK=file('+BBfitDir+'mask.fits)',
            'LINEAR='+str(linear),
            'DISTANCE='+str(distance),
            'PLOTS=false',
            'TWOSTAGE=false',
            'FLAGERRORS=false']
    if (redshift): opts = opts+['REDSHIFT='+str(redshift)]
    if (restwave): opts = opts+['RESTWAVE='+str(restwave)]
    if (restfreq): opts = opts+['RESTFREQ='+str(restfreq)]
    if (fitKin):
        opts = opts+['3DFIT=true',
                'VDISP='+str(vdisp),
                'VROT='+str(vrot),
                'FREE=VROT VDISP']                
    else:
        opts = opts+['GALMOD=true',
                     'VDISP=file('+BBfitDir+'rings_final1.txt'+',4,2)',
                     'VROT=file('+BBfitDir+'rings_final1.txt'+',3,2)']
                     
    cmd = [f'{BBexec}', '-c']
    cmd.extend(opts)
    if (finalModel): subprocess.call(cmd)
    subprocess.call(cmd,stdout=subprocess.DEVNULL)


def makeFirstModel(_toFit):
    _pa=pa
    _z0=z0
    _radsep=radsep
    if (runmcmc):
        _xpos=_toFit[1]; _ypos=_toFit[2]
        if (len(_toFit)==4):
            if (fitPA): _pa=_toFit[3]
            if (fitZ0): _z0=_toFit[3]
            if (fitSep): _radsep=_toFit[3]
        if (len(_toFit)==5):
            if (fitPA and fitZ0): _pa=_toFit[3]; _z0=_toFit[4]
            if (fitPA and fitSep): _pa=_toFit[3]; _radsep=_toFit[4]
            if (fitSep and fitZ0): _z0=_toFit[3]; _radsep=_toFit[4]
        if (len(_toFit)==6): _pa=_toFit[3]; _z0=_toFit[4]; _radsep=_toFit[5]
#    if (runmcmc):
#        _xpos=_toFit[1]; _ypos=_toFit[2]
#        if (len(_toFit)==4):
#            if (fitPA): _pa=_toFit[3]
#            else: _z0=_toFit[3]
#        if (len(_toFit)==5): _pa=_toFit[3]; _z0=_toFit[4]
    else:
        if (secondPar == 'x0'):
            _xpos=_toFit[1]
            _ypos=ypos
        if (secondPar == 'y0'):
            _ypos=_toFit[1]
            _xpos=xpos
        if (secondPar == 'radsep'):
            _xpos=xpos; _ypos=ypos
            _radsep=_toFit[1]
        if (secondPar == 'z0'):
            _xpos=xpos; _ypos=ypos
            _z0=_toFit[1]
    # General options
    opts = ['FITSFILE='+str(filename),
            '3DFIT=true',
            'NRADII='+str(nradii),
            'RADSEP='+str(_radsep),
            'STARTRAD=0',
            'VSYS='+str(vsys),
            'VDISP='+str(vdisp),
            'VROT='+str(vrot),
            'VRAD='+str(vrad),
            'Z0='+str(_z0),
            'PA='+str(_pa),
            'CDENS='+str(cdens),
            'NORM=AZIM',
            'FREE=VROT VDISP', 
            'MASK='+str(mask),
            'FACTOR='+str(factor),
            'BLANKCUT='+str(blankcut),
            'SNRCUT='+str(snrcut),
            'GROWTHCUT='+str(growthcut),
            'TWOSTAGE=false',
            'FLAGERRORS=false',            
            'LINEAR='+str(linear),
            'DISTANCE='+str(distance),
            'PLOTS=false']
    if (redshift): opts = opts+['REDSHIFT='+str(redshift)]
    if (restwave): opts = opts+['RESTWAVE='+str(restwave)]
    if (restfreq): opts = opts+['RESTFREQ='+str(restfreq)]
    #    if ((fitKin) or (not (fitKin) and (trustGuess))):
    if (trustGuess):
        opts = opts+['XPOS='+str(_xpos),
                'YPOS='+str(_ypos),
                'INC='+str(_toFit[0])]
    cmd = [f'{BBexec}', '-c']
    cmd.extend(opts)
    subprocess.call(cmd)


def calcResiduals():


    if (useAllCube):
        hdu_list = fits.open(BBfitDir+galname+'mod_azim.fits')
        modcube = hdu_list[0].data
        where_are_NaNs = np.isnan(modcube)
        modcube[where_are_NaNs] = 0.0


        if (normaliseMod):
            # Using the comparison of the flux in the total map between model and data #
            # Read total flux map of the current model #
            hdu_list = fits.open(BBfitDir+'maps/'+galname+'_azim_0mom.fits')
            modelTot = hdu_list[0].data
            # Removes NANs #
            where_are_NaNs = np.isnan(modelTot)
            modelTot[where_are_NaNs] = 0.0
            # calculate flux in the brightest pixels
            temp1Dmod=modelTot.flatten()
            temp1Dmod.sort()
            normaliz=sum(temp1Dmod[-int(nPixels/100.*fromTop):])/flux_brightest
            # Normalise the model #
            modcube/=normaliz
            
        # Set zeros outside the mask #
        if (useMaskMod=='union'):
            # masks the model below a measure of the extent of mask of the data
            where_is_outside = np.less(modcube, noise*growthcut/np.sqrt(factor))
            modcube[where_is_outside] = 0.0
        else:
            if (useMaskMod=='same'):
                modcube*=mask 
            else:
                if (useMaskMod=='threshold'):
                    # exclude whether both model and data are below thresholdAll
                    exclude_outside = np.less(modcube, thresholdAll*noise)
                    # exclude_outside2 = np.less(datacube, thresholdAll*noise)
                    # exclude_outside=exclude_outside1*exclude_outside2
                    modcube[exclude_outside] = 0.0
        hdu_list.close()


        if (errorType=='marasco'):
            # For details see Marasco et al. 2019 #
            # This division by nChansMask is a rough way to have the variation in the individual channel maps #
            sigma=rotatedRMS*(1.1331*(beam/cdelt2)**2)/np.sqrt(nChansMask)
        if (errorType=='dof'):
            # Obsolete - will be discarded #
            dof=nPixels/(np.pi*(0.5*beam/cdelt2)**2)-ndim
            sigma=rotatedRMS*np.sqrt(dof)/np.sqrt(nChansMask)
        sigma2=sigma**2


        if (residualType=='chisq'):
            residuals=0.5*sum(sum(sum((datacube-modcube)**2)))/sigma2
        if (residualType=='abs'):
            residuals=sum(sum(sum(abs(datacube-modcube))))/sigma
            
        return residuals


    else:


        # Read total flux map of the current model #
        hdu_list = fits.open(BBfitDir+'maps/'+galname+'_azim_0mom.fits')
        modelTot = hdu_list[0].data
        # Removes NANs #
        where_are_NaNs = np.isnan(modelTot)
        modelTot[where_are_NaNs] = 0.0
        # Normalise the model total-flux map #
        if (normaliseMod):
            # calculate flux in the brightest pixels
            temp1Dmod=modelTot.flatten()
            temp1Dmod.sort()
            normaliz=sum(temp1Dmod[-int(nPixels/100.*fromTop):])/flux_brightest
            modelTot/=normaliz


        # Set the model total map to zero outside the mask #
        if (useMaskMod=='union'):
            # mask model below the minimum value measure in the total-flux map #
            where_is_outside = np.less(modelTot,minTot_flux)
            modelTot[where_is_outside] = 0.0
        else:
            if (useMaskMod=='same'):
                # mask model as the data total-flux map # 
                modelTot*=totmask
            else:
                if (useMaskMod=='threshold'):
                    # mask model below a threshold
                    if (useMaskTot):
                        thresholdTot=lowestContour
                    else:
                        thresholdTot=lowestContUser
                    exclude_outside = np.less(modelTot, thresholdTot)
 #                   exclude_outside2 = np.less(dataTot, thresholdTot)
#                    exclude_outside=exclude_outside1*exclude_outside2
                    modelTot[exclude_outside] = 0.0
        hdu_list.close()


        if (errorType=='marasco'):
            # From Marasco et al. 2019
            sigma=rotatedRMS*(1.1331*(beam/cdelt2)**2)
            #1.331 = np.pi/(4*np.log(2)))
        if (errorType=='dof'):
            # Obsolete - will be discarded #
            dof=nPixels/(np.pi*(0.5*beam/cdelt2)**2)-ndim
            sigma=rotatedRMS*np.sqrt(dof)
        sigma2=sigma**2


        if (residualType=='chisq'):
            residualsTot=0.5*sum(sum((resDataTot-modelTot)**2))/sigma2
        if (residualType=='abs'):
            residualsTot=(sum(sum(abs(resDataTot-modelTot))))/sigma


        return residualsTot


def log_prior(theta):
    if len(theta)==3:
        incl, x0, y0 = theta
        if inclMin <incl <inclMax and x0min <x0 <x0max and y0min <y0 <y0max:
            return 0.0
    if len(theta)==4:
        if (fitPA):
            incl, x0, y0, pa = theta
            if inclMin <incl <inclMax and x0min <x0 <x0max and y0min <y0 <y0max and paMin <pa <paMax:
                return 0.0
        if (fitZ0):
            incl, x0, y0, z0 = theta
            if inclMin <incl <inclMax and x0min <x0 <x0max and y0min <y0 <y0max and z0min <z0 <z0max:
                return 0.0
        if (fitSep):
            incl, x0, y0, radsep = theta
            if inclMin <incl <inclMax and x0min <x0 <x0max and y0min <y0 <y0max and sepMin <radsep <sepMax:
                return 0.0
    if len(theta)==5:
        if (fitPA and fitZ0):
            incl, x0, y0, pa, z0 = theta
            if inclMin <incl <inclMax and x0min <x0 <x0max and y0min <y0 <y0max and paMin <pa <paMax and z0min <z0 <z0max:
                return 0.0
        if (fitPA and fitSep):
            incl, x0, y0, pa, radsep = theta
            if inclMin <incl <inclMax and x0min <x0 <x0max and y0min <y0 <y0max and paMin <pa <paMax and sepMin <radsep <sepMax:
                return 0.0
        if (fitSep and fitZ0):
            incl, x0, y0, z0, radsep = theta
            if inclMin <incl <inclMax and x0min <x0 <x0max and y0min <y0 <y0max and sepMin <radsep <sepMax and z0min <z0 <z0max:
                return 0.0
    if len(theta)==6:
        incl, x0, y0, pa, z0, radsep = theta
        if inclMin <incl <inclMax and x0min <x0 <x0max and y0min <y0 <y0max and paMin <pa <paMax and z0min <z0 <z0max and sepMin <radsep <sepMax:
            return 0.0            
    return -np.inf


def log_likelihood(theta):
    makeModel(theta)
#    print (theta)
    return -calcResiduals()


def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta)
    return lp+ll


###                ###
### BEGINS TO WORK ###
###                ###


### Running model first time to produce mask and total map ###
print ()
print ("Making first model with BBarolo")
makeFirstModel(par_ini)
if (vrot==-1):
    vrot=np.mean(np.ndarray.tolist(np.genfromtxt(fname=BBfitDir+'rings_final1.txt', dtype='float',  skip_header=1, usecols=2, unpack=True)))
if (vdisp==-1):
    vdisp=np.mean(np.ndarray.tolist(np.genfromtxt(fname=BBfitDir+'rings_final1.txt', dtype='float',  skip_header=1, usecols=3, unpack=True)))
if (vsys==-1):
    vsys=np.ndarray.tolist(np.genfromtxt(fname=BBfitDir+'rings_final1.txt', dtype='float',  skip_header=1, usecols=11, max_rows=1, unpack=True))


    
### READING MASK ###
hdu_list = fits.open(BBfitDir+'mask.fits')
mask = hdu_list[0].data
hdu_list.close()


### READING and masking the DATACUBE ###
hdu_list = fits.open(filename)
datacube = hdu_list[0].data
# Remove NANs # 
where_are_NaNs = np.isnan(datacube)
datacube[where_are_NaNs] = 0.0 
datacube*=mask
hdu_list.close()


### WORKING ON THE TOTAL-FLUX MAP ###
# Reading #
hdu_list = fits.open(BBfitDir+'maps/'+galname+'_0mom.fits')
dataTot = hdu_list[0].data
# Remove NANs #
where_are_NaNs = np.isnan(dataTot)
dataTot[where_are_NaNs] = 0.0
totHeader = hdu_list[0].header


# Calculate conversion factor between total-flux map in 3D-Barolo and one by simply summing channels #
# This is important because 3D-Barolo changes the units into Jy km/s (Jy/beam km/s, after vers. 1.6.4) #
if (naxis == 3):
    dataTotSimpleSum = sum(datacube) # note that datacube is already masked
if (naxis == 4):
    dataTotSimpleSum = sum(sum(datacube)) # note that datacube is already masked
conversionTot = sum(sum(dataTot))/sum(sum(dataTotSimpleSum))
conversionTot *= linear/0.4246 # if linear=0.85 there is twice the flux in the SimpleSum map (BB instead accounts for this) 
#print (conversionTot)


# Sum of the mask #
totmask=sum(mask) # for the moment each pixel = number of channels that contribute to the mask
# Estimate of average channels that contribute to the mask #
nChansMask=totmask[np.nonzero(totmask)].mean()
# Building S/N total map #
noiseTot=np.sqrt(totmask/(linear/0.4246))*conversionTot*noise # divided by linear/0.4246 because independent channels
StoNtot=np.divide(dataTot, noiseTot, out=np.zeros(dataTot.shape, dtype=float), where=noiseTot!=0)
# Identification of the pixels at the user-defined S/N #
pseudo1sigma = (StoNtot> 0.95)*(StoNtot< 1.05)
pseudoContour = (StoNtot> noiseTimes*0.95)*(StoNtot<noiseTimes*1.05)
# Estimating noise in the total map #
averageNoiseTot =(np.average(np.array(dataTot)[pseudo1sigma]))
lowestContour =(np.average(np.array(dataTot)[pseudoContour]))


# Output fits files with Signa to Noise map and number of channels that contribute to the mask #
hduMask=fits.PrimaryHDU(StoNtot)
hduMask.header=totHeader
hduMask.writeto('StoNtot_'+galname+'.fits', overwrite=True)
hduMask=fits.PrimaryHDU(totmask)
hduMask.header=totHeader
hduMask.writeto('channelsInMask_'+galname+'.fits', overwrite=True)


# MASKING THE TOTAL-FLUX MAP #
if (useMaskTot):
    # Make total mask with the criterion of the maximum number of channels (cutMask) # 
    where_is_outside = np.less(totmask,cutMask)
    totmask[where_is_outside] = 0.0
    # Set region inside the mask = 1
    where_is_inside = np.greater_equal(totmask,cutMask)
    totmask[where_is_inside] = 1.0
    # set total-flux map at zero outside the total mask #
    dataTot*=totmask 
    # Export total-flux map in fits # 
    hduMask=fits.PrimaryHDU(totmask)
    hduMask.header=totHeader
    hduMask.writeto('totmask_'+galname+'.fits', overwrite=True)
    thresholdTot=lowestContour
else:
    # exclude outside outer contour #
    exclude_outside = np.less_equal(dataTot,lowestContUser)
    dataTot[exclude_outside] = 0.0
    thresholdTot=lowestContUser


# ROTATE MAP BY 180 DEGREES and subtract # 
nx, ny = dataTot.shape[0], dataTot.shape[1]
x0, y0 = int(xpos), int(ypos)
diffRotated=np.zeros((nx, ny))
for i in range(0,nx):
    for j in range(0,ny):
        if (2*x0-1 >= 0 and 2*x0-i <nx):
            if (2*y0-1 >= 0 and 2*y0-j <ny):
                diffRotated[2*x0-i,2*y0-j]=dataTot[i,j]
diffRotated=(diffRotated-dataTot)
# Put to NaN any value below pseuso-1 or pseuso-noisetimes sigma
where_is_outside = np.less(diffRotated, pseudo1sigma)
diffRotated[where_is_outside] = np.nan
# Mediam value of the (absolute) residuals - 0.5 because an ideal model will be in between the 2 sides #
rotatedRMS=np.nanstd(diffRotated)
#print (rotatedRMS)
#rotatedRMS=np.nanmean(abs(diffRotated))
#print (rotatedRMS)


# Output fits file with masked total map #
hduMasked=fits.PrimaryHDU(dataTot)
hduMasked.header=totHeader
hduMasked.writeto('totfluxMap_'+galname+'.fits', overwrite=True)
# Create residual array
resDataTot=dataTot
# Write dataTot for threshold residuals
if (useMaskMod=='threshold'):
    excludeData_outside = np.less(dataTot, thresholdTot)
    resDataTot[excludeData_outside] = 0.0


# FLUXES after masking # 
nPixels=np.count_nonzero(dataTot)
tot_flux=sum(sum(dataTot)) 
minTot_flux=np.min(dataTot[np.nonzero(dataTot)])


print ()
print ('In the total-flux map')
print ('Pixels inside the mask: ', nPixels)
print ("Flux inside the mask: {:.2f}".format(tot_flux), totHeader['BUNIT'])


# calculate sum of flux in the brightest pixels # 
tempTot=np.copy(dataTot)
temp1D=tempTot.flatten()
temp1D.sort()
flux_brightest=(sum(temp1D[-int(nPixels/100.*fromTop):]))
print ("Flux from the brighetest pixels: {:.2f}".format(flux_brightest), totHeader['BUNIT'])
print ("Average RMS noise in the total-flux map: {:.6f}".format(averageNoiseTot), totHeader['BUNIT'])
print ("Estimated std deviation after 180-deg rotation and subtraction (used in likelihood): {:.6f}".format(rotatedRMS), totHeader['BUNIT'])


print ()
print ("RMS noise per channel: {:.6f}".format(noise))
print ("Conversion: 1 arsec = {:.2f}".format(1./lengthConv), " kpc")




# Plot total-flux map #
fig, ax = plt.subplots(1,1)
totPlot=np.flip(dataTot,0)
if (useMaskTot):
    maskPlot=np.flip(totmask,0)
img = ax.imshow(totPlot, interpolation='none', aspect='equal', cmap='summer')
fig.colorbar(img)
if (useMaskTot):
    lowcont=lowestContour
else:
    lowcont=lowestContUser
levels=np.array((lowcont, lowcont*2, lowcont*4, lowcont*8, lowcont*16, lowcont*32, lowcont*64, lowcont* 128))
ax.contour(totPlot, levels, colors='k')
if (plotTotMask):
    ax.contour(maskPlot, 1, colors='white')
fig.savefig('totFluxMap_'+galname+'.jpg', format='jpg', dpi=300, orientation='portrait')


hdu_list.close()


if not (fitKin):
    if (runmcmc):
        print ()
        print ("The kinematic is only estimated in the first iteration and kept fixed.")
        if (trustGuess):
            print ("For this, I have used the first guesses provided by the user for inclination and centre.\n")
        else:
            print ("For this, I have automatically found values for inclination and centre.\n")


print ()


# SUMMARY #


print ("====================== Summary of parameter values =======================")
print ()
print ("   Working on {}".format(filename))
print ("   {:48s} {:>16s} {} ".format('Fit position angle?', '[fitPA] =', bool(fitPA)))
print ("   {:48s} {:>16s} {}".format('Fit thickness?', '[fitZ0] =', bool(fitZ0)))
print ("   {:48s} {:>16s} {}".format('Fit extension of the disc (radSep)?', '[fitSep] =', bool(fitSep)))
print ("   {:48s} {:>16s} {}".format('Fit kinematics at each iteration?', '[fitKin] =', bool(fitKin)))
print ("   {:48s} {:>16s} {}".format('Trust initial guesses for centre?', '[trustGuess] =', bool(trustGuess)))
print ("   {:48s} {:>16s} {}".format('Use the whole datacube (not the total flux map)?', '[useAllCube] =', bool(useAllCube)))
print ("   {:48s} {:>16s} {}".format('Normalise the flux in the model?', '[normaliseMod] =', bool(normaliseMod)))
if not (useAllCube):
    print ("   {:48s} {:>16s} {}".format('Mask the total flux map in the data?', '[useMaskTot] =', bool(useMaskTot)))
print ("   {:48s} {:>16s} {}".format('How to merge model and data masks?', '[useMaskMod] =', useMaskMod))
print ("   {:48s} {:>16s} {}".format('Likelihood sigma: error type ', '[errorType] =', errorType))
#print ("   How to calculate the likelihood? {}".format(residualType))
if (residualType == 'abs'):
    print ('   Using absolute-value residuals')
else:
    if (residualType == 'chisq'):
        print ('   Using chi-square residuals')
    else:
        print ('WARNING: No residualType specified, using absolute value')
        residualType='abs'
print ()
if (useMaskTot):
    print ("{} {} {} {:.6f} {:s}".format('The lowest contour in the total map at approximately', float(noiseTimes), 'r.m.s. noise is', float(lowestContour), totHeader['BUNIT']))
    
###                    ###
### FITTING & PLOTTING ###
###                    ###


if (runmcmc):


    ### MCMC ###
    print ()
    print ('STARTING MCMC exploration')
    pos = [par_ini*(1.+1e-2*np.random.randn(ndim)) for i in range(nwalkers)]
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(), threads=15)


    if (runs == 'convergence'):
        max_runs = 20000
        index = 0
        autocorr = np.empty(max_runs)
        old_tau = np.inf


#        for pos,lnp,rstate in tqdm(sampler.sample(pos0, iterations=nsteps)):
#            pass
        for sample in sampler.sample(pos, iterations=max_runs, progress=True):
            if sampler.iteration % 10: # Check every 10 steps
                continue


            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            print("autocorrelation times: ", tau)
            print("Average autocorrelation time: {:.2f}".format(autocorr[index]))
            index += 1


            # Check convergence
            converged = np.all(tau * 50 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau
        samples = sampler.get_chain()
        flat_samples = sampler.get_chain(discard=int(sampler.iteration*0.2), thin=15, flat=True)
    else:
        sampler.run_mcmc(pos, int(runs), progress=True);
        samples = sampler.get_chain()
        flat_samples = sampler.get_chain(discard=int(int(runs)*0.2), thin=15, flat=True)


    print("flat chain shape: {0}".format(samples.shape))
            
    ### CORNER PLOT ###
    fig = corner.corner(
        flat_samples, labels=toFit, smooth=True, quantiles=[0.16, 0.5, 0.84], title_fmt=".3f", show_titles=True
    );
    fig.savefig('cannubi_'+galname+'.jpg', format='jpg', dpi=300, orientation='portrait')


    finalPar=[]
    tfinalPar=[]
    print ()
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(toFit[i],": %.5f: -%.5f +%.f" %(mcmc[1], q[0], q[1]))
        tfinalPar.append(mcmc[1])
    if (fitZ0):
        print ("Conversion: 1 arsec = {:.2f}".format(1./lengthConv), " kpc")
    finalPar=tfinalPar
    
    ### Plot total-flux map overlaid with the final model ###
    fig, ax = plt.subplots(1,1)
    img = ax.imshow(totPlot, interpolation='none', aspect='equal', cmap='summer')
    fig.colorbar(img)
    levels=np.array((lowcont, lowcont*4, lowcont*16, lowcont*64))
    ax.contour(totPlot, levels, colors='k')
    if (plotTotMask):
        ax.contour(maskPlot, 1, colors='white')
    finalModel=True
    makeModel(finalPar)
    hdu_list = fits.open(BBfitDir+'maps/'+galname+'_azim_0mom.fits')
    modtotlast = hdu_list[0].data
    where_are_NaNs = np.isnan(modtotlast)
    modtotlast[where_are_NaNs] = 0.0
    if (normaliseMod):
        temp1Dmod=modtotlast.flatten()
        temp1Dmod.sort()
        normaliz=sum(temp1Dmod[-int(nPixels/100.*fromTop):])/flux_brightest
        modtotlast/=normaliz
    modtotPlot=np.flip(modtotlast,0)
    ax.contour(modtotPlot, levels, colors='darkred')
    fig.savefig('totFluxMapCompare_'+galname+'.jpg', format='jpg', dpi=300, orientation='portrait')
    print ()
    print ('The 3D-Barolo model with the final parameters is in ./output/')
    
else:


    ### SPACEPAR (INCL VS SECOND PARAMETER) ###
    print ()
    print ('STARTING SPACEPAR exploration')
    deltaIncl=(inclMax-inclMin)/(nincl-1)
    deltaSec=(secondMax-secondMin)/(nSecondPar-1)
    resMatrix=np.zeros((nincl, nSecondPar))
    for i in range(0,nincl):
        for j in range(0,nSecondPar):
            incl=inclMin+deltaIncl*i
            secondParVal=secondMin+deltaSec*j
            thetaSpace = incl, secondParVal
            makeModel(thetaSpace)
            resMatrix[i,j]=calcResiduals()
            print ("Progress: i={:d}".format(i), " j={:d}".format(j), " incl={:.1f}".format(incl), secondPar, "={:.1f}".format(secondParVal), " residuals={:.2f}".format(resMatrix[i,j]), end='\r', flush=True)
            
    ### PLOTTING SPACEPAR RESULTS ###
    fig, ax = plt.subplots(1,1)
    img = ax.imshow(resMatrix, interpolation='none', extent=[secondMin, secondMax, inclMax, inclMin], aspect='auto')
    levels=np.arange(np.amin(resMatrix), np.amax(resMatrix)*1.2, (np.amax(resMatrix)-np.amin(resMatrix))/20.)
    ax.contour(resMatrix, levels, colors='k', extent=[secondMin, secondMax, inclMin, inclMax])
    fig.colorbar(img)
    fig.savefig('cannubiSpacepar_'+galname+'.jpg', format='jpg', dpi=300, orientation='portrait')


    ### SAVE FITS FILE WITH SPACEPAR RESULTS ###
    residualMap=fits.PrimaryHDU(resMatrix)
    residualMap.header.append(('CTYPE1', ' pixel'), end=True)
    residualMap.header.append(('CTYPE2', ' pixel'), end=True)
    residualMap.header.append(('CUNIT1', ' deg'), end=True)
    residualMap.header.append(('CUNIT2', ' pixel'), end=True)
    residualMap.header.append(('CRPIX1', 1), end=True)
    residualMap.header.append(('CRPIX2', 1), end=True)
    residualMap.header.append(('CDELT1', deltaSec), end=True)
    residualMap.header.append(('CDELT2', deltaIncl), end=True)
    residualMap.header.append(('CRVAL1', secondMin), end=True)
    residualMap.header.append(('CRVAL2', inclMin), end=True)
    residualMap.writeto('residualMap_'+galname+'.fits', overwrite=True)
