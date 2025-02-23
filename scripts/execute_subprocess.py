import subprocess
import numpy as np

# EXECUTABLE LISTS:
# PARAMETER BOUNDS:
#vrots = [[0, 30], [0, 30], [0, 30], [0, 30], [0, 30], [0, 30], [0, 350], [0, 350], [0, 350], [0, 350], [0, 350], [0, 350], [0, 500], [0, 500], [0, 500], [0, 500], [0, 500], [0, 500]]
#vdisps = [[1, 30], [1, 30], [1, 30], [1, 30], [1, 30], [1, 30], [1, 40], [1, 40], [1, 40], [1, 40], [1, 40], [1, 40], [1, 100], [1, 100], [1, 100], [1, 100], [1, 100], [1, 100]]
#incs = [[25, 65], [25, 65], [25, 65], [25, 65], [25, 65], [25, 65], [25, 80], [25, 80], [25, 80], [25, 80], [25, 80], [25, 80], [1, 89], [1, 89], [1, 89], [1, 89], [1, 89], [1, 89]]
#phis = [[-20, 20], [-20, 20], [-20, 20], [-20, 20], [-20, 20], [-20, 20], [-20, 60], [-20, 60], [-20, 60], [-20, 60], [-20, 60], [-20, 60], [-40, 120], [-40, 120], [-40, 120], [-40, 120], [-40, 120], [-40, 120]]
#denss = [[0, 20], [0, 20], [0, 20], [0, 20], [0, 20], [0, 20], [1, 60], [1, 60], [1, 60], [1, 60], [1, 60], [1, 60], [1, 100], [1, 100], [1, 100], [1, 100], [1, 100], [1, 100]]

#freepar = [['vrot', 'vdisp', 'inc_single', 'phi_single'], ['vrot', 'vdisp', 'dens', 'inc_single', 'phi_single']]
#fittings = [freepar[0], freepar[0], freepar[0], freepar[0], freepar[0]]
#fittings = [freepar[0]]

""" masks = ["SEARCH", "SEARCH", "SEARCH","SEARCH","SEARCH"]
models = ["CGal_6_70_0.01_D","CGal_5_70_0.01_D", "CGal_4_70_0.01_D", "CGal_3_70_0.01_D", "CGal_2_70_0.01_D_inctest"]
beamsizes = [40,48,60,80,120]
halfbeams = [20,24,30,40,60]
centres = [38.5,32,25.5,19,12.5]
fitsnames = [ "/home/user/THESIS/models/A_MODELS_new/new_PA/CGal_6_70_0.01/CGal_6_70_0.01.fits",\
              "/home/user/THESIS/models/A_MODELS_new/new_PA/CGal_5_70_0.01/CGal_5_70_0.01.fits",\
              "/home/user/THESIS/models/A_MODELS_new/new_PA/CGal_4_70_0.01/CGal_4_70_0.01.fits",\
              "/home/user/THESIS/models/A_MODELS_new/new_PA/CGal_3_70_0.01/CGal_3_70_0.01.fits",\
              "/home/user/THESIS/models/A_MODELS_new/new_PA/CGal_2_70_0.01/CGal_2_70_0.01.fits"] """



masks = ["SEARCH", "SEARCH", "SEARCH","SEARCH","SEARCH", "SEARCH"]
models = ["CGal_4_30_0.01_D_5000", "CGal_4_40_0.01_D_5000", "CGal_4_50_0.01_D_5000", "CGal_4_60_0.01_D_5000", "CGal_4_70_0.01_D_5000", "CGal_4_80_0.01_D_5000"] 
beamsizes = np.full(len(models),60)  
halfbeams = np.full(len(models),30)
centres = np.full(len(models),25.5)
fitsnames = [ "/home/user/THESIS/models/A_MODELS_new/new_attempt/CGal_4_30_0.01/CGal_4_30_0.01.fits",\
              "/home/user/THESIS/models/A_MODELS_new/new_attempt/CGal_4_40_0.01/CGal_4_40_0.01.fits",\
              "/home/user/THESIS/models/A_MODELS_new/new_attempt/CGal_4_50_0.01/CGal_4_50_0.01.fits",\
              "/home/user/THESIS/models/A_MODELS_new/new_attempt/CGal_4_60_0.01/CGal_4_60_0.01.fits",\
              "/home/user/THESIS/models/A_MODELS_new/new_attempt/CGal_4_70_0.01/CGal_4_70_0.01.fits",\
              "/home/user/THESIS/models/A_MODELS_new/new_attempt/CGal_4_80_0.01/CGal_4_80_0.01.fits"]

# Ensure all lists have the same length
assert """ len(vrots) == len(vdisps) == len(incs) == len(phis) == len(denss) ==  len(fittings) == """ 

len(masks) == len(models) == len(fitsnames) == len(beamsizes) == len(centres) == len(halfbeams), "All parameter lists must have the same length"

# Loop through the parameters and execute BBB_template
for i in range(len(models)):
    #vrot = vrots[i]
    #vdisp = vdisps[i]
    #inc = incs[i]
    #phi = phis[i]
    #dens = denss[i]
    #fitting = fittings[i]
    mask = masks[i]
    model = models[i]
    fitsname = fitsnames[i]
    beamsize = beamsizes[i]
    centre = centres[i]
    halfbeam = halfbeams[i]

    print(f"Running BBB_template with  mask: {mask}, model: {model}, beamsize: {beamsize}, fitsname: {fitsname}, centre: {centre}, halfbeam: {halfbeam}") #vrot: {vrot}, vdisp: {vdisp}, inc: {inc}, phi: {phi}, dens: {dens}, fitting: {fitting},
    subprocess.run(['python', '/home/user/THESIS/scripts/BBB_test_subprocess.py', '--mask', mask, '--model', model,  '--beamsize', str(beamsize),  '--fitsname', fitsname, '--centre', str(centre),  '--halfbeam', str(halfbeam) ])# '--fitting', ','.join(fitting),