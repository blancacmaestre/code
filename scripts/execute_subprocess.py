import subprocess

# EXECUTABLE LISTS:
# PARAMETER BOUNDS:
#vrots = [[0, 30], [0, 30], [0, 30], [0, 30], [0, 30], [0, 30], [0, 350], [0, 350], [0, 350], [0, 350], [0, 350], [0, 350], [0, 500], [0, 500], [0, 500], [0, 500], [0, 500], [0, 500]]
#vdisps = [[1, 30], [1, 30], [1, 30], [1, 30], [1, 30], [1, 30], [1, 40], [1, 40], [1, 40], [1, 40], [1, 40], [1, 40], [1, 100], [1, 100], [1, 100], [1, 100], [1, 100], [1, 100]]
#incs = [[25, 65], [25, 65], [25, 65], [25, 65], [25, 65], [25, 65], [25, 80], [25, 80], [25, 80], [25, 80], [25, 80], [25, 80], [1, 89], [1, 89], [1, 89], [1, 89], [1, 89], [1, 89]]
#phis = [[-20, 20], [-20, 20], [-20, 20], [-20, 20], [-20, 20], [-20, 20], [-20, 60], [-20, 60], [-20, 60], [-20, 60], [-20, 60], [-20, 60], [-40, 120], [-40, 120], [-40, 120], [-40, 120], [-40, 120], [-40, 120]]
#denss = [[0, 20], [0, 20], [0, 20], [0, 20], [0, 20], [0, 20], [1, 60], [1, 60], [1, 60], [1, 60], [1, 60], [1, 60], [1, 100], [1, 100], [1, 100], [1, 100], [1, 100], [1, 100]]

freepar = [['vrot', 'vdisp', 'inc_single', 'phi_single'], ['vrot', 'vdisp', 'dens', 'inc_single', 'phi_single']]
#fittings = [freepar[0], freepar[0], freepar[0], freepar[0], freepar[0]]
fittings = [freepar[0]]
#masks = ["SEARCH", "SEARCH", "SEARCH", "SEARCH","SEARCH"]
masks = ["SEARCH"]
#models = ["CGal_6_70_0.01_B", "CGal_5_70_0.01_B", "CGal_4_70_0.01_B", "CGal_3_70_0.01_B","CGal_2_70_0.01_B"]
models = ["model4_test"]
#beamsizes = [40,48,60,80,120]
beamsizes = [60]
#fitsnames = [ "/home/user/THESIS/models_tests/CGal_6_70_0.01/CGal_6_70_0.01.fits","/home/user/THESIS/models_tests/CGal_5_70_0.01/CGal_5_70_0.01.fits","/home/user/THESIS/models_tests/CGal_4_70_0.01/CGal_4_70_0.01.fits","/home/user/THESIS/models_tests/CGal_3_70_0.01/CGal_3_70_0.01.fits","/home/user/THESIS/models_tests/CGal_2_70_0.01/CGal_2_70_0.01.fits"]
fitsnames = ["/home/user/THESIS/models/model4/model4.fits"]
# Ensure all lists have the same length
assert """ len(vrots) == len(vdisps) == len(incs) == len(phis) == len(denss) ==  """ 
len(fittings) == len(masks) == len(models) == len(fitsnames) == len(beamsizes), "All parameter lists must have the same length"

# Loop through the parameters and execute BBB_template
for i in range(len(fittings)):
    #vrot = vrots[i]
    #vdisp = vdisps[i]
    #inc = incs[i]
    #phi = phis[i]
    #dens = denss[i]
    fitting = fittings[i]
    mask = masks[i]
    model = models[i]
    fitsname = fitsnames[i]
    beamsize = beamsizes[i]

    print(f"Running BBB_template with fitting: {fitting}, mask: {mask}, model: {model}, beamsize: {beamsize}, fitsname: {fitsname}") #vrot: {vrot}, vdisp: {vdisp}, inc: {inc}, phi: {phi}, dens: {dens}, 
    subprocess.run(['python', '/home/user/THESIS/scripts/BBB_test_subprocess.py','--fitting', ','.join(fitting), '--mask', mask, '--model', model,  '--beamsize', str(beamsize),  '--fitsname', fitsname])#