import subprocess

# EXECUTABLE LISTS:
# PARAMETER BOUNDS:
vrots = [[0, 30], [0, 30], [0, 30], [0, 30], [0, 30], [0, 30], [0, 350], [0, 350], [0, 350], [0, 350], [0, 350], [0, 350], [0, 500], [0, 500], [0, 500], [0, 500], [0, 500], [0, 500]]
vdisps = [[1, 30], [1, 30], [1, 30], [1, 30], [1, 30], [1, 30], [1, 40], [1, 40], [1, 40], [1, 40], [1, 40], [1, 40], [1, 100], [1, 100], [1, 100], [1, 100], [1, 100], [1, 100]]
incs = [[25, 65], [25, 65], [25, 65], [25, 65], [25, 65], [25, 65], [25, 80], [25, 80], [25, 80], [25, 80], [25, 80], [25, 80], [1, 89], [1, 89], [1, 89], [1, 89], [1, 89], [1, 89]]
phis = [[-20, 20], [-20, 20], [-20, 20], [-20, 20], [-20, 20], [-20, 20], [-20, 60], [-20, 60], [-20, 60], [-20, 60], [-20, 60], [-20, 60], [-40, 120], [-40, 120], [-40, 120], [-40, 120], [-40, 120], [-40, 120]]
denss = [[0, 20], [0, 20], [0, 20], [0, 20], [0, 20], [0, 20], [1, 60], [1, 60], [1, 60], [1, 60], [1, 60], [1, 60], [1, 100], [1, 100], [1, 100], [1, 100], [1, 100], [1, 100]]

freepar = [['vrot', 'vdisp', 'inc_single', 'phi_single'], ['vrot', 'vdisp', 'dens', 'inc_single', 'phi_single']]
fittings = [freepar[0], freepar[0], freepar[0], freepar[1], freepar[1], freepar[1], freepar[0], freepar[0], freepar[0], freepar[1], freepar[1], freepar[1], freepar[0], freepar[0], freepar[0], freepar[1], freepar[1], freepar[1]]

masks = ["SEARCH", "NONE", "SMOOTH", "SEARCH", "NONE", "SMOOTH", "SEARCH", "NONE", "SMOOTH", "SEARCH", "NONE", "SMOOTH", "SEARCH", "NONE", "SMOOTH", "SEARCH", "NONE", "SMOOTH"]

models = ["C_reg_Se", "C_reg_No", "C_reg_Sm", "C_reg_den_Se", "C_reg_den_No", "C_reg_den_Sm", "C_irr_Se", "C_irr_No", "C_irr_Sm", "C_irr_den_Se", "C_irr_den_No", "C_irr_den_Sm", "C_all_Se", "C_all_No", "C_all_Sm", "C_all_den_Se", "C_all_den_No", "C_all_den_Sm"]

# Ensure all lists have the same length
assert len(vrots) == len(vdisps) == len(incs) == len(phis) == len(denss) == len(fittings) == len(masks) == len(models), "All parameter lists must have the same length"

# Loop through the parameters and execute BBB_template
for i in range(len(vrots)):
    vrot = vrots[i]
    vdisp = vdisps[i]
    inc = incs[i]
    phi = phis[i]
    dens = denss[i]
    fitting = fittings[i]
    mask = masks[i]
    model = models[i]
    print(f"Running BBB_template with vrot: {vrot}, vdisp: {vdisp}, inc: {inc}, phi: {phi}, dens: {dens}, fitting: {fitting}, mask: {mask}, model: {model}")
    subprocess.run(['python', '/home/user/THESIS/scripts/BBB_test_new.py', '--vrot', str(vrot), '--vdisp', str(vdisp), '--inc', str(inc), '--phi', str(phi), '--dens', str(dens), '--fitting', ','.join(fitting), '--mask', mask, '--model', model])