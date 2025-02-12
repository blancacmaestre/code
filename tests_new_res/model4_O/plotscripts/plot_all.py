########################################################################
#### This script simply calls all other python scripts for plotting ####
########################################################################
import os 

scriptdir = '/Users/blanca/Documents/TESIS/software/THESIS/tests_new_res//model4_O/plotscripts/' 
cmd = '' 

for f in os.listdir(scriptdir): 
	if '.py' in f and f!='plot_all.py': 
		cmd += 'python "%s/%s" & '%(scriptdir,f) 

os.system(cmd[:-2]) 
