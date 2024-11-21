from pyBBarolo.wrapper import PVSlice

BBmain = "/Users/blanca/Documents/TESIS/software/Bbarolo-1.7/BBarolo"

fitsname = "/Users/blanca/Documents/TESIS/software/code/tests/first_try/barbamodel/barbamodelmod_nonorm.fits"

angle = 30

pvslice_a = PVSlice( fitsname=fitsname, XPOS_PV=25.5, YPOS_PV=25.5,  PA_PV=angle, OUTFOLDER= "SLICES",  )              
pvslice_a.run(BBmain)
