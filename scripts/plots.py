import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

# Load the data cube (replace with your actual data cube FITS file path)
cube_fits_file = '/Users/blanca/Documents/TESIS/software/code/TESTS/tests_inc/velset_inc30/barbamodel/barbamodel.fits'  # Replace with your actual data cube file path
hdulist = fits.open(cube_fits_file)

# Extract the data cube (assuming the cube is stored in the first HDU)
cube_data = hdulist[0].data  # This is the 3D array (shape: [velocity, position, position])
header = hdulist[0].header

# Extract WCS to get the velocity axis (assuming the velocity axis is the 0th axis)
wcs = WCS(header)

# Create an array of pixel indices for the velocity axis
velocity_pixels = np.arange(cube_data.shape[0])  # Pixel indices for the velocity axis

# Create meshgrid for the position axes (assuming the position axes are the 1st and 2nd axes)
x_pixels, y_pixels = np.meshgrid(np.arange(cube_data.shape[1]), np.arange(cube_data.shape[2]), indexing='ij')

# Flatten the x and y position pixel arrays to match the total number of positions
x_pixels_flat = x_pixels.flatten()
y_pixels_flat = y_pixels.flatten()

# Now we need to create a grid of velocity with all position combinations
# Repeat the velocity pixels for each position pixel
velocity_grid = np.tile(velocity_pixels, len(x_pixels_flat) // len(velocity_pixels))  # Repeat velocity for each position

# Check the sizes of the arrays before stacking
print(f"velocity_grid size: {velocity_grid.size}")
print(f"x_pixels_flat size: {x_pixels_flat.size}")
print(f"y_pixels_flat size: {y_pixels_flat.size}")

# Combine the velocity, x, and y positions into a single 2D array
pixel_coordinates = np.vstack((velocity_grid, np.tile(x_pixels_flat, len(velocity_pixels)), np.tile(y_pixels_flat, len(velocity_pixels)))).T

# Convert pixel indices to world coordinates (velocity in km/s)
world_coordinates = wcs.all_pix2world(pixel_coordinates, 0)

# Extract the velocity, x, and y world coordinates
velocities = world_coordinates[:, 0]  # Velocity world coordinates
positions_x = world_coordinates[:, 1]  # X world coordinates
positions_y = world_coordinates[:, 2]  # Y world coordinates

# Load the PV slice (replace with your actual PV slice FITS file path)
pv_slice_fits_file = '/Users/blanca/Documents/TESIS/software/code/TESTS/tests_inc/velset_inc30/slices/barbamodelpv_80.fits'  # Replace with your FITS file path containing the PV slice
hdulist_pv = fits.open(pv_slice_fits_file)
pv_slice = hdulist_pv[0].data  # Assuming the PV slice is stored in the first HDU
hdulist_pv.close()

# Check the shape of the PV slice
print(f"PV slice shape: {pv_slice.shape}")  # Should print something like (128, 52)

# Define the position axis
position = np.linspace(0, 10, pv_slice.shape[1])  # Adjust as per your data's position range

# Calculate the mean and standard deviation (sigma) of the PV slice
mean_intensity = np.mean(pv_slice)
std_dev_intensity = np.std(pv_slice)

# Plotting the PV slice using contour plot
plt.figure(figsize=(8, 6))

# Contour plot for PV diagram (position vs. velocity)
cp = plt.contourf(position, velocities, pv_slice, levels=30, cmap='viridis')

# Add 1-sigma, 2-sigma, and 3-sigma contours (contour levels based on intensity values)
sigma_levels = [mean_intensity + 1*std_dev_intensity, 
                mean_intensity + 2*std_dev_intensity,
                mean_intensity + 3*std_dev_intensity]

# Overlay the sigma contours
contour_sigma = plt.contour(position, velocities, pv_slice, levels=sigma_levels, colors='white', linewidths=1.5)

# Add labels and title
plt.title("Position-Velocity (PV) Diagram with Sigma Contours")
plt.xlabel("Position (x or y)")  # Change based on your axes labeling
plt.ylabel("Velocity (km/s)")  # Change based on your velocity unit

# Add colorbar to indicate intensity
plt.colorbar(cp, label="Intensity")

# Display the plot
plt.show()
