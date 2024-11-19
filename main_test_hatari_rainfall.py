import numpy as np
from click import pause
from matplotlib import pyplot as plt
import rasterio
from rasterio.plot import show
from shapely.geometry import LineString
import geopandas as gpd
from landlab import RasterModelGrid, imshow_grid
from landlab.components.overland_flow import OverlandFlow

# Open raster image
rasterObj = rasterio.open('dem_small.tif')

# Extract array from raster
elevArray = rasterObj.read(1)
plt.imshow(elevArray)

# Create grid from raster attributes
nrows = rasterObj.height
ncols = rasterObj.width
dxy = (rasterObj.transform.a, -rasterObj.transform.e)
mg = RasterModelGrid(shape=(nrows, ncols), xy_spacing=dxy, xy_of_lower_left=(0, 0))

# Show number of rows, cols and resolution
print(nrows, ncols, dxy)

# Create a dataset of zero values
zr = mg.add_zeros("topographic__elevation", at="node")

# Apply cell elevation to defined array
zr += elevArray[::-1, :].ravel()
imshow_grid(mg, "topographic__elevation", shrink=0.5)

# Set and apply an initial height
initialHeight = 0.0
depthArray = np.ones(elevArray.shape) * initialHeight
mg.at_node["surface_water__depth"] = depthArray

# Define the overland flow object
of = OverlandFlow(mg, steep_slopes=True)

# Define rainfall rate (e.g., 10 mm/hr converted to meters per second)
rainfall_rate = 100 / 1000 / 3600  # mm/hr to m/s
print(f"Rainfall rate: {rainfall_rate} m/s")

# Define rainfall duration (e.g., 2 hours converted to seconds)
rainfall_duration = 3* 3600  # hours to seconds
print(f"Rainfall duration: {rainfall_duration} seconds")

# List to store times
dtList = []

# Run once and store elapsed time
of.run_one_step()
dtList.append(of.dt)
print(of.dt)

# Explore the output data and location
print(of.output_var_names)
print(of.var_loc("surface_water__depth"))
print(mg.at_node["surface_water__depth"])
imshow_grid(mg, "surface_water__depth", shrink=0.5)

# Run the model for 100 time steps with rainfall input
total_time_elapsed = 0
for i in range(100):
    of.run_one_step()
    dtList.append(of.dt)
    total_time_elapsed += of.dt

    # Add rainfall contribution if within rainfall duration
    if total_time_elapsed <= rainfall_duration:
        mg.at_node["surface_water__depth"] += rainfall_rate * of.dt

    print("Time step: %.2f seconds. Elapsed time %.2f seconds" % (of.dt, total_time_elapsed))

# Plot the resulting water depth for the 101st run
imshow_grid(mg, "surface_water__depth", shrink=0.5)

# Convert the resulting data to a numpy array
zArray = mg.at_node["surface_water__depth"].reshape((nrows, ncols))[::-1, :]
plt.imshow(zArray)

# Create the output file name with rainfall rate and duration
output_file_name = f"floodSurfaceDepth_{rainfall_rate * 3600 * 1000:.2f}mmhr_{rainfall_duration / 3600:.1f}hr.tif"

# Write the array to a new GeoTIFF file
with rasterio.open(
        output_file_name,
        'w',
        driver='GTiff',
        height=rasterObj.height,
        width=rasterObj.width,
        count=1,
        dtype=zArray.dtype,
        crs=rasterObj.crs,
        transform=rasterObj.transform
) as dst:
    dst.write(zArray, 1)

# Show the resulting raster
floodObj = rasterio.open(output_file_name)
