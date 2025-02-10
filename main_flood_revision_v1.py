import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.plot import show
import matplotlib.pyplot as plt
from landlab import RasterModelGrid, imshow_grid
from landlab.io import read_esri_ascii, write_esri_ascii

# Import statement for OverlandFlow moved here
from landlab.components import OverlandFlow
shape_path = 'Basin_rhodes.shp'
roi = gpd.read_file(shape_path)

# Load the DEM
dem_path = 'dem_rhodes.tif'
with rasterio.open(dem_path) as dem_dataset:
    dem = dem_dataset.read(1).astype(np.float64)  # Convert to float64 for processing
    dem_transform = dem_dataset.transform
    dem_crs = dem_dataset.crs
    dxy = dem_transform.a  # Grid cell size based on DEM transform (assuming square cells)

# Create a RasterModelGrid based on the DEM
nrows, ncols = dem.shape
mg = RasterModelGrid((nrows, ncols), xy_spacing=(dxy, dxy))
# Flip the DEM array along the vertical axis
dem_flipped = np.flipud(dem)
# Add the flipped DEM as a field to the grid
mg.add_field('topographic__elevation', dem_flipped, at='node', clobber=True)

# Load the lakes shapefile
lakes_path = 'lakes.shp'
lakes = gpd.read_file(lakes_path)
# Create an initial water height array initialized to zero
initial_water_height = np.zeros(dem.shape, dtype=np.float64)
# Rasterize the lakes shapefile
lakes_geom = [(geom, z_mean) for geom, z_mean in zip(lakes.geometry, lakes['z_mean'])]
lakes_raster = rasterize(lakes_geom, out_shape=dem.shape, transform=dem_transform, fill=0, dtype=np.float64)
lakes_raster = np.flipud(lakes_raster)
# Add the rasterized lakes as initial water height to the grid
initial_water_height += lakes_raster
mg.add_field('surface_water__depth', initial_water_height, at='node', clobber=True)

# Visualize the topographic elevation
plt.figure()
imshow_grid(mg, 'topographic__elevation', cmap='terrain')
plt.title('Topographic Elevation')
#plt.show()

# Visualize the initial water height
plt.figure()
imshow_grid(mg, 'surface_water__depth', cmap='Blues')
plt.title('Initial Water Height')
#plt.show()

# Load the runoff coefficient shapefile
runoff_path = '/Users/ioanniszarikos/Documents/ Flood simulation/Rhodes_floos_test/Hydrobasins/hybas_late_eu_lev01-12_vlv_GR regions_CN.shp'
runoff = gpd.read_file(runoff_path)

# Create a runoff coefficient array initialized to zeros
# Transform the runoff shapefile to a raster
runoff_geom = [(geom, val / 100.0) for geom, val in zip(runoff.geometry, runoff['CN'])]  # Convert from percentage to fraction
runoff_raster = rasterize(runoff_geom, out_shape=dem.shape, transform=dem_transform, fill=0, dtype=np.float64)
runoff_raster = np.flipud(runoff_raster)
runoff_coefficient = np.zeros(dem.shape, dtype=np.float64)
runoff_coefficient += runoff_raster

# Add the runoff coefficient to the grid
mg.add_field('runoff_coefficient', runoff_coefficient, at='node', clobber=True)

# Visualize the runoff coefficient
plt.figure()
imshow_grid(mg, 'runoff_coefficient', cmap='viridis')
plt.title('Runoff Coefficient')
#plt.show()

# Time and rainfall parameters for the simulation
rainfall_rate = 10.0  # Example value, in mm/hr
rainfall_duration = 5  # Duration in hours
total_time = rainfall_duration * 3600  # Total simulation time in seconds
time_step = 600 # Time step in seconds
rainfall_flux = np.full(mg.number_of_nodes, rainfall_rate / 3600.0)  # Convert rainfall_rate to mm/s
mg.add_field('rainfall_flux', rainfall_flux, at='node', clobber=True)

# Initialize the OverlandFlow component
of = OverlandFlow(mg, steep_slopes=True)

# Flood simulation
current_time = 0
while current_time < total_time:
    print(f"Time step: {current_time} seconds")
    runoff_rate = mg.at_node['rainfall_flux'] * mg.at_node['runoff_coefficient']
    mg.at_node['surface_water__depth'] += runoff_rate * time_step
    of.run_one_step(dt=time_step)
    current_time += time_step

    # Store the final surface water depth as a GeoTIFF in the output folder
    output_folder = 'output'
    output_path = f"{output_folder}/final_surface_water_depth_{rainfall_rate}mmhr_{rainfall_duration}hr_{time_step}s.tif"
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=mg.shape[0],
        width=mg.shape[1],
        count=1,
        dtype=mg.at_node['surface_water__depth'].dtype,
        crs=dem_crs,
        transform=dem_transform) as dst:
            dst.write(np.flipud(mg.at_node['surface_water__depth'].reshape(mg.shape[0], mg.shape[1])), 1)

# Visualize the final surface water depth
plt.figure()
imshow_grid(mg, 'surface_water__depth', cmap='Blues')
plt.title('Final Surface Water Depth')
plt.show()






