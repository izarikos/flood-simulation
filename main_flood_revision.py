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

# Plot the shapefile and DEM
fig, ax = plt.subplots(figsize=(10, 10))
roi.plot(ax=ax, edgecolor='red', facecolor='none')
show(dem, ax=ax, transform=dem_transform, cmap='terrain')
plt.title('Region of Interest and DEM')
plt.show()

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

# Update the initial water height based on the lakes information
for idx, lake in lakes.iterrows():
    lake_geometry = lake.geometry
    z_mean = lake['z_mean']

    # Create a mask for the lake area
    lake_mask = rasterize([(lake_geometry, 1)], out_shape=dem.shape, transform=dem_transform, fill=0, all_touched=True, dtype=np.uint8).astype(bool)

    # Ensure lake_mask is correctly oriented
    lake_mask = np.flipud(lake_mask)

    # Set initial water height for the lake area
    initial_water_height[lake_mask] = z_mean



# Add the initial water height as a field to the grid
mg.add_field('initial_water_height', initial_water_height, at='node', clobber=True)

# Display the initial water height
plt.figure(figsize=(10, 10))
imshow_grid(mg, 'initial_water_height', shrink=0.5, at='node')
plt.title('Initial Water Height on RasterModelGrid')
plt.show()

# Display the initial grid elevation
plt.figure(figsize=(10, 10))
imshow_grid(mg, "topographic__elevation", shrink=0.5, at='node')
plt.title('DEM Elevation on RasterModelGrid')
plt.show()

# Parameters for the flood simulation
rainfall_intensity = 10  # mm/hr
rainfall_duration = 5  # hours

# Convert rainfall intensity to m/s
rainfall_intensity_m_per_s = (rainfall_intensity / 1000) / 3600
rainfall_flux = np.full(dem.shape, rainfall_intensity_m_per_s, dtype=np.float64)

# Add the rainfall flux as a field to the grid
mg.add_field('rainfall_flux', rainfall_flux, at='node', clobber=True)

# Initialize the surface water depth field to zero
dtList = []
surface_water_depth = np.zeros_like(dem_flipped)
mg.add_field('surface_water__depth', surface_water_depth, at='node', clobber=True)
print("Shape of initial_water_height: ", initial_water_height.shape)
# Load the runoff coefficient shapefile
runoff_path = '/Users/ioanniszarikos/Documents/ Flood simulation/Rhodes_floos_test/Hydrobasins/hybas_late_eu_lev01-12_vlv_GR regions_CN.shp'
runoff = gpd.read_file(runoff_path)

# Create a runoff coefficient array initialized to zeros
runoff_coefficient = np.zeros(dem.shape, dtype=np.float64)

# Update the runoff coefficient array based on the shapefile information
#runoff_coefficient[lake_mask] = 0
for idx, feature in runoff.iterrows():
    feature_geometry = feature.geometry
    coefficient = feature['CN']

    # Create a mask for the feature area
    feature_mask = rasterize([(feature_geometry, 1)], out_shape=dem.shape, transform=dem_transform, fill=0, all_touched=True, dtype=np.uint8).astype(bool)

    # Ensure feature_mask is correctly oriented
    feature_mask = np.flipud(feature_mask)

    # Set runoff coefficient for the feature area
    runoff_coefficient[feature_mask] = coefficient

# Remove the runoff coefficient within lake mask regions
total_runoff_coefficient = np.zeros(dem.shape, dtype=np.float64)

# Create a runoff coefficient array initialized to zeros
runoff_coefficient = np.zeros(dem.shape, dtype=np.float64)

# Load the lakes shapefile
lakes_path = 'lakes.shp'
lakes = gpd.read_file(lakes_path)

# Update the runoff coefficient based on lakes information
for idx, lake in lakes.iterrows():
    lake_geometry = lake.geometry
    z_mean = lake['z_mean']

    # Create a mask for the lake area
    lake_mask = rasterize([(lake_geometry, 1)], out_shape=dem.shape, transform=dem_transform, fill=0, all_touched=True, dtype=np.uint8).astype(bool)

    # Ensure lake_mask is correctly oriented
    lake_mask = np.flipud(lake_mask)

    # Set initial water height for the lake area
    initial_water_height[lake_mask] = z_mean

    # Remove the runoff coefficient within lake mask regions
    total_runoff_coefficient[lake_mask] = 0

 # Transform the runoff shapefile to a raster
 runoff_geom = [(geom, val / 100.0) for geom, val in zip(runoff.geometry, runoff['CN'])]
 runoff_raster = rasterize(runoff_geom, out_shape=dem.shape, transform=dem_transform, fill=0, dtype=np.float64)
 runoff_raster = np.flipud(runoff_raster)
 runoff_coefficient += runoff_raster
# Add the runoff coefficient to the grid
mg.add_field('runoff_coefficient', runoff_coefficient, at='node', clobber=True)
# Plot the updated runoff coefficient
runoff_coefficient = np.zeros(dem.shape, dtype=np.float64)
imshow_grid(mg, 'total_runoff_coefficient', shrink=0.5, at='node')
plt.title('Updated Runoff Coefficient on RasterModelGrid')

# Load the runoff coefficient shapefile
runoff_path = '/Users/ioanniszarikos/Documents/ Flood simulation/Rhodes_floos_test/Hydrobasins/hybas_late_eu_lev01-12_vlv_GR regions_CN.shp'
runoff = gpd.read_file(runoff_path)

# Update the runoff coefficient array based on the shapefile information
for idx, feature in runoff.iterrows():
    feature_geometry = feature.geometry
    coefficient = feature['CN']

    # Create a mask for the feature area
    feature_mask = rasterize([(feature_geometry, 1)], out_shape=dem.shape, transform=dem_transform, fill=0, all_touched=True, dtype=np.uint8).astype(bool)

    # Ensure feature_mask is correctly oriented
    feature_mask = np.flipud(feature_mask)

    # Set runoff coefficient for the feature area
    total_runoff_coefficient[feature_mask] = coefficient

# Add the updated runoff coefficient as a field to the grid
mg.add_field('total_runoff_coefficient', total_runoff_coefficient, at='node', clobber=True)
# Plot the updated runoff coefficient
plt.figure(figsize=(10, 10))
imshow_grid(mg, 'total_runoff_coefficient', shrink=0.5, at='node')
plt.title('Updated Runoff Coefficient on RasterModelGrid')
plt.show()
# Before initializing the OverlandFlow component, print the shapes of the arrays
print("Shape of initial_water_height: ", initial_water_height.shape)
print("Shape of rainfall_flux_flipped: ", rainfall_flux.shape)
print("Shape of mg.at_node['initial_water_height']: ", mg.at_node['initial_water_height'].shape)
print("Shape of mg.at_node['rainfall_flux']: ", mg.at_node['rainfall_flux'].shape)
print("Shape of mg.at_node['rainfall_flux']: ", mg.at_node['rainfall_flux'].shape)
# Time parameters for the simulation
total_time = rainfall_duration * 3600  # Total simulation time in seconds
time_step = 60  # Time step in seconds

# Before the OverlandFlow initialization
print("Shape of mg.at_node['initial_water_height']:", mg.at_node['initial_water_height'].shape)
print("Shape of mg.at_node['rainfall_flux']:", mg.at_node['rainfall_flux'].shape)

 # You can add print statements within the library code as well
 # For example, just before self._h_links += self._h_init
 # Note: Correct variable names and access

# Initialize the overland flow component
 # OverlandFlow already imported
of = OverlandFlow(mg, h_init='initial_water_height', rainfall_intensity='rainfall_flux')
of.after_one_step()  # Ensure correct method call
of.run_one_step()
dtList.append(of.dt)

print("Time step duration:", of.dt)
print(of.output_var_names)
print(of.var_loc("surface_water__depth"))
print(mg.at_node["surface_water__depth"])
imshow_grid(mg, "surface_water__depth", shrink=0.5)
plt.figure(figsize=(10, 10))

for i in range(100):
    of.run_one_step()
    dtList.append(of.dt)
    print("Time step: %.2f seconds. Elapsed time %.2f seconds" % (of.dt, sum(dtList)))

imshow_grid(mg, "surface_water__depth", shrink=0.5)
zArray = mg.at_node["surface_water__depth"].reshape((nrows, ncols))[::-1, :]
plt.imshow(zArray)

dem_path = 'dem_rhodes.tif'
with rasterio.open(dem_path) as rasterObj:
    with rasterio.open('floodSurfaceDepth_v0.tif', 'w', driver='GTiff', height=rasterObj.height, width=rasterObj.width, count=1, dtype=zArray.dtype, crs=rasterObj.crs, transform=rasterObj.transform) as dst:
        dst.write(zArray, 1)

floodObj = rasterio.open('floodSurfaceDepth_v0.tif')

plt.figure(figsize=(10, 10))
imshow_grid(mg, 'surface_water__depth', shrink=0.5, at='node')
plt.title('Simulated Water Depth after Rainfall Event')
plt.show()


