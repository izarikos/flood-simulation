# import generic packages
import numpy as np
from click import pause
from matplotlib import pyplot as plt

# import geospatial packages
import rasterio
from rasterio.plot import show
from shapely.geometry import LineString
import geopandas as gpd

# import landlab components
from landlab import RasterModelGrid, imshow_grid
from landlab.components.overland_flow import OverlandFlow

# Open raster image
rasterObj = rasterio.open('dem_small.tif')

#extract array from raster
elevArray = rasterObj.read(1)
plt.imshow(elevArray)

#create grid from raster attributes
nrows = rasterObj.height  # number of raster cells on each side, initially 150
ncols = rasterObj.width
dxy = (rasterObj.transform.a,-rasterObj.transform.e)  # side length of a raster model cell, or resolution [m], initially 50

# define a landlab raster
mg = RasterModelGrid(shape=(nrows, ncols),
                     xy_spacing=dxy,
                     #xy_of_lower_left=(rasterObj.bounds[0],rasterObj.bounds[1]))
                     xy_of_lower_left=(0,0))

# show number of rows, cols and resolution
print(nrows, ncols, dxy)

# create a dataset of zero values
zr = mg.add_zeros("topographic__elevation", at="node")

# apply cell elevation to defined arrray
zr += elevArray[::-1,:].ravel()

imshow_grid(mg, "topographic__elevation", shrink=0.5)

#set and apply and initial height
initialHeight = 0.1
depthArray = np.ones(elevArray.shape)*initialHeight
mg.at_node["surface_water__depth"] = depthArray
#define the flood objeds
of = OverlandFlow(mg, steep_slopes=True)

#list to store times
dtList = []
#Run once and store elapsed time
of.run_one_step()
dtList.append(of.dt)
print(of.dt)

# explore the output data and location

# model outputs
print(of.output_var_names)

# where this nodes are locates
print(of.var_loc("surface_water__depth"))

# show the water depth array
print(mg.at_node["surface_water__depth"])

# plot the resulting water depth for the first run
imshow_grid(mg, "surface_water__depth", shrink=0.5)

# run the model for 100 time steps
for i in range(100):
    of.run_one_step()
    dtList.append(of.dt)
    print("Time step: %.2f seconds. Elapsed time %.2f seconds"%(of.dt,sum(dtList)))

# plot the resulting water depth for the 101th run
imshow_grid(mg, "surface_water__depth", shrink=0.5)

#convert the resulting data to a numpy array
zArray = mg.at_node["surface_water__depth"].reshape((nrows,ncols))[::-1,:]
#plot the array
plt.imshow(zArray)

# Write the array to a new GeoTIFF file
with rasterio.open(
        'floodSurfaceDepth_v0.tif',
        'w',
        driver='GTiff',
        height=rasterObj.height,
        width=rasterObj.width,
        count=1,  # number of bands
        dtype=zArray.dtype,
        crs=rasterObj.crs,
        transform=rasterObj.transform
) as dst:
    dst.write(zArray, 1)  # write the data to the first band
# Show the resulting raster
floodObj = rasterio.open('floodSurfaceDepth_v0.tif')


