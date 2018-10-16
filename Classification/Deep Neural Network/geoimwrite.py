# -*- coding: utf-8 -*-

# geoimwrite
# Write image data with georeference

# 29/04/2014 - Version 1.0
# 01/06/2014 - Version 1.8 - Handle properly single band imagery
# 04/02/2015 - Version 2.0 - Added import gdal from osgeo
# 12/05/2015 - Version 2.4 - Added hack which works with uint8 arrays
# 16/12/2015 - Version 2.6 - Fixed import order

# Author: Aristidis D. Vaiopoulos.

# Usage:
# geoimwrite('example.tif', image_array, geoTransform, proj, 'GTiff')


import numpy as np
# Import gdal
try:
    from osgeo import gdal
    from osgeo.gdalconst import *
except ImportError:
    import gdal
    from gdalconst import *

# Other useful GDAL commands:
# driver = dataset.GetDriver()
# datatype = band.DataType

def geoimwrite(filename, imdata, geoTransform, proj, drv_name):
    # Get the image Driver by its short name
    driver = gdal.GetDriverByName(drv_name)
    # Get image dimensions from array
    cols = np.size(imdata,1)
    rows = np.size(imdata,0)
    dims = np.shape(imdata.shape)[0]
    if dims == 2:
        bands = 1
    elif dims == 3:
        bands = np.size(imdata,2)
    else:
        raise Exception('Error x01: Invalid image dimensions.')
    # Image datatype
    dt = imdata.dtype
    datatype = gdal.GetDataTypeByName( dt.name )
    # Prepare the output dataset
    if datatype == 0:
        # Unknown datatype, try to use uint8 code
        datatype = 1
    outDataset = driver.Create(filename , cols, rows, bands, datatype)
    # Set the Georeference first
    outDataset.SetGeoTransform(geoTransform)
    outDataset.SetProjection(proj)
    if bands == 1:
        outBand = outDataset.GetRasterBand(1)
        outBand.WriteArray( imdata, 0, 0)
    else:
        # Iterate the bands and write them to the file
        for b_idx in range(1,bands+1):
            outBand = outDataset.GetRasterBand(b_idx)
            outBand.WriteArray( imdata[:,:,b_idx-1], 0, 0)
    # Clear variables and close the file
    outBand = None
    outDataset = None
    