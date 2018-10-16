# -*- coding: utf-8 -*-

# geoimread
# Read image data and georeference

# 29/04/2014 - Version 1.0
# 04/02/2015 - Version 1.2 - Added import gdal from osgeo
# 16/12/2015 - Version 1.4 - Fixed import order

# Author: Aristidis D. Vaiopoulos.

# Usage:
# (imdata, geoTransform, proj, drv_name) = geoimread('example.tif')


import numpy as np
# Import gdal
try:
    from osgeo import gdal
    from osgeo.gdalconst import *
except ImportError:
    import gdal
    from gdalconst import *

def geoimread(filename):
    # Open filename with ReadOnly access
    dataset = gdal.Open(filename, GA_ReadOnly)
    # Get the Driver name
    drv_name = dataset.GetDriver().ShortName
    ###dataset.GetDriver().LongName
    ###driver = dataset.GetDriver()
    # Get image dimensions
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    # Build a 3-D array where the 3rd dimension represent the bands
    for b_idx in range(1,bands+1):
        band = dataset.GetRasterBand(b_idx)
        if b_idx == 2:
            imdata = tdata
        # Read the numeric data of the band and put it in a temp array
        tdata = band.ReadAsArray(0, 0, cols, rows)
        # Stack the bands
        if b_idx >= 2:
            imdata = np.dstack( (imdata,tdata) )
    # In case we have only one band...
    if b_idx == 1:
        imdata = tdata
    
    # Get Georeference info
    geoTransform = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    # Clear variables and release the file
    dataset = None
    band = None
    
    return(imdata, geoTransform, proj, drv_name)
