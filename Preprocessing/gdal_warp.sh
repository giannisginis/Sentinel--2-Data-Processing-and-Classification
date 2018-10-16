#!/bin/bash

#author: Ioannis Gkinis
#first u need to make this script executable with this command #chmod +x gdal_warp.sh
#This script performs resampling of the bands with 20 m to 10m resolution using a bilinear method, by using GDAL
#Place this script inside the folder of the Sentinel-2 bands with 20m resolution
#usage: ./gdal_warp.sh

for kep in *.jp2;do # You can change the file extension with any valid file extension
    gdalwarp -overwrite -s_srs EPSG:32634 -t_srs EPSG:32634 -r bilinear -ts 10980 10980 -of HFA $kep ${kep/.jp2}_10m.tif;done
