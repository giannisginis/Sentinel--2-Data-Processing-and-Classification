#!/bin/bash
#author: Ioannis Gkinis
#first u need to make this script executable with this command #chmod +x ndvi.sh
#This script calculates NDVI
#usage: ./ndvi.sh

echo "Calculate Sentinel2 NDVI"
echo ""
inp= $(`ls | grep '.*stacked_clipped*\.jp2'`) # You can change the file extension with any valid file extension
echo "* the input file is"
echo $inp
echo ""
sds_names=""
for kep in *clipped.jp2;do
    #echo "$(basename "$kep")"
    sds=$(echo "$(basename "$kep")")
    sds_names="$sds_names $sds"
done
echo "* the input file is"
echo $sds_names


output=`ls | grep '.*stacked_clipped*\.jp2'| awk -F "." '{print $1}'` # You can change the file extension with any valid file extension
output+='_NDVI.tif'
echo "* the output name is:"
echo $output
echo "* now apply gdal_calc: Command line raster calculator with numpy syntax"
gdal_calc.py -A $sds_names --A_band=7 -B $sds_names --B_band=3  --outfile=$output  --calc="(A.astype(float)-B)/(A.astype(float)+B)" --type='Float32'
echo "look at some histogram statistics"
gdalinfo -hist -stats $outfn
echo "<---------------------"
echo "* Finished"

