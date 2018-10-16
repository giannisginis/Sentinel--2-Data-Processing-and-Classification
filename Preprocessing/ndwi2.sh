#!/bin/bash
#author: Ioannis Gkinis
#first u need to make this script executable with this command #chmod +x ndwi2.sh
#This script calculates Normalized Difference Water Index by McFeeters
#usage: ./ndwi2.sh

echo "Calculate Sentinel2 NDWI by McFeeters"
echo ""
inp= $(`ls | grep '.*stacked_clipped*\.jp2'`)
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


output=`ls | grep '.*stacked_clipped*\.jp2'| awk -F "." '{print $1}'`
output+='_NDWI2.tif'
echo "* the output name is:"
echo $output
echo "* now apply gdal_calc: Command line raster calculator with numpy syntax"
gdal_calc.py -A $sds_names --A_band=2 -B $sds_names --B_band=7 --outfile=$output  --calc="(A.astype(float)-B)/(A.astype(float)+B)" --type='Float32'
echo "look at some histogram statistics"
gdalinfo -hist -stats $outfn
echo "<---------------------"
echo "* Finished"




