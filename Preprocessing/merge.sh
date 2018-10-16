#!/bin/bash

#author: Ioannis Gkinis
#first u need to make this script executable with this command #chmod +x merge.sh
#This script performs a layer stack of all .tif images inside the folder
#usage: ./merge.sh
##############################################

output=`ls *.tif| head -n 1 | awk -F "_B" '{print $1}'`
output+='_cropped_stacked.tif'
echo $output
echo ""
echo "<----------------------------------------"
sds_names=""

for kep in *cropped.tif;do # You can change the file extension with any valid file extension
    echo "$(basename "$kep")"
    sds=$(echo "$(basename "$kep")")
    sds_names="$sds_names $sds"
done

echo ""
echo "<----------------------------------------"
echo "All bandnames:"
echo $sds_names

# Create the stack
gdal_merge.py -n -9999 -a_nodata -9999 -separate -of HFA -o $output $sds_names 
# Query the new stack for metadata
gdalinfo $output
