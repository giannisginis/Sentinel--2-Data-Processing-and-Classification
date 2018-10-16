#!/bin/bash
#author: Ioannis Gkinis
#first u need to make this script executable with this command #chmod +x clipper.sh
#This script crops every image inside the folder with a specific based on a vector file, by using GDAL, and moves the cropped images to a specific folder location
#usage: ./clipper.sh
echo ""
echo "<----------------------------------------"
echo "Start Clipper"
for kep in *cropped_stacked.tif;do # You can change the file extension with any valid file extension
    echo "$(basename "$kep")"
    gdalwarp -q -cutline /home/john/noa_project/training/diss.shp -crop_to_cutline -tr 10.0 10.0 -of GTiff  $kep ${kep/.tif}_cropped.tif;done

echo ""
echo "<----------------------------------------"
echo "Start Moving files"
mv *cropped_stacked_cropped* /path_to_destination_folder/
