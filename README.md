# Open Source Geoprocessing Methodologies of Sentinel-2 -- Data Processing and Classification
## Overview
Scripts of basic remote sensing methodologies using Unix and Python. This repository covers workflow to perform Sentinel-2 classification using machine learning and deep learning classifiers:

* **Preprocessing:** Unix scripts that perform cropping, stacking and resampling of Sentinel-2 images,
* **Spectral indices:** Unix scripts that calculate vegetation and water indices from Sentinel-2 images,
* **Classification:** Python scripts that perform Land Cover Classification based on Machine Learning and Deep Learning classifiers

## Script Files
_**Explanation of the files:**_

* merge.sh
*Staking all image files in target directory*
* clipper.sh
*Cropping all image file in target directory and copies them to a destination folder*
* gdal_warp.sh
*Resampling of all image files in target directory to 10m resolution*
* NDVI.sh
*Calculates Normalised Difference Vegetation Index*
* MNDWI.sh
*Calculates Modified Normalised Difference Water Index*
* NDWI.sh
*Calculates Normalised Difference Water Index By Gao*
* NDWI2.sh
*Calculates Normalised Difference Water Index By McFeeters*
* rasterization_vector_file.py
*Rasterization of a vector file*
* classification.py
*Performs image classification based on a Machine Learning Classifiers*
* RNN_keras_python.py
*Performs image classification based on a Deep Learning architecture*

