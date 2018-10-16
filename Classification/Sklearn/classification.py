#!/usr/bin/env python
# coding: utf-8

"""
Satellite Image Classification: Random Forest - Gradient Boost - SVM
@author: Ioannis Gkinis
"""
# Import Python 3's print function and division
from __future__ import print_function, division

# Use the GDAL based functions to read and write the cropped satellite data
from geoimread import *
from geoimwrite import *

# ## Preparing the dataset
# #### Opening the images

# In[2]:


# Import Python 3's print function and division
#from __future__ import print_function, division

# Import GDAL, NumPy, and matplotlib
from osgeo import gdal, gdal_array
import numpy as np
import matplotlib.pyplot as plt

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()

# Read in our image and ROI image
img_ds = gdal.Open('path_to_sentinel_stacked_image', gdal.GA_ReadOnly)
roi_ds = gdal.Open('path_to_raster_with_training_samples', gdal.GA_ReadOnly)

img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
               gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
for b in range(img.shape[2]):
    img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()
    
roi = roi_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)

##or use the custom function geoimread
(image, geoTransform, proj, drv_name) = geoimread('path_to_sentinel_stacked_image')

# Display them
plt.subplot(121)
plt.imshow(img[:, :, 1], cmap='gray')
plt.title('MNDWI')

plt.subplot(122)
plt.imshow(roi, cmap=plt.cm.Spectral)
plt.title('ROI Training Data')

plt.show()


# #### Pairing Y with X
# Now that we have the image we want to classify (our X feature inputs), 
#and the ROI with the land cover labels (our Y labeled data), we need to pair them 
#up in NumPy arrays so we may feed them to the models:

# In[3]:


# Find how many non-zero entries we have -- i.e. how many training data samples?
n_samples = (roi > 0).sum()
print('We have {n} samples'.format(n=n_samples))

# What are our classification labels?
labels = np.unique(roi[roi > 0])
print('The training data include {n} classes: {classes}'.format(n=labels.size, 
                                                                classes=labels))
# We will need a "X" matrix containing our features, and a "y" array containing our labels
#     These will have n_samples rows

X = img[roi > 0, :]  # include 8th band, which is Fmask, for now
y = roi[roi > 0]

print('Our X matrix is sized: {sz}'.format(sz=X.shape))
print('Our y array is sized: {sz}'.format(sz=y.shape))

# Split the data up in train and test sets

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score, accuracy_score
from sklearn import metrics


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# ## Training the Models
# Now that we have our X matrix of feature inputs (the spectral bands) and our y array (the labels), we can train our model.

# In[4]:

from sklearn.ensemble import  RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm
# Initialize our models
rf = RandomForestClassifier( n_estimators = 500, criterion = 'gini', max_depth = 4, 
                                min_samples_split = 2, min_samples_leaf = 1, max_features = 'auto', 
                                bootstrap = True, oob_score = True, n_jobs = 1, random_state = None, verbose = True)  
gb = GradientBoostingClassifier(n_estimators = 300, min_samples_leaf = 1, min_samples_split = 4, max_depth = 4,    
                                    max_features = 'auto', learning_rate = 0.8, subsample = 1, random_state = None,         
                                    warm_start = True)
svm = svm.SVC(gamma=0.001)
# Fit our model to training data
rf = rf.fit(X_train, Y_train)
print ("Trained model :: ", rf)
gb = gb.fit(X_train, Y_train)
print ("Trained model :: ", gb)
svm = svm.fit(X_train, Y_train)
print ("Trained model :: ", svm)


# #### Random Forest diagnostics

# With our Random Forest model fit, we can check out the "Out-of-Bag" (OOB) prediction score:

# In[5]:

# ACCURACY ASSESSMENT
print('Our OOB prediction of accuracy is: {oob}%'.format(oob=rf.oob_score_ * 100))
##########  Random Forest ########
# Model evaluation with test data set 
# Prediction at test data set
Y_pred = rf.predict(X_test)
Y_pred_train=rf.predict(X_train)
# Train and Test Accuracy
print ("Train Accuracy :: ", accuracy_score(Y_train, Y_pred_train))
print ("Test Accuracy  :: ", accuracy_score( Y_test, Y_pred))

## Confusion matrix
print("Confussion matrix:\n%s" %
      metrics.confusion_matrix(Y_test, Y_pred))
      
# Precission and accuracy:
print("Classification report:\n%s" %
      metrics.classification_report(Y_test, Y_pred))
print("Classification accuracy: %f" %
      metrics.accuracy_score(Y_test, Y_pred))
      
### Confusion matrix with Pandas
import pandas as pd

# Setup a dataframe -- just like R
df = pd.DataFrame()
df['truth'] = Y_test
df['predict'] = rf.predict(X_test)

# Cross-tabulate predictions
print(pd.crosstab(df['truth'], df['predict'], margins=True))
      
      ##########  GradientBoostingClassifier ########
Y_pred = gb.predict(X_test)
Y_pred_train=gb.predict(X_train)
# Train and Test Accuracy
print ("Train Accuracy :: ", accuracy_score(Y_train, Y_pred_train))
print ("Test Accuracy  :: ", accuracy_score( Y_test, Y_pred))

## Confusion matrix
print("Confussion matrix:\n%s" %
      metrics.confusion_matrix(Y_test, Y_pred))

# Precission and accuracy:
print("Classification report:\n%s" %
      metrics.classification_report(Y_test, Y_pred))
print("Classification accuracy: %f" %
      metrics.accuracy_score(Y_test, Y_pred))
      
### Confusion matrix with Pandas
import pandas as pd

# Setup a dataframe -- just like R
df = pd.DataFrame()
df['truth'] = Y_test
df['predict'] = gb.predict(X_test)

# Cross-tabulate predictions
print(pd.crosstab(df['truth'], df['predict'], margins=True))

##########  Support Vector Machines ########
# Model evaluation with test data set 
# Prediction at test data set
Y_pred = svm.predict(X_test)
Y_pred_train=svm.predict(X_train)
# Train and Test Accuracy
print ("Train Accuracy :: ", accuracy_score(Y_train, Y_pred_train))
print ("Test Accuracy  :: ", accuracy_score( Y_test, Y_pred))

## Confusion matrix
print("Confussion matrix:\n%s" %
      metrics.confusion_matrix(Y_test, Y_pred))
      
# Precission and accuracy:
print("Classification report:\n%s" %
      metrics.classification_report(Y_test, Y_pred))
print("Classification accuracy: %f" %
      metrics.accuracy_score(Y_test, Y_pred))
      
### Confusion matrix with Pandas
import pandas as pd

# Setup a dataframe -- just like R
df = pd.DataFrame()
df['truth'] = Y_test
df['predict'] = svm.predict(X_test)

# Cross-tabulate predictions
print(pd.crosstab(df['truth'], df['predict'], margins=True))
      

# To help us get an idea of which spectral bands were important, we can look at the feature importance scores:

# In[6]:


bands = [1, 2, 3, 4, 5, 7, 6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]

for b, imp in zip(bands, rf.feature_importances_):
    print('Band {b} importance: {imp}'.format(b=b, imp=imp))
    
for b, imp in zip(bands, gb.feature_importances_):
    print('Band {b} importance: {imp}'.format(b=b, imp=imp))

for b, imp in zip(bands, svm.feature_importances_):
    print('Band {b} importance: {imp}'.format(b=b, imp=imp))


# In[8]:
# ## Predicting the rest of the image
# 
# With our Random Forest-Gradient Boost-SVM classifier fit, we can now proceed by trying to classify the entire image:

# Take our full image and reshape into long 2d array (nrow * ncol, nband) for classification
new_shape = (img.shape[0] * img.shape[1], img.shape[2])

img_as_array = img[:, :, :].reshape(new_shape)
print('Reshaped from {o} to {n}'.format(o=img.shape,
                                        n=img_as_array.shape))

# Now predict for each pixel (RF-GB)
class_prediction = rf.predict(img_as_array)
class_prediction_gb = gb.predict(img_as_array)
class_prediction_svm = svm.predict(img_as_array)

# Reshape our classification map
class_prediction = class_prediction.reshape(img[:, :, 0].shape)
class_prediction_gb = class_prediction_gb.reshape(img[:, :, 0].shape)
class_prediction_svm = class_prediction_svm.reshape(img[:, :, 0].shape)
# now export your classificaiton
#without projection
import skimage.io as io
io.imsave('class_prediction_rf874.tiff', class_prediction)
#with projection
geoimwrite('/path_to_output_folder/class_prediction_rf.tiff',class_prediction,geoTransform, proj, drv_name)
geoimwrite('/path_to_output_folder/class_prediction_gb.tiff',class_prediction_gb,geoTransform, proj, drv_name)
geoimwrite('/path_to_output_folder/class_prediction_svm.tiff',class_prediction_svm,geoTransform, proj, drv_name)
