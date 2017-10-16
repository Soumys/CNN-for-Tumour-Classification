# Data Preparation
The data consists of many 2D "slices," which, when combined, produce a 3-dimensional rendering of whatever was scanned. In this case, that's the chest cavity of the patient. 

The database consisting of CT scans of 1400 patients which is in the DICOM format was obtained from LIDC-IDRI (https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) , and then there is another file that contains the labels for this data. CT scans of 1200 patients is used for training and the remaining for testing. 

Each CT scan is 3-dimensional, containing multiple axial slices of the chest cavity. Preprocessing is performed for the standardization of the database. This includes conversion to Hounsfield Unit and Resampling the pixel spacing to 1mm 1mm 1mm. 

Segmentation is done on the 3D image by using image processing algorithms like thresholding, region growing and morphological operations to separate the lungs from the bones and residual tissues.

To run the code install the following dependencies
* numpy
* pandas
*dicom
4.os
5.matplotlib pyplot
6.OpenCV
7.Skimage
8.scipy

Run the commands
```
$ sudo apt-get install python-pip python-dev build-essential 
$ sudo pip install --upgrade pip 
$ sudo pip install --upgrade virtualenv 
```
