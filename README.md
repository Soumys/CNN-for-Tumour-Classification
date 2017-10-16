# Data Preparation
The data consists of many 2D "slices," which, when combined, produce a 3-dimensional rendering of whatever was scanned. In this case, that's the chest cavity of the patient. 

The database consisting of CT scans of 1400 patients which is in the DICOM format was obtained from LIDC-IDRI (https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) , and then there is another file that contains the labels for this data. CT scans of 1200 patients is used for training and the remaining for testing. 

Each CT scan is 3-dimensional, containing multiple axial slices of the chest cavity. Preprocessing is performed for the standardization of the database. This includes conversion to Hounsfield Unit and Resampling the pixel spacing to 1mm 1mm 1mm. 

Segmentation is done on the 3D image by using image processing algorithms like thresholding, region growing and morphological operations to separate the lungs from the bones and residual tissues.

To run the code install the following dependencies
* numpy
* pandas
* dicom
* matplotlib pyplot
* OpenCV
* Scikit image
* scipy

Run the commands

* Installing pip
```
$ sudo apt-get install python-pip python-dev build-essential 
$ sudo pip install --upgrade pip 
$ sudo pip install --upgrade virtualenv 
```
* Installing dependencies 
```
$ pip install numpy
$ pip install pandas
$ pip install pydicom
$ pip install matplotlib
$ pip install scipy
$ pip install scikit-image
```
* Installing Opencv using the instrcutions given in https://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html

The data preparation code outputs a numpy array containing the 3-d image along with its label for all the patients 

# Running the Convolutional Neural Network

The network.py code requires Tensorflow library dependencies. This can be installed using the instructions provided in https://www.tensorflow.org/install/
The numpy array is fed as the input to the CNN. The CNN contains two convolution layers immediately after the input layer, followed by a pooling layer, a dropout layer and a fully connected layer. Each convolution layer is in turn followed by a rectified linear output layer and the pooling layer is followed by a Dropout layer.

The network2_saving.py can save the network parameters after training and can restore them at any point.

# Graphical User Interface
One of the easiest ways to popularise the project for the ease of use of both doctors and laymen alike
is the addition of a GUI interface. Some of the reasons for using a GUI interface is:
* GUI allows easy implementation
* Allows the layman to visualize
* Allows the doctor to understand the presence of tumors
* Prediction of tumour as benign or malignant.

GUI is created using the Tkinter module (“Tk interface”) which is the standard Python interface to the Tk GUI toolkit. Both Tk and Tkinter are available on most Unix platforms, as well as on Windows systems.

# Ackowledment

* Kaggle Data Science Bowl https://www.kaggle.com/c/data-science-bowl-2017
* www.tensorflow.org

[[ https://github.com/nishalpereira/CNN-for-Tumour-Classification/blob/master/screenshot/gui.gif?raw=true | height = 100px ]]
