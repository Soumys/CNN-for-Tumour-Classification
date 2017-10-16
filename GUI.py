import numpy as np
import pandas as pd
import matplotlib
import dicom
import os
import time
#import tensorflow as tf
import matplotlib.pyplot as plt
#import cv2
import math
import scipy.ndimage
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import Tkinter as tk
import ttk
import network__load as nl
matplotlib.use("TkAgg")

LARGE_FONT = ("Verdana", 12)

abouttxt = open("about.txt", 'r')
abttxt = abouttxt.read()

path_entry = '/home/prajwaljpj/Desktop/sample/'
patients = os.listdir(path_entry)
global patient_id
global x, y,IMG_SIZE_PX, SLICE_COUNT, keep_rate, n_classes, validation_data, prediction
global predict
patient_id = 1
global first_pass
first_pass = '...'
predict = 'benign'

#Back end code
#####SEGMENTATION#########

def load_scan(path):
    try:
        global first_pass
        first_pass = "Loading Scans"
        slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
        slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        for s in slices:
            s.SliceThickness = slice_thickness

        first_pass = "Done!"
        return slices
    except:
        first_pass = "Problem Loading Data... \nABORT"



def get_pixels_hu(slices):
    image = np.stack([each_slice.pixel_array for each_slice in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)
    #image = np.stack([cv2.resize(np.array(i), (50, 50)) for i in image])
    return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing=[3, 3, 3]):
    # Determine current pixel spacing
    try:
        global first_pass
        first_pass = 'Resampling...'
        spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor

        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
        first_pass = 'Done!'
        return image, new_spacing
    except:
        first_pass = 'Unable To Resample...\n ABORT'



def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    try:
        global first_pass
        first_pass= 'Segmenting...'
        binary_image = np.array(image > -320, dtype=np.int8) + 1
        labels = measure.label(binary_image)

        # Pick the pixel in the very corner to determine which label is air.
        #   Improvement: Pick multiple background labels from around the patient
        #   More resistant to "trays" on which the patient lays cutting the air
        #   around the person in half
        background_label = labels[0, 0, 0]

        # Fill the air around the person
        binary_image[background_label == labels] = 2

        # Method of filling the lung structures (that is superior to something like
        # morphological closing)
        if fill_lung_structures:
            # For every slice we determine the largest solid structure
            for i, axial_slice in enumerate(binary_image):
                axial_slice = axial_slice - 1
                labeling = measure.label(axial_slice)
                l_max = largest_label_volume(labeling, bg=0)

                if l_max is not None:  # This slice contains some lung
                    binary_image[i][labeling != l_max] = 1

        binary_image -= 1  # Make the image actual binary
        binary_image = 1 - binary_image  # Invert it, lungs are now 1

        # Remove other air pockets insided body
        labels = measure.label(binary_image, background=0)
        l_max = largest_label_volume(labels, bg=0)
        if l_max is not None:  # There are air pockets
            binary_image[labels != l_max] = 0
        first_pass = 'Done!'
        return binary_image
    except:
        first_pass = 'Problem in Segmentation...\n ABORT'


def plot_3d(image_1, image_2, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera

    try:
        print('Plotting...')
        global first_pass
        first_pass = 'Plotting...'
        p_1 = image_1.transpose(2, 1, 0)
        verts1, faces1 = measure.marching_cubes(p_1, threshold)


        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(121, projection='3d')

        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh1 = Poly3DCollection(verts1[faces1], alpha=0.70)
        face_color = [0.45, 0.45, 0.75]
        mesh1.set_facecolor(face_color)
        #return mesh
        ax1.add_collection3d(mesh1)
        ax1.set_xlim(0, p_1.shape[0])
        ax1.set_ylim(0, p_1.shape[1])
        ax1.set_zlim(0, p_1.shape[2])
        p_2 = image_2.transpose(2, 1, 0)

        verts2, faces2 = measure.marching_cubes(p_2, threshold)

        ax2 = fig.add_subplot(122, projection='3d')
        mesh2 = Poly3DCollection(verts2[faces2], alpha=0.70)
        mesh2.set_facecolor(face_color)

        ax2.add_collection3d(mesh2)

        ax2.set_xlim(0, p_2.shape[0])
        ax2.set_ylim(0, p_2.shape[1])
        ax2.set_zlim(0, p_2.shape[2])

        print('Done!')
        first_pass = 'Done!'
    except:
        print('Problem Plotting Graph...\n ABORT')
        first_pass = 'Problem Plotting Graph...\n ABORT'
    return fig
#######################################################################################################################
#######################################################################################################################

##Front end code

class SeaofBTCapp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.wm_title(self, "Cancer Analyzer")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (IntroPage, UploadPage, RunPage, AboutPage, PredictPage):
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(IntroPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class IntroPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Cancer Analyzer", font=('Veranda', 30))

        button = ttk.Button(self, text="Upload",
                            command=lambda: controller.show_frame(UploadPage))

        button2 = ttk.Button(self, text="Run",
                             command=lambda: controller.show_frame(RunPage))

        button3 = ttk.Button(self, text="About",
                             command=lambda: controller.show_frame(AboutPage))
        photo = tk.PhotoImage("hello.png")
        ph_label = tk.Label(self, image=photo)

        global first_pass
        status = tk.Label(self, textvariable=first_pass)
        status.grid(row=3, columnspan=3, sticky=tk.E)
        label.grid(row=0, columnspan=3, pady=50, padx=450)
        ph_label.grid(row=2, columnspan=5, pady=10, padx=10)
        button.grid(row=2, column=0, pady=20, padx=20)
        button2.grid(row=2, column=1, pady=20, padx=15)
        button3.grid(row=2, column=2, pady=20, padx=15)


class UploadPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Patient ID:", font=LARGE_FONT)
        self.entry = tk.Entry(self)

        def printcommand():
            global patient_id
            patient_id = int(self.entry.get())
            print(patient_id)

        button1 = ttk.Button(self, text="Back",
                             command=lambda: controller.show_frame(IntroPage))


        button2 = ttk.Button(self, text="OK",
                             command=printcommand)

        button3 = ttk.Button(self, text="Run",
                             command=lambda: controller.show_frame(RunPage))
        global first_pass
        status = tk.Label(self, textvariable=first_pass)
        status.grid(row=3, columnspan=10, sticky=tk.E)
        label.grid(row=3, column=4, columnspan=2, pady=50, padx=450)
        self.entry.grid(row=3, column=3, columnspan=10, pady=50, padx=450)
        button1.grid(row=6, column=3, columnspan=2, pady=20, padx=20)
        button2.grid(row=6, column=6, columnspan=2, pady=20, padx=20)
        button3.grid(row=6, column=9, columnspan=2, pady=20, padx=20)




    #def but2(self):
     #   self.on_button(self)
      #  lambda: controller.show_frame(RunPage)



class RunPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="PLOT", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back",
                             command=lambda: controller.show_frame(IntroPage))
        button1.pack()

        button2 = ttk.Button(self, text="Render",
                             command=self.RunPage_imgs)
        button3 = ttk.Button(self, text="Prediction",
                             command=lambda: controller.show_frame(PredictPage))
        button3.pack(side=tk.BOTTOM)
        button2.pack(side=tk.BOTTOM)

    def RunPage_imgs(self):
        global patient_id
        patient_data = load_scan(path_entry + patients[patient_id])

        patient_pixels = get_pixels_hu(patient_data)

        pix_resampled, spacing = resample(patient_pixels, patient_data, [1, 1, 1])

        segmented_lungs = segment_lung_mask(pix_resampled, False)
        segmented_lungs_fill = segment_lung_mask(pix_resampled, True)

        img_data = segmented_lungs_fill - segmented_lungs
        segmented_lungs_1 = plot_3d(segmented_lungs, img_data, 0)

        # seg_lung_2 = plot_3d(segmented_lungs_fill, 0)
        global first_pass
        status = tk.Label(self, textvariable=first_pass)
        status.pack(side=tk.BOTTOM, fill=tk.X)

        # f = Figure(figsize=(5, 5), dpi=100)
        # a = f.add_subplot(111)

        canvas = FigureCanvasTkAgg(segmented_lungs_1, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


class AboutPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text=abttxt, font=LARGE_FONT)
        label.pack(pady=150, padx=30)

        button1 = ttk.Button(self, text="Home",
                             command=lambda: controller.show_frame(IntroPage))
        button1.pack(side=tk.BOTTOM)


class PredictPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Prediction", font=LARGE_FONT)
        label.pack(pady=10, padx=10)
        global prediction, patient_id
        button1 = ttk.Button(self, text="Home",
                             command=lambda: controller.show_frame(IntroPage))

        global validation_data, patient_id
        x = 'data'
        val = nl.test_neural_network(x, patient_id)
        val = 'The tumor is ' + val
        label = tk.Label(self, text=val, font=LARGE_FONT)
        label.pack(pady=10, padx=10)
        button1.pack()


app = SeaofBTCapp()
app.geometry("1280x720")
app.mainloop()
