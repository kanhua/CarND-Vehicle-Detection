import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pickle

from utils import slide_window,draw_boxes
from load_images import load_images



### TODO: Tweak these parameters and see how the results change.
color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 0  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 16  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [470, None]  # Min and max in y to search in slide_window()


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        tt_img=np.zeros((1,*test_img.shape))
        tt_img[0]=test_img
        # 5) Scale extracted features to be fed to classifier
        # 6) Predict using your classifier
        prediction = clf.predict(tt_img)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


with open("final_clf.p", 'rb') as fp:
    pip_clf=pickle.load(fp)

image = cv2.imread('./examples/bbox-example-image.jpg')
draw_image = np.copy(image)

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
# image = image.astype(np.float32)/255

windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                       xy_window=(96, 96), xy_overlap=(0.5, 0.5))

windows_2 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                       xy_window=(48, 48), xy_overlap=(0.5, 0.5))

#windows_3 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
#                         xy_window=(24, 24), xy_overlap=(0.5, 0.5))

hot_windows = search_windows(image, windows+windows_2, pip_clf, color_space=color_space,
                             spatial_size=spatial_size, hist_bins=hist_bins,
                             orient=orient, pix_per_cell=pix_per_cell,
                             cell_per_block=cell_per_block,
                             hog_channel=hog_channel, spatial_feat=spatial_feat,
                             hist_feat=hist_feat, hog_feat=hog_feat)

window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

plt.imshow(np.flip(window_img,axis=2))
plt.savefig("test_find_obj.png")

with open("bbox_pickle.p",'wb') as fp:
    pickle.dump(hot_windows,fp)