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

from utils import slide_window, draw_boxes
from load_images import load_images
from scipy.ndimage.measurements import label


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        tt_img = np.zeros((1, *test_img.shape),dtype='uint8')
        tt_img[0] = test_img
        # 5) Scale extracted features to be fed to classifier
        # 6) Predict using your classifier
        prediction = clf.predict(tt_img)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


class VehicleIdentifier(object):
    def __init__(self, clf, heat_thres=2):
        self.clf = clf
        self.heat_thres = heat_thres
        self.window_sizes = [256,128,96,64]
        self.y_start_stop = [[484, 676],[384,554],[394,520],[417,550]]

    def find_car_windows(self, image):
        # y_start_stop = [470, None]  # Min and max in y to search in slide_window()

        windows = []

        for idx,ws in enumerate(self.window_sizes):
            add_windows = slide_window(image, x_start_stop=[None, None], y_start_stop=self.y_start_stop[idx],
                                       xy_window=(ws, ws), xy_overlap=(0.5, 0.5))

            windows += add_windows

        hot_windows = search_windows(image, windows, self.clf)

        draw_image = np.copy(image)

        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

        return window_img, hot_windows

    def gen_heat_map(self, image, hot_windows):
        heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        # Add heat to each box in box list
        heat = add_heat(heat, hot_windows)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, self.heat_thres)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(image), labels)

        return heatmap, draw_img

    def process_image(self, image):
        _, windows = self.find_car_windows(image)

        _, drawn_output_img = self.gen_heat_map(image, windows)

        return drawn_output_img


if __name__ == "__main__":
    with open("final_clf.p", 'rb') as fp:
        pip_clf = pickle.load(fp)

    vif = VehicleIdentifier(pip_clf, heat_thres=2)
    # image1 = cv2.imread('./examples/bbox-example-image.jpg')
    image1 = cv2.imread('./test_images/test1.jpg')

    window_img, hot_windows = vif.find_car_windows(image1)
    heatmap, draw_img = vif.gen_heat_map(image1, hot_windows)

    plt.imshow(np.flip(window_img, axis=2))
    plt.savefig("test_find_obj.png")
    plt.close()

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(np.flip(draw_img, axis=2))
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()
    fig.savefig("heatmap_search.png")

    # with open("bbox_pickle.p", 'wb') as fp:
    #    pickle.dump(hot_windows, fp)
