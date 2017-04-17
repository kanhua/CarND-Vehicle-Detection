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
        tt_img = np.zeros((1, *test_img.shape), dtype='uint8')
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
    new_heatmap = np.copy(heatmap)
    new_heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return new_heatmap


def save_heamap_image(save_file, heatmap, draw_img,
                      raw_heatmap, raw_img_with_box):
    fig = plt.figure()
    plt.subplot(224)
    plt.imshow(np.flip(draw_img, axis=2))
    plt.title('Car Positions')
    plt.subplot(223)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map (after threshold)')

    plt.subplot(221)
    plt.imshow(raw_heatmap, cmap='hot')
    plt.title("Raw Heat Map")

    plt.subplot(222)
    plt.imshow(np.flip(raw_img_with_box, axis=2))
    plt.title("Hot Windows")

    fig.tight_layout()
    fig.savefig(save_file)


def draw_labeled_bboxes(img, labels, ratio_bound=2.5):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        draw_box = True
        # Draw the box on the image

        # Discard the resulting box if its shape is out of spec
        if ratio_bound is not None:
            bbox_width = np.max(nonzerox) - np.min(nonzerox)
            bbox_height = np.max(nonzeroy) - np.min(nonzeroy)
            long_edge = max(bbox_height, bbox_width)
            short_edge = min(bbox_height, bbox_width)
            if float(long_edge / short_edge) > ratio_bound:
                draw_box = False

        if draw_box:
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


class VehicleIdentifier(object):
    def __init__(self, clf, heat_thres=2,
                 vis_filename_root=None, add_past=False,
                 window_sizes=[160, 128, 96, 64],
                 xy_over_lap=[(0.6, 0.6), (0.6, 0.6), (0.5, 0.5), (0.5, 0.5)],
                 y_start_stop=[[440, 680], [384, 554], [394, 520], [417, 550]]
                 ):
        self.clf = clf
        self.heat_thres = heat_thres
        self.window_sizes = window_sizes
        self.y_start_stop = y_start_stop
        self.xy_overlap = xy_over_lap
        self.prev_heatmap = []
        self.max_heatmap_num = 5
        self.add_past = add_past
        self.vis_filename_root = vis_filename_root

    def find_car_windows(self, image):
        # y_start_stop = [470, None]  # Min and max in y to search in slide_window()

        windows = []

        if self.vis_filename_root is not None:
            rows=np.ceil(len(self.window_sizes)/2).astype(np.int)
            fig,ax=plt.subplots(ncols=2, nrows=rows)

        for idx, ws in enumerate(self.window_sizes):

            add_windows = slide_window(image, x_start_stop=[None, None], y_start_stop=self.y_start_stop[idx],
                                       xy_window=(ws, ws), xy_overlap=self.xy_overlap[idx], sliding_window_file=None)

            if self.vis_filename_root is not None:
                img_with_box = draw_boxes(image, add_windows)
                img_with_box = draw_boxes(img_with_box, [add_windows[0]], color=(255, 0, 0))

                ax_index=np.unravel_index(idx,(rows,2))
                ax[ax_index].imshow(np.flip(img_with_box,axis=2))
                ax[ax_index].set_title("size:{}x{},overlap:({},{})".format(ws,ws,
                                                                                   self.xy_overlap[idx][0],
                                                                                   self.xy_overlap[idx][1]))

            windows += add_windows

        if self.vis_filename_root is not None:
            fig.savefig(self.vis_filename_root+"_sw.jpg")

        hot_windows = search_windows(image, windows, self.clf)

        draw_image = np.copy(image)

        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

        return window_img, hot_windows

    def gen_heat_map(self, image, hot_windows):
        prev_heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        init_heat = np.copy(prev_heat)
        # Add heat to each box in box list
        raw_heat = add_heat(init_heat, hot_windows)

        if self.add_past:
            if self.prev_heatmap:
                prev_heat += self.prev_heatmap[-1] * 0.5
                prev_heat += self.prev_heatmap[-2] * 0.3
                prev_heat += self.prev_heatmap[-3] * 0.2
            else:
                self.prev_heatmap = [np.copy(prev_heat) for i in range(self.max_heatmap_num)]

            all_heat = raw_heat * 0.5 + prev_heat * 0.5
        else:
            all_heat = raw_heat

        # Apply threshold to help remove false positives
        filtered_heat = apply_threshold(all_heat, self.heat_thres)

        # Update the queue of heatmap in the last few frames
        if self.add_past:
            self.prev_heatmap.pop(0)
            self.prev_heatmap.append(raw_heat)

        # Visualize the heatmap when displaying
        heatmap = np.clip(filtered_heat, 0, 20)

        raw_heatmap = np.clip(raw_heat, 0, 20)

        # Draw images with the hot_windows found in the beginning.
        raw_img_with_box = draw_boxes(image, hot_windows)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(image), labels)

        if self.vis_filename_root is not None:
            save_file = self.vis_filename_root + "_heatmap.jpg"
            save_heamap_image(save_file, heatmap, draw_img, raw_heatmap, raw_img_with_box)

        return heatmap, draw_img

    def process_image(self, image):
        _, windows = self.find_car_windows(image)

        _, drawn_output_img = self.gen_heat_map(image, windows)

        return drawn_output_img


if __name__ == "__main__":
    with open("final_clf.p", 'rb') as fp:
        pip_clf = pickle.load(fp)

    vif = VehicleIdentifier(pip_clf, heat_thres=0)
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
