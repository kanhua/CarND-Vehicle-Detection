import cv2
from sklearn.svm import LinearSVC
import glob
import numpy as np
import pickle
import os

def load_images(cache_file="all_images.p"):

    if os.path.exists(cache_file):
        with open(cache_file,'rb') as fp:
            img_obj=pickle.load(fp)
        return img_obj['images'],img_obj['labels']
    else:
        images,y=read_image_from_files()
        with open(cache_file,'wb') as fp:
            img_obj={}
            img_obj['images']=images
            img_obj['labels']=y
            pickle.dump(img_obj,fp)
        return images,y


def read_image_from_files(small=False):
    car_image_files = glob.glob("/Users/kanhua/Downloads/carnd_dataset/vehicles/*/*.png")
    non_car_image_files = glob.glob("/Users/kanhua/Downloads/carnd_dataset/non-vehicles/*/*.png")

    car_images = []
    non_car_images = []
    for file in car_image_files:
        image = cv2.imread(file)
        car_images.append(image)

    for file in non_car_image_files:
        image = cv2.imread(file)
        non_car_images.append(image)

    car_images = np.array(car_images)
    non_car_images = np.array(non_car_images)
    all_images = np.concatenate((car_images, non_car_images), axis=0)

    y = np.concatenate((np.ones(car_images.shape[0]), np.zeros(non_car_images.shape[0])))

    return all_images, y


if __name__ == "__main__":
    X,y= load_images()
    print("total images: %s"%X.shape[0])
    print("total labels: %s"%y.shape[0])
