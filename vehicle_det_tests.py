import unittest
from full_process import VehicleIdentifier
import pickle
import glob
import cv2


class MyTestCase(unittest.TestCase):

    def setUp(self):
        with open("final_clf.p", 'rb') as fp:
            self.pip_clf = pickle.load(fp)

        self.files=glob.glob("./test_images/*.jpg")

    def test_sw_output(self):

        vif = VehicleIdentifier(self.pip_clf, heat_thres=1,vis_filename_root="./output_images/test_sw")

        test_file="./test_images/test1.jpg"

        vif.process_image(cv2.imread(test_file))

    def test_prob_images(self):

        prob_files=glob.glob("./problematic_images/frame*.jpg")

        vif = VehicleIdentifier(self.pip_clf, heat_thres=1)

        for idx,f in enumerate(prob_files):
            image=cv2.imread(f)
            new_image=vif.process_image(image)

            cv2.imwrite("./output_images/pb_fp_%s.jpg" % idx, new_image)



    def test_images(self):

        vif = VehicleIdentifier(self.pip_clf, heat_thres=1)

        for idx,f in enumerate(self.files):
            image=cv2.imread(f)
            new_image=vif.process_image(image)

            cv2.imwrite("./output_images/test_fp_%s.jpg" % idx, new_image)



if __name__ == '__main__':
    unittest.main()
