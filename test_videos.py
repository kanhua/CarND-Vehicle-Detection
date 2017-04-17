import unittest
import pickle
from full_process import VehicleIdentifier
from moviepy.editor import VideoFileClip


class MyTestCase(unittest.TestCase):
    def setUp(self):

        with open("final_clf.p", 'rb') as fp:
            pip_clf = pickle.load(fp)

        self.vif = VehicleIdentifier(pip_clf, heat_thres=1, add_past=True)

    def process_video(self,input_video_file,output_video_file):

        white_output = output_video_file
        clip1 = VideoFileClip(input_video_file)
        white_clip = clip1.fl_image(self.vif.process_image)  # NOTE: this function expects color images!!
        white_clip.write_videofile(white_output, audio=False)

    def test_video_1(self):
        input_video_file = "./test_video.mp4"
        output_video_file = "./output_videos/test_video_output.mp4"

        self.process_video(input_video_file,output_video_file)


    def test_video_2(self):
        input_video_file = "./test_video_2.mp4"
        output_video_file = "./output_videos/test_video_2_output.mp4"

        self.process_video(input_video_file,output_video_file)


if __name__ == '__main__':
    unittest.main()
