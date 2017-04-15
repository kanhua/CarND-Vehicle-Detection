from moviepy.editor import VideoFileClip
import cv2
import pickle

from full_process import VehicleIdentifier




with open("final_clf.p", 'rb') as fp:
    pip_clf = pickle.load(fp)


vif = VehicleIdentifier(pip_clf, heat_thres=1)


input_video_file="./project_video.mp4"
output_video_file="./project_video_output.mp4"



white_output = output_video_file
clip1 = VideoFileClip(input_video_file)
white_clip = clip1.fl_image(vif.process_image)  # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)