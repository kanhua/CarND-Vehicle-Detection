from moviepy.editor import VideoFileClip

PROJECT_VIDEO_FILE = "./project_video.mp4"


def save_fp_frame_1():
    frame_t = [1.9, 27.48, 28.00]
    clip1 = VideoFileClip(PROJECT_VIDEO_FILE)

    for idx, t in enumerate(frame_t):
        clip1.save_frame("./problematic_images/frame_{}.jpg".format(t), t=t)

def save_fp_video_1():


    clip1 = VideoFileClip(PROJECT_VIDEO_FILE)

    clip2=clip1.subclip(26.5,29.9)

    clip2.write_videofile("test_video_2.mp4",audio=False)


if __name__ == "__main__":
    save_fp_frame_1()
    save_fp_video_1()
