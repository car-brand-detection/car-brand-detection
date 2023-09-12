# !pip install MoviePy
from moviepy import editor

def convert_avi_to_mp4(input_file, output_file):
    # Load the video clip
    video = editor.VideoFileClip(input_file)

    # Export the video as .mp4
    video.write_videofile(output_file)


convert_avi_to_mp4('../final/Efficient_b2/ArcFace/PROCESSED_TR-D2224WDZIR7 2_20230911-083755--20230911-090755.avi', '../final/Efficient_b2/ArcFace/hueta.mp4')
