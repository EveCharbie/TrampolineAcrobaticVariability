import moviepy.editor as mp
import os


home_path = "/home/lim/Documents/StageMathieu/Video_AnSt/"


video_files = []

for root, dirs, files in os.walk(home_path):
    for file in files:
        file_path = os.path.join(root, file)
        video_files.append(file_path)



# Load all videos
clips = [mp.VideoFileClip(video).set_opacity(0.4) for video in video_files]

# Find the duration of the longest video
max_duration = max(clip.duration for clip in clips)

# Adjust all clips to the duration of the longest video
clips = [clip.set_duration(max_duration) for clip in clips]

# Overlay all clips on top of each other
final_clip = mp.CompositeVideoClip(clips)

# Write the result to a file
output_path = "/home/lim/Documents/StageMathieu/composite_video_AnSt.mp4"
final_clip.write_videofile(output_path, codec='libx264')

output_path
