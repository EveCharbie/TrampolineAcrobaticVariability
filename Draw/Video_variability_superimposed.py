import moviepy.editor as mp
import os

home_path = "/home/lim/Documents/StageMathieu/Video_outcome/"

video_files = []

for root, dirs, files in os.walk(home_path):
    for file in files:
        file_path = os.path.join(root, file)
        video_files.append(file_path)

clips = [mp.VideoFileClip(video).set_opacity(0.5) for video in video_files]
max_duration = max(clip.duration for clip in clips)
clips = [clip.set_duration(max_duration) for clip in clips]
final_clip = mp.CompositeVideoClip(clips)

output_path = "/home/lim/Documents/StageMathieu/composite_video_outcome.mp4"
final_clip.write_videofile(output_path, codec='libx264')

