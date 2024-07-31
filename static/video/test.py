from transformers import pipeline
from moviepy.video.io.VideoFileClip import VideoFileClip
import json
import os

# Load the transcriber model
transcriber = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-small")

# Read the JSON file
with open('video2.json', 'r') as f:
    data = json.load(f)

# Load the video file
input_video = "video/video2.mp4"
video = VideoFileClip(input_video)
fps = video.fps

# Open output text file for writing
with open("output.txt", "w") as txt_file:
    for scene in data['scenes']:
        start_frame = scene['scene'][0]
        end_frame = scene['scene'][1]
        scene_id = scene['id']

        start_time = start_frame / fps
        end_time = end_frame / fps

        # Extract video clip for the scene
        clip = video.subclip(start_time, end_time)
        
        # Save the video clip to a file
        output_video = f"scene_{scene_id}.mp4"
        clip.write_videofile(output_video, codec="libx264", audio_codec="aac")

        # Extract audio from the clip
        # audio = clip.audio
        # output_audio = f"scene_{scene_id}.wav"
        # audio.write_audiofile(output_audio)

        # Transcribe the audio
        # text_content = transcriber(output_audio)['text']
        # print(text_content)
        
        # Write the transcription to the output text file
        # txt_file.write(f'"scene[{start_frame}, {end_frame}]": ')
        # txt_file.write(f'"{text_content}",\n')

        # Remove the audio file to save space
        # os.remove(output_audio)

# Close video file resources
video.reader.close()
video.audio.reader.close_proc()