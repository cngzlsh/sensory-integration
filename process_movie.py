# this file processes the movie captured by Unity Recorder to frames and audio
import cv2
from moviepy.editor import VideoFileClip
import os
import argparse
import pathlib

def split_video_into_frames(video_path, output_folder):

    # Initialize the video capture
    if isinstance(video_path, pathlib.Path):
        video_path = str(pathlib.PurePath(video_path))
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        # Read frame by frame
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there are no frames left

        # Save each frame as an image
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Total frames extracted: {frame_count}")

def extract_audio_in_stereo(video_path, output_audio_path):
    # Load the video file
    if isinstance(video_path, pathlib.Path):
        video_path = str(pathlib.PurePath(video_path))
    clip = VideoFileClip(video_path)
    
    # Extract audio
    audio = clip.audio
    audio.write_audiofile(output_audio_path, codec='mp3', ffmpeg_params=["-ac", "2"])  # Ensure 2 channels for stereo

    # Close the clip
    clip.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--movie_path', type=str)
    parser.add_argument('--save_folder', type=str, default='./data/processed/')
    args = parser.parse_args()
    
    movie_path = pathlib.Path(args.movie_path)
    save_folder = pathlib.Path(args.save_folder)
    movie_name = movie_path.stem

    if not os.path.exists(save_folder / movie_name):
        frames_folder = save_folder / movie_name / 'frames'
        frames_folder.mkdir(parents=True, exist_ok=False)
        audio_folder = save_folder / movie_name / 'audio'
        audio_folder.mkdir(parents=True, exist_ok=False)
    
    split_video_into_frames(movie_path, save_folder / movie_name / "frames/")
    extract_audio_in_stereo(movie_path, save_folder / movie_name / 'audio/audio.mp3')
    
