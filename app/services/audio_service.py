import os
import ffmpeg
from fastapi import UploadFile
from app.config import Settings

class AudioService:
    @staticmethod
    def extract_audio_from_video(video_file: UploadFile) -> str:
        temp_video_path = os.path.join(Settings.TEMP_DIR, "temp_video.mp4")
        temp_audio_path = os.path.join(Settings.TEMP_DIR, "temp_audio.wav")

        with open(temp_video_path, "wb") as f:
            f.write(video_file.file.read())

        stream = ffmpeg.input(temp_video_path)
        stream = ffmpeg.output(stream, temp_audio_path)
        ffmpeg.run(stream)
        return temp_audio_path
