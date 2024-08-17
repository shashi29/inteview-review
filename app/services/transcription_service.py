import os
import whisper
import speech_recognition as sr
from app.config import Settings

class TranscriptionService:
    @staticmethod
    def audio_to_text(audio_file_path: str) -> str:
        model = whisper.load_model(Settings.WHISPER_MODEL)
        result = model.transcribe(audio_file_path)
        os.remove(audio_file_path)
        return result["text"]
    
    def audio_to_text_gcp(audio_file_path):
        print("Audio file path", audio_file_path)
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)  # Correct method to use recognizer
            text = recognizer.recognize_google(audio_data)
        result = dict()
        result["text"] = text
        print("Result", result)
        return result