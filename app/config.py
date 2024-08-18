import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
    TEMP_DIR=os.getenv('TEMP_DIR')
    WHISPER_MODEL=os.getenv('WHISPER_MODEL')
    AUDIO_MODEL=os.getenv('AUDIO_MODEL')