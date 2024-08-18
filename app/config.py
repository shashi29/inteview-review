import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
    OPENAI_TEMPERATURE=os.getenv('OPENAI_TEMPERATURE')
    OPEANI_MODEL=os.getenv('OPEANI_MODEL')
    
    TEMP_DIR=os.getenv('TEMP_DIR')
    WHISPER_MODEL=os.getenv('WHISPER_MODEL')
    AUDIO_MODEL=os.getenv('AUDIO_MODEL')