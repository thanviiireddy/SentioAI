import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file if present
load_dotenv(BASE_DIR / ".env")

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Gemini Model Name (Free tier eligible model on Google AI Studio)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

# External Prompt File Location
PROMPT_FILE_PATH = BASE_DIR / "prompts" / "emotion_analysis_prompt.txt"

# DistilRoBERTa Model Name
NLP_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
