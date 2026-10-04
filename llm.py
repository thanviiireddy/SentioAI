import os
import logging
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, GEMINI_MODEL, PROMPT_FILE_PATH

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_prompt_template() -> str:
    """Reads the prompt template from the external prompt file."""
    if not PROMPT_FILE_PATH.exists():
        raise FileNotFoundError(f"Prompt file not found at: {PROMPT_FILE_PATH}")
    with open(PROMPT_FILE_PATH, "r", encoding="utf-8") as f:
        return f.read()

def analyze_emotion_with_llm(text: str, emotion: str, confidence: float) -> str:
    """
    Calls the Google Gemini LLM API to analyze emotional context based on original text
    and DistilRoBERTa model prediction.

    Args:
        text (str): Input text to analyze.
        emotion (str): Primary emotion detected by DistilRoBERTa.
        confidence (float): Confidence score (0.0 - 1.0 or percentage).

    Returns:
        str: Gemini generated contextual analysis response.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Please add your GEMINI_API_KEY to the .env file.")

    # 1. Load prompt template from external prompt file
    prompt_template = load_prompt_template()

    # Format confidence string
    conf_str = f"{confidence * 100:.2f}%" if confidence <= 1.0 else f"{confidence:.2f}%"

    # 2. Insert variables into prompt template
    formatted_prompt = prompt_template.format(
        text=text,
        emotion=emotion,
        confidence=conf_str
    )

    # 3. Initialize Google GenAI Client
    client = genai.Client(api_key=GEMINI_API_KEY)

    # 4. Request Gemini API completion
    logger.info(f"Sending prompt to Gemini API model ({GEMINI_MODEL})...")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=formatted_prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=600,
        )
    )

    if not response or not response.text:
        raise RuntimeError("Gemini API returned an empty or invalid response.")

    return response.text.strip()
