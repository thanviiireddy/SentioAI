import json
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, GEMINI_MODEL

def generate_business_insights(emotion_counts: dict, keyword_freq: dict, fake_count: int, total: int) -> str:
    """
    Generates a business intelligence insight report using the Gemini LLM API.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured in .env file.")

    prompt = f"""
You are a senior business intelligence analyst. Analyze this customer review emotion data and write a professional insight report in 3-4 sentences.
Be specific, data-driven, and end with one clear actionable recommendation.

Data:
- Total reviews: {total}
- Genuine reviews: {total - fake_count}
- Flagged/fake reviews: {fake_count}
- Emotion breakdown: {json.dumps(emotion_counts)}
- Top keywords mentioned: {json.dumps(keyword_freq)}

Write like a real analyst report. No bullet points, just flowing professional sentences.
"""

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4,
            max_output_tokens=350,
        )
    )

    if not response or not response.text:
        raise RuntimeError("Received empty response from Gemini API for business insights.")

    return response.text.strip()