import os
from groq import Groq
import json

def generate_business_insights(emotion_counts: dict, keyword_freq: dict, fake_count: int, total: int) -> str:

    client = Groq(api_key="your_groq_key_here")

    prompt = f"""
You are a senior business intelligence analyst. Analyze this customer review data and write a professional insight report in 3-4 sentences.
Be specific, data-driven, and end with one clear actionable recommendation.

Data:
- Total reviews: {total}
- Genuine reviews: {total - fake_count}
- Flagged/fake reviews: {fake_count}
- Emotion breakdown: {json.dumps(emotion_counts)}
- Top keywords mentioned: {json.dumps(keyword_freq)}

Write like a real analyst report. No bullet points, just flowing professional sentences.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.7,
    )

    return response.choices[0].message.content