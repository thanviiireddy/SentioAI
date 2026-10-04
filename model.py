from transformers import pipeline
from config import NLP_MODEL_NAME

# Pretrained Hugging Face Transformer model: j-hartmann/emotion-english-distilroberta-base
emotion_model = pipeline(
    "text-classification",
    model=NLP_MODEL_NAME,
    top_k=None
)

def detect_emotion(text: str):
    """
    Detects primary emotion and confidence score using DistilRoBERTa.

    Returns:
        tuple: (label, score, all_results_list)
    """
    if not text or not text.strip():
        return "neutral", 0.0, []

    results = emotion_model(text)

    # Handle nested output from pipeline
    if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
        results = results[0]

    best = max(results, key=lambda x: x['score'])

    return best['label'], best['score'], results
