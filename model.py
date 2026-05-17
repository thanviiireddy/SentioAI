from transformers import pipeline

emotion_model = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None
)

def detect_emotion(text):
    results = emotion_model(text)

    # handle nested output
    if isinstance(results, list) and isinstance(results[0], list):
        results = results[0]

    best = max(results, key=lambda x: x['score'])

    return best['label'], best['score'], results  # also return full results now
