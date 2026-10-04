# SentioAI
> **Emotion-Aware Text Analyzer**

SentioAI is an advanced Natural Language Processing (NLP) application that combines a fine-tuned Transformer model (**DistilRoBERTa**) with Large Language Model (**Google Gemini LLM**) contextual reasoning to perform deep, nuanced emotion analysis of text.

---

## Overview

Traditional sentiment analysis tools classify text into simple categories like positive, negative, or neutral. SentioAI goes beyond basic sentiment classification by performing fine-grained emotion detection (such as *sadness*, *joy*, *anger*, *fear*, *surprise*, *disgust*) and augmenting it with an LLM-powered contextual interpretation layer.

By combining **DistilRoBERTa** for high-precision emotion classification with **Google Gemini 2.5 Flash** for contextual reasoning, SentioAI provides a complete breakdown of emotional cues, underlying context, and subtle secondary/mixed emotional signals.

---

## Problem Statement

Classifying emotions from text using traditional machine learning or simple LLM prompts alone presents key challenges:
1. **Classifier Limitation:** Standard emotion classification models (like RoBERTa or DistilRoBERTa) output probabilities and labels, but cannot explain *why* a text expresses a particular emotion or highlight specific contextual phrases.
2. **LLM Redundancy:** Relying solely on an LLM to produce raw emotion labels is computationally inefficient and subject to output variability.
3. **Complex / Mixed Emotions:** Sentences often contain subtle mixed signals (e.g., feeling exhausted after working all night, yet happy about finishing a project). A single label does not capture the full narrative context.

SentioAI solves these challenges using a **hybrid multi-stage architecture**.

---

## Objectives

- Develop a robust NLP pipeline using Hugging Face Transformers (`j-hartmann/emotion-english-distilroberta-base`).
- Integrate Google Gemini LLM via an external API to provide contextual reasoning without replacing the specialized classifier.
- Externalize prompt engineering into a modular prompt template (`prompts/emotion_analysis_prompt.txt`).
- Ensure complete security by loading API credentials from environment variables (`.env`).
- Build an interactive, production-ready Streamlit web application.
- Implement fail-safe API error handling so primary emotion classification remains operational even if LLM service is offline.

---

## System Architecture

The architecture separates classification from contextual interpretation into two distinct, complementary stages:

```
                    USER TEXT
                        |
                        v
             +---------------------+
             |   DistilRoBERTa     |
             | Emotion Classifier  |
             +----------+----------+
                        |
                 Emotion + Score
                        |
                        v
             +---------------------+
             |     GEMINI LLM      |
             | Contextual Analysis |
             +----------+----------+
                        |
                        v
              AI Interpretation
                        |
                        v
                 STREAMLIT UI
```

### Model Responsibilities:

- **DistilRoBERTa (`j-hartmann/emotion-english-distilroberta-base`):**
  - Detects the primary emotion (e.g., Sadness, Joy, Anger).
  - Computes exact probability confidence scores.
  - Operates locally via Hugging Face Transformers.

- **Google Gemini LLM (`gemini-2.5-flash`):**
  - Receives the original text, detected emotion label, and confidence score.
  - Explains the emotional context of the passage.
  - Identifies key emotional cues and phrases.
  - Highlights secondary or mixed emotional signals (e.g., frustration alongside sadness).
  - **Does NOT** override or alter the classifier's primary prediction.

---

## NLP Methodology

1. **Preprocessing & Tokenization:** Text is tokenized using the Byte-Pair Encoding (BPE) tokenizer aligned with DistilRoBERTa architecture.
2. **Transformer Inference:** The sequence is passed through 6 Transformer layers to generate contextual embeddings and logits across emotion classes.
3. **Probability Computation:** Softmax activation produces confidence scores for each emotion category.
4. **LLM Prompt Injection:** Top prediction metrics are injected into a structured prompt template.
5. **Generative Contextual Synthesis:** Google Gemini generates a concise 3-part structured breakdown (Interpretation, Cues, Secondary Signals).

---

## LLM Integration

SentioAI uses the current, official Google GenAI Python SDK (`google-genai`).

- **SDK Package:** `google-genai`
- **Model:** `gemini-2.5-flash` (Available via Google AI Studio's free tier)
- **Execution Flow:** `analyze_emotion_with_llm(text, emotion, confidence)` in `llm.py` constructs the API payload, sends a single structured request, and returns the LLM response.

---

## Prompt Engineering

All LLM instructions are externalized to ensure clean code architecture and ease of iteration:

- **File Path:** `prompts/emotion_analysis_prompt.txt`
- **Placeholders:** `{text}`, `{emotion}`, `{confidence}`

### Prompt Guidelines Enforced:
- Instructs Gemini to act as an NLP emotion analysis assistant.
- Strictly forbids overriding DistilRoBERTa's primary predicted emotion.
- Prohibits medical or psychological diagnosis.
- Enforces concise, structured output.

---

## Technologies Used

- **Language:** Python 3.10+
- **Frontend Framework:** Streamlit
- **NLP Model:** Hugging Face Transformers (`j-hartmann/emotion-english-distilroberta-base`)
- **Deep Learning Engine:** PyTorch
- **LLM API:** Google Gemini API (`google-genai` SDK)
- **Environment & Config:** `python-dotenv`
- **Data Visualization:** Plotly Express & Plotly Graph Objects

---

## Installation

### 1. Clone or Open Project
Navigate to the root directory of SentioAI:
```bash
cd SentioAI
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

### 3. Activate Virtual Environment
- **Windows (PowerShell / Command Prompt):**
  ```bash
  venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## API Key Setup

1. Visit [Google AI Studio](https://aistudio.google.com/) and create a **Free Gemini API Key**.
2. Copy `.env.example` to create `.env`:
   ```bash
   cp .env.example .env
   ```
3. Open `.env` in any text editor and paste your API key:
   ```env
   GEMINI_API_KEY=AIzaSy...YourActualGeminiApiKey
   GEMINI_MODEL=gemini-2.5-flash
   ```

> **Security Note:** The `.env` file is excluded from Git via `.gitignore`. Never commit actual API keys to GitHub.

---

## Running the Project

Start the Streamlit application:
```bash
streamlit run app.py
```

The web app will automatically open in your default browser at `http://localhost:8501`.

---

## Example Input & Output

### Input Text:
> *"I studied really hard for my exam but still failed. I feel completely useless."*

### --------------------------------
### NLP EMOTION RESULT (DistilRoBERTa)
### --------------------------------
- **Detected Emotion:** 😢 SADNESS
- **Confidence Score:** 91.40%

### --------------------------------
### AI CONTEXTUAL ANALYSIS (Google Gemini)
### --------------------------------
- **AI Interpretation:** The text expresses deep sadness and disappointment resulting from an unsuccessful academic outcome despite significant effort.
- **Emotional Cues:**
  - *"studied really hard"*
  - *"still failed"*
  - *"feel completely useless"*
- **Possible Secondary Emotions:** Contextual frustration and feelings of helplessness regarding the gap between effort and result.

---

## Security

- All API keys are loaded dynamically from environment variables using `python-dotenv`.
- Secret variables are strictly isolated from codebase logic (`app.py`, `llm.py`, `config.py`).
- `.env` and virtual environment folders (`venv/`) are listed in `.gitignore`.

---

## Limitations

- **Language Scope:** The current DistilRoBERTa pipeline is optimized for English text.
- **API Rate Limits:** Free-tier Gemini API calls are subject to rate limits (RPM/TPM) set by Google AI Studio.
- **Short Texts:** Extremely brief inputs (<3 words) may yield lower confidence scores.

---

## Future Improvements

- Multilingual emotion classification pipeline using XLM-RoBERTa.
- Real-time speech-to-text input processing.
- Historical emotion trend tracking with database persistence.
