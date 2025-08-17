# Neom_Ml
# 🚀 AI/ML Take-Home Project: Psychometric Matching on Conversation Topics

## Overview

This project implements a mini-ML pipeline that:
- Transcribes audio content using OpenAI Whisper
- Extracts & vectorizes discussion topics using KeyBERT + Sentence Transformers
- Combines topic vectors with synthetic user psychometric profiles
- Computes a compatibility score based on cosine similarity

---

## 📁 Project Structure

├── app/
│ ├── main.py # FastAPI app entry point
│ ├── transcription.py # Whisper-based audio transcription
│ ├── topics.py # Topic extraction and summarization
│ ├── vectorize.py # Topic embedding + fusion logic
│ ├── match.py # Matching logic based on vector fusion
│ └── models.py # Pydantic request/response schemas
├── sample_data/
│ ├── sample_audio.wav
│ └── synthetic_users.json
├── README.md
├── requirements.txt


---

## 📦 Loading and Using Sample Data

1. Place `sample_audio.wav` and `synthetic_users.json` inside `sample_data/`
2. Make sure Whisper and SentenceTransformers are installed
3. Launch the API and access via Swagger at: `http://127.0.0.1:8000/docs`

---

## 🧠 Architecture & Design Decisions

- **Transcription**: OpenAI Whisper for robustness on spoken language
- **Topic Extraction**: `keyBERT` + noun phrase filtering + semantic deduplication
- **Embedding**: SentenceTransformers (`all-MiniLM-L6-v2`) used for dense vectors
- **Fusion**: Psychometric vector (5D) + scaled topic embedding (384D) → 389D combined vector
- **Similarity**: Cosine similarity computed on fused vectors

---

## 📊 Topic Vectorization & Fusion

- Topics are converted to embeddings using `SentenceTransformer`
- Each user provides an interest weight (e.g., 0.8 vs 0.6)
- The topic embedding is scaled by user-specific interest
- Final vector: `[psychometric ⊕ (topic_embedding × interest_scale)]`

---

## 🤝 Matching Logic

- `cosine_similarity(user1_vector, user2_vector)`
- Similarity score → label: `low`, `medium`, `high`
- Thresholds: customizable if needed
- Edge cases: handled by fallback vector averaging and normalization

---

## ▶️ How to Run

```bash
# 1. Set up Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Launch FastAPI server
uvicorn app.main:app --reload

# 3. Visit Swagger UI
http://127.0.0.1:8000/docs
