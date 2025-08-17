# 🧠 Psychometric + Topic Embedding Matcher

This is an end-to-end AI/ML pipeline that:

- Transcribes audio using Whisper
- Extracts topics and summary using Anthropic Claude (LLM)
- Falls back to KeyBERT + SpaCy + TF-IDF + BART when LLM fails
- Converts topics into vector space using SentenceTransformer
- Fuses topic embeddings with user psychometric profiles (from synthetic data)
- Computes user compatibility using cosine similarity

---

## 📁 Project Structure

```
Neom_Ml/
├── app/
│   ├── main.py              # FastAPI app entrypoint
│   ├── transcription.py     # Whisper-based audio transcription
│   ├── topics.py            # Topic extraction using Anthropic or fallback
│   ├── vectorize.py         # Topic embedding and fusion with psychometrics
│   ├── match.py             # Compatibility score logic
│   └── models.py            # Pydantic request/response models
├── sample_data/
│   ├── sample_audio.wav
│   └── synthetic_users.json
├── requirements.txt
├── README.md
└── .env                     # Add Anthropic API key here
```

---

## 🚀 Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/VikasRaika/Neom_Ml.git
cd Neom_Ml
```

### 2. Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install System Dependencies

**FFmpeg** (required by Whisper)

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows (via Chocolatey)
choco install ffmpeg
```

**SpaCy English Model**

```bash
python -m spacy download en_core_web_sm
```

### 4. Add Anthropic API Key (Optional but Recommended)

```bash
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=your-key-here
```

---

## ▶️ Running the App

```bash
uvicorn app.main:app --reload
```

Visit: http://127.0.0.1:8000/docs for interactive API documentation

---

## 🧪 Testing the Workflow

### Step 1: Transcribe Audio

```bash
POST /transcribe
{
  "file": "sample_audio.wav"
}
```

### Step 2: Extract Topics + Summary

```bash
POST /summarise
{
  "text": "<transcription text>"
}
```

**Returns:**
```json
{
  "topics": ["Mars Mission", "Autonomous Landing", "Space Technology"],
  "topic_weights": [0.25, 0.2, 0.15],
  "summary": "This interview focuses on..."
}
```

### Step 3: Match Two Users

```bash
POST /match
{
  "user1_id": "user_1",
  "user2_id": "user_2",
  "topics": ["AI Agents", "Future Workplace"],
  "topic_embedding": [...],
  "topic_scale_user1": 0.9,
  "topic_scale_user2": 0.6
}
```

**Returns:**
```json
{
  "score": 0.8237,
  "label": "high",
  "detail": "combined_dim=389"
}
```

---

## 🧠 Architecture Highlights

- **Transcription**: Whisper (base model) — fast and accurate
- **Topic Extraction**:
  - Primary: Anthropic Claude (`claude-sonnet-4-20250514`)
  - Fallback: KeyBERT + SpaCy noun chunking + BART summarizer
- **Embeddings**: `all-MiniLM-L6-v2` via SentenceTransformers
- **Fusion**:
  - Psychometrics = 5D vector
  - Topics = 384D vector scaled by interest
  - Final: 389D vector per user
- **Similarity**: Cosine similarity + label buckets

---

## 💡 Key Design Concepts

### 🔍 Topic Vectorization
- Topic phrases → embedded via MiniLM
- Optionally merged semantically (using cosine threshold)
- Scaled per-user via topic interest (`0.6` or `0.8`)

### 🔗 Vector Fusion

```python
fused_vector = concat(psychometric_vector, topic_embedding * scale)
```

Different scales per user provide **differentiated alignment** in similarity calculations.

### ⚖️ Matching Logic
- Cosine similarity between 389D fused vectors
- Score mapped to labels:
  - `> 0.8` → High compatibility
  - `0.6–0.8` → Medium compatibility
  - `< 0.6` → Low compatibility
- **Robust Error Handling**:
  - Missing topic vectors: fallback to psychometric-only matching
  - Zero-divisions: handled via `np.clip` & graceful fallbacks

---

## ✅ Creative Extensions

- Recommend **new friends** or **podcasts** based on topic resonance
- Real-time summarization + psychometric matching in meetings
- Chat UI integration (React/Vue frontend)
- **Audio-based matching**: Users "listen" to podcasts/interviews and receive compatibility recommendations
- **Interest-based conversational recommender**: Evolve matching into broader recommendation system

---

## 📈 Next Steps & Improvements

- **Enhanced Topic Models**: Train domain-specific topic extraction models
- **Multi-user Matching**: Support group dynamics and cluster-based compatibility
- **Dynamic Learning**: Implement adaptive matching weight optimization
- **Advanced Similarity**: Replace static cosine similarity with learned similarity functions
- **User Preference Vectors**: Add persistent, evolving topic preference tracking
- **Real-time UI**: Build responsive frontend for live conversation analysis

---

## 🛠️ Technical Stack

- **Backend**: FastAPI
- **Audio Processing**: OpenAI Whisper
- **NLP**: Anthropic Claude API, KeyBERT, SpaCy, BART
- **Embeddings**: SentenceTransformers (MiniLM)
- **Vector Operations**: NumPy, SciPy
- **Data Handling**: Pandas, Pydantic

---

## 📝 API Documentation

Once running, visit `/docs` for complete interactive API documentation with request/response schemas and testing interface.




