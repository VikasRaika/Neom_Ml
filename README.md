# Psychometric + Topic Embedding Matcher

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

## Setup & Installation

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
rename .env.example to .env
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

  "transcript": "Hey, imagine a software that doesn't need you. It can think, plan and take action on its own. Sounds futuristic right? Well I welcome to the world of AI agents. Sounds futuristic right? I welcome you to the world of AI agents. My name is Vikas and I will help you understand what AI agents are, why they are trending and how they are changing the world. AI agents are nothing but advanced software programs that can do complex tasks on their own. Unlike traditional softwares which needs your input. So why they are trending? Companies everywhere see AI agents as big time savers instead of needing a big team to do small routine tasks. So now is the right time for you to learn and work with AI agents. Today, you might be using a phone or a website to order food but soon AI agents will find you the best restaurants with best deals, apply coupon, make the order, schedule delivery without you lifting a finger. Without you lifting a finger. AI agents like Replete, Devon AI can make your software application in no time and the number of new AI agents is increasing every day. Product predict by end of this year 25% of the companies who use generative AI will rely on such AI agents for their daily operations. And by the year 2027 this number might jump to 50%. Industries like how AI agents can simplify and speed up the work. What Zuckerberg says there will be more AI agents than people but not to replace us but to empower us. So like share and comment on this video and follow the trend screen for more such updates for more such updates on tech and AI."
,
  "top_k": 5
}
```

**Returns:**
```json
{
  "topics": [
    "AI Agents Technology",
    "Autonomous Software Capabilities",
    "Business Automation Trends",
    "Industry Adoption Predictions",
    "Future Workplace Transformation"
  ],
  "summary": "The video introduces AI agents as advanced software programs that can perform complex tasks autonomously without human input, representing a significant shift from traditional software that requires constant user interaction. Companies are increasingly adopting AI agents to save time and reduce the need for large teams to handle routine tasks, with predictions suggesting 25% of generative AI companies will rely on them by year-end and 50% by 2027. The technology promises to transform daily activities like food ordering and software development, with leaders like Zuckerberg envisioning AI agents as tools to empower rather than replace humans.",
  "topic_embedding": [
    -0.03390064835548401,
    -0.05011272057890892,
    -0.03252394124865532,
    -0.052154291421175,....]
```

### Step 3: Match Two Users

```bash
POST /match
{
  "topics": [
    "AI Agents Technology",
    "Autonomous Software Capabilities",
    "Business Automation Trends",
    "Industry Adoption Predictions",
    "Future Workplace Transformation"
  ],
  "topic_scale_user1": 0.4,
  "topic_scale_user2": 0.8
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




