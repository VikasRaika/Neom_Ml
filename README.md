# Neom_Ml
ml-pipeline/
├─ app/
│  ├─ main.py                 # FastAPI app & endpoints
│  ├─ models.py               # Pydantic request/response models
│  ├─ services/
│  │  ├─ transcription.py     # audio → text
│  │  ├─ topics.py            # transcript → top-5 topics + summary
│  │  ├─ vectorize.py         # topics → topic vector; fusion with psychometrics
│  │  └─ match.py             # cosine similarity + interpretation
│  └─ utils/
│     └─ io.py                # load users JSON, helpers
├─ sample_data/
│  ├─ synthetic_users.json
│  └─ sample_audio.wav        # (your audio)
├─ requirements.txt
├─ README.md
└─ .env                       # (optional; keys if you use cloud APIs)
