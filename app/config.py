# app/config.py
import os
from dotenv import load_dotenv

# load .env if present
load_dotenv()

USE_LLM_TOPICS = os.getenv("USE_LLM_TOPICS", "false").lower() in {"1", "true", "yes", "on"}
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
