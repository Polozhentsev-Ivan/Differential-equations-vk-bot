from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
LLM_MODEL_LIGHT: str = os.getenv("LLM_MODEL_LIGHT", "")
LLM_MODEL_HEAVY: str = os.getenv("LLM_MODEL_HEAVY", "")

VK_KEY: str = os.getenv("VK_KEY", "")
