import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DEFAULT_MODEL = "gemini-2.5-flash"
    API_KEY = os.getenv("GEMINI_API_KEY", "")
