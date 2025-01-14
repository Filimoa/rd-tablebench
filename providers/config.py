from pydantic import BaseModel
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()


class Settings(BaseModel):
    input_dir: Path
    output_dir: Path
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    anthropic_api_key: str | None = None


settings = Settings(
    input_dir=os.getenv("INPUT_DIR"),
    output_dir=os.getenv("OUTPUT_DIR"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    gemini_api_key=os.getenv("GEMINI_API_KEY"),
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
)
