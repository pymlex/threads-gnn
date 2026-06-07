import os
from pathlib import Path

from dotenv import load_dotenv


def load_dotenv_file() -> None:
    """Load environment variables from the project .env file."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path)


def get_env(name: str) -> str:
    """Return a required environment variable."""
    load_dotenv_file()
    return os.environ[name]
