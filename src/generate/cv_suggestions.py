"""
Module 3 — CV improvement generator.

Given the parsed resume and a target job, this module uses the
CV_SUGGESTIONS_PROMPT from prompts.py to generate specific
resume improvement suggestions.
"""

import os

from google import genai

from src import config
from src.generate.prompts import CV_SUGGESTIONS_PROMPT


def create_client():
    api_key = os.getenv(config.API_KEY_ENV)

    if not api_key:
        raise ValueError(
            f"{config.API_KEY_ENV} is not set. "
            "Please add your Gemini API key to .env.example."
        )

    return genai.Client(
        api_key=api_key
    )


def generate_cv_suggestions(
    resume_text: str,
    job_text: str
) -> str:
    """
    Generate CV suggestions using the prompt
    imported from prompts.py.
    """

    if not resume_text or not resume_text.strip():
        raise ValueError(
            "Resume text cannot be empty."
        )

    if not job_text or not job_text.strip():
        raise ValueError(
            "Target job text cannot be empty."
        )

    prompt = CV_SUGGESTIONS_PROMPT.format(
        resume_text=resume_text.strip(),
        job_text=job_text.strip()
    )

    client = create_client()

    response = client.models.generate_content(
        model=config.CHAT_MODEL,
        contents=prompt
    )

    if not response.text:
        return (
            "No CV improvement suggestions "
            "were generated."
        )

    return response.text.strip()