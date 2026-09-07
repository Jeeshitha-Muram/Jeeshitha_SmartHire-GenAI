import os
import json

from google import genai
from google.genai import types

from src import config
from src.generate.prompts import RESUME_PARSE_PROMPT



RESUME_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "skills": {
            "type": "array",
            "items": {"type": "string"}
        },
        "experience": {"type": "string"},
        "education": {"type": "string"},
        "target_role": {"type": "string"}
    },
    "required": [
        "name",
        "skills",
        "experience",
        "education",
        "target_role"
    ]
}





def create_client():
    api_key = os.getenv(config.API_KEY_ENV)

    if not api_key:
        raise ValueError(
            f"{config.API_KEY_ENV} not found."
        )

    return genai.Client(
        api_key=api_key
    )


def parse_resume(resume_text: str) -> dict:
    """
    Parse resume text using the prompt imported
    from src.generate.prompts.
    """

    if not resume_text or not resume_text.strip():
        raise ValueError(
            "Resume text cannot be empty."
        )

    prompt = RESUME_PARSE_PROMPT.replace(
         "{resume_text}",
          resume_text.strip()
)

    client = create_client()

    response = client.models.generate_content(
        model=config.CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESUME_SCHEMA,
            temperature=0.2,
            max_output_tokens=2048
        )
    )

    if not response.text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    try:
        return json.loads(response.text)

    except json.JSONDecodeError as error:
        raise ValueError(
            "Invalid JSON returned by Gemini."
        ) from error