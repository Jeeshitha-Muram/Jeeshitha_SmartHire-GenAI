"""Module 4 — AI Career Mentor (RAG).

A LangChain pipeline that answers career questions grounded in the career-notes
FAISS index (config.NOTES_INDEX_DIR): retrieve the top-K relevant chunks, stuff them
into the mentor prompt, and generate an answer that sticks to the documents. The
mentor must say "I don't know" when the answer is not in the notes. Prototype the
chain in notebook 03, then move it here.
"""
"""AI Career Mentor using RAG."""

import os
import json
from pathlib import Path

import faiss

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src import config
from src.search.embed import embed_text
from src.safety.guardrails import check_question, check_answer
from src.generate.prompts import MENTOR_SYSTEM_PROMPT


# ------------------------------------------------------------
# PROJECT ROOT AND ENVIRONMENT
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_FILE = PROJECT_ROOT / ".env.example"

load_dotenv(
    ENV_FILE,
    override=True
)


# ------------------------------------------------------------
# LOAD NOTES INDEX
# ------------------------------------------------------------

def load_notes_index():

    index_file = (
        config.NOTES_INDEX_DIR / "notes.faiss"
    )

    metadata_file = (
        config.NOTES_INDEX_DIR / "notes.json"
    )

    if not index_file.exists():
        raise FileNotFoundError(
            f"Notes FAISS index not found:\n{index_file}"
        )

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Notes metadata not found:\n{metadata_file}"
        )

    index = faiss.read_index(
        str(index_file)
    )

    with open(
        metadata_file,
        "r",
        encoding="utf-8"
    ) as file:
        notes = json.load(file)

    if index.ntotal != len(notes):
        raise ValueError(
            "FAISS index count does not match "
            "notes metadata count."
        )

    if index.d != config.EMBED_DIM:
        raise ValueError(
            f"Expected FAISS dimension "
            f"{config.EMBED_DIM}, "
            f"got {index.d}"
        )

    return index, notes


# ------------------------------------------------------------
# RETRIEVE NOTES
# ------------------------------------------------------------

def retrieve_notes(
    question: str,
    index,
    notes,
    top_k: int = None
):

    if top_k is None:
        top_k = config.TOP_K_NOTES

    query_vector = embed_text(
        question
    ).astype("float32")

    query_vector = query_vector.reshape(
        1,
        -1
    )

    scores, indices = index.search(
        query_vector,
        min(top_k, index.ntotal)
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        results.append({
            "score": float(score),
            "note": notes[int(idx)]
        })

    return results


# ------------------------------------------------------------
# CREATE LLM
# ------------------------------------------------------------

def create_llm():

    api_key = os.getenv(
        config.API_KEY_ENV
    )

    if not api_key:
        raise ValueError(
            f"{config.API_KEY_ENV} not found."
        )

    return ChatGoogleGenerativeAI(
        model=config.CHAT_MODEL,
        temperature=0
    )


# ------------------------------------------------------------
# CONVERT RESPONSE TO TEXT
# ------------------------------------------------------------

def extract_response_text(response):

    content = getattr(
        response,
        "content",
        response
    )

    # Normal string response
    if isinstance(content, str):
        return content.strip()

    # List response such as:
    # [{'type': 'text', 'text': '...'}]
    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):

                if item.get("type") == "text":

                    text = item.get(
                        "text",
                        ""
                    )

                    if text:
                        text_parts.append(
                            str(text)
                        )

            elif isinstance(item, str):

                text_parts.append(item)

        return "\n".join(
            part.strip()
            for part in text_parts
            if part.strip()
        )

    return str(content).strip()


# ------------------------------------------------------------
# ANSWER QUESTION
# ------------------------------------------------------------

def answer_question(
    question: str,
    index=None,
    note_texts=None,
    top_k=None
):

    allowed, message = check_question(
        question
    )

    if not allowed:
        return message

    if index is None or note_texts is None:
        index, note_texts = load_notes_index()

    retrieved = retrieve_notes(
        question,
        index,
        note_texts,
        top_k
    )

    if not retrieved:
        return (
            "I don't know based on the "
            "provided career notes."
        )

    context_parts = []

    for item in retrieved:

        note = item["note"]

        if isinstance(note, dict):

            text = (
                note.get("text")
                or note.get("content")
                or note.get("note")
                or ""
            )

        else:

            text = str(note)

        if text.strip():

            context_parts.append(
                text.strip()
            )

    context = "\n\n".join(
        context_parts
    )

    if not context:

        return (
            "I don't know based on the "
            "provided career notes."
        )

    prompt = MENTOR_SYSTEM_PROMPT.replace(
        "{context}",
        context
    ).replace(
        "{question}",
        question.strip()
    )

    llm = create_llm()

    response = llm.invoke(
        prompt
    )

    answer = extract_response_text(
        response
    )

    return check_answer(
        answer
    )


# ------------------------------------------------------------
# CAREER MENTOR
# ------------------------------------------------------------

def career_mentor(
    question: str
):

    return answer_question(
        question
    )