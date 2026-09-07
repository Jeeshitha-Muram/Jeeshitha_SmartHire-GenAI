"""Paths, model names, and constants. Import from here instead of hard-coding."""

from pathlib import Path

# ---- Folders ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
JOBS_DIR = DATA_DIR / "jobs"                  # job dataset (CSV)
RESUMES_DIR = DATA_DIR / "resumes"            # sample resumes to test the parser
CAREER_NOTES_DIR = DATA_DIR / "career_notes"  # docs the AI mentor retrieves from
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"  # saved FAISS index lives here
JOBS_INDEX_DIR = VECTORSTORE_DIR / "jobs_faiss"       # FAISS index of job postings
NOTES_INDEX_DIR = VECTORSTORE_DIR / "notes_faiss"     # FAISS index of career notes

# ---- Job dataset file (what download_data.py places in data/jobs/) ----
JOBS_CSV = JOBS_DIR / "naukri_com-job_sample.csv"

# ---- Models (Gemini, same stack used across the course) ----
# The LLM used for parsing, CV suggestions, and the mentor's answers.
CHAT_MODEL = "gemini-3.5-flash-lite"
# The embedding model used for semantic job search and RAG retrieval.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

# The API key is read from the environment variable below (see .env.example).
# Never put the real key in code — keep it in .env, which .gitignore excludes.
API_KEY_ENV = "GOOGLE_API_KEY"

# ---- Chunking (same values taught in the RAG sessions) ----
CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks

# ---- Retrieval ----
TOP_N_JOBS = 5   # matching jobs returned by semantic search
TOP_K_NOTES = 3  # career-note chunks the mentor retrieves per question

# ---- Job columns we keep from the raw dataset ----
JOB_TEXT_COLUMNS = ["jobtitle", "skills", "jobdescription"]  # combined -> embedded text
