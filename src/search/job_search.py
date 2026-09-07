"""Module 2 — Semantic job search.

Embed every job description, store the vectors in a FAISS index
(saved under config.JOBS_INDEX_DIR), embed the candidate profile,
and run a top-N similarity search.

"""

from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from src import config
from src.search.embed import embed_text


# ============================================================
# 1. LOAD JOB DATA
# ============================================================

def load_jobs(job_csv: Path = None) -> pd.DataFrame:
    """Load the job dataset and prepare text for semantic search."""

    if job_csv is None:
        job_csv = config.JOBS_CSV

    if not job_csv.exists():
        raise FileNotFoundError(
            f"Job CSV file not found:\n{job_csv}"
        )

    jobs_df = pd.read_csv(
        job_csv,
        low_memory=False
    )

    # Required columns
    required_columns = {
        "jobtitle",
        "jobdescription"
    }

    missing_columns = (
        required_columns - set(jobs_df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Replace missing values
    jobs_df = jobs_df.fillna("")

    # Convert important columns to strings
    jobs_df["jobtitle"] = (
        jobs_df["jobtitle"].astype(str)
    )

    jobs_df["jobdescription"] = (
        jobs_df["jobdescription"].astype(str)
    )

    # Some datasets may not have a skills column
    if "skills" not in jobs_df.columns:
        jobs_df["skills"] = ""

    jobs_df["skills"] = (
        jobs_df["skills"].astype(str)
    )

    # --------------------------------------------------------
    # Combine job information for embedding
    # --------------------------------------------------------

    jobs_df["job_text"] = (
        "Job Title: "
        + jobs_df["jobtitle"]
        + "\nSkills: "
        + jobs_df["skills"]
        + "\nJob Description: "
        + jobs_df["jobdescription"]
    )

    return jobs_df


# ============================================================
# 2. LOAD SAVED FAISS JOB INDEX
# ============================================================

def load_job_index(
    index_dir: Path = None
):
    """Load the FAISS job index created in Notebook 02."""

    if index_dir is None:
        index_dir = config.JOBS_INDEX_DIR

    index_file = index_dir / "jobs.faiss"

    if not index_file.exists():
        raise FileNotFoundError(
            f"FAISS job index not found:\n{index_file}\n\n"
            "Run Notebook 02 first to create the job index."
        )

    index = faiss.read_index(
        str(index_file)
    )

    # Verify embedding dimension
    if index.d != config.EMBED_DIM:
        raise ValueError(
            f"FAISS embedding dimension mismatch.\n"
            f"Expected: {config.EMBED_DIM}\n"
            f"Found: {index.d}"
        )

    return index


# ============================================================
# 3. SEARCH JOBS
# ============================================================

def search_jobs(
    profile_text: str,
    jobs_df: pd.DataFrame,
    index,
    top_n: int = None
):
    """Search for jobs most similar to a candidate profile."""

    if not profile_text or not profile_text.strip():
        raise ValueError(
            "Candidate profile cannot be empty."
        )

    if top_n is None:
        top_n = config.TOP_N_JOBS

    if index.ntotal != len(jobs_df):
        raise ValueError(
            f"FAISS contains {index.ntotal} vectors, "
            f"but jobs_df contains {len(jobs_df)} jobs."
        )

    top_n = min(
        top_n,
        index.ntotal
    )

    # --------------------------------------------------------
    # Embed candidate profile
    # --------------------------------------------------------

    profile_embedding = embed_text(
        profile_text
    )

    profile_embedding = np.asarray(
        profile_embedding,
        dtype=np.float32
    )

    if profile_embedding.shape != (
        config.EMBED_DIM,
    ):
        raise ValueError(
            f"Candidate embedding dimension mismatch.\n"
            f"Expected: {config.EMBED_DIM}\n"
            f"Received: {profile_embedding.shape}"
        )

    profile_embedding = profile_embedding.reshape(
        1,
        -1
    )

    # --------------------------------------------------------
    # FAISS similarity search
    # --------------------------------------------------------

    scores, indices = index.search(
        profile_embedding,
        top_n
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        job = jobs_df.iloc[int(idx)]

        results.append({
            "jobid": job.get("jobid", ""),
            "jobtitle": job.get("jobtitle", ""),
            "skills": job.get("skills", ""),
            "similarity": float(score),
            "jobdescription": job.get(
                "jobdescription",
                ""
            )
        })

    return results
