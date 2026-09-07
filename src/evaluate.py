# ============================================================
# SMART HIRE - EVALUATION
# ============================================================

"""
SmartHire evaluation script.

Evaluates:
1. Job retrieval relevance
2. Mentor answer quality
3. Grounding
4. Helpfulness
5. Prompt comparison
6. Hallucination handling

Output:
reports/answer_quality.md
"""

# ============================================================
# 1. IMPORTS
# ============================================================

from pathlib import Path
import sys
import os
import json
import re

import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv


# ============================================================
# 2. FIND PROJECT ROOT
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

for folder in [CURRENT_FILE.parent] + list(CURRENT_FILE.parents):

    if (
        (folder / "src").is_dir()
        and (folder / "data").is_dir()
        and (folder / "notebooks").is_dir()
    ):
        PROJECT_ROOT = folder
        break

else:
    raise FileNotFoundError(
        "Could not find SmartHire project root."
    )


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 3. LOAD ENVIRONMENT
# ============================================================

ENV_FILE = PROJECT_ROOT / ".env.example"

if not ENV_FILE.exists():
    raise FileNotFoundError(
        f".env.example not found at:\n{ENV_FILE}"
    )

load_dotenv(
    ENV_FILE,
    override=True
)


# ============================================================
# 4. PROJECT IMPORTS
# ============================================================

from src import config

from src.search.embed import (
    embed_text
)

from src.generate.prompts import (
    MENTOR_SYSTEM_PROMPT
)

from src.safety.guardrails import (
    check_question,
    check_answer
)


# ============================================================
# 5. PATHS
# ============================================================

REPORTS_DIR = (
    PROJECT_ROOT / "reports"
)

REPORT_PATH = (
    REPORTS_DIR / "answer_quality.md"
)

JOBS_INDEX_PATH = (
    config.JOBS_INDEX_DIR / "jobs.faiss"
)

JOBS_METADATA_PATH = (
    config.JOBS_INDEX_DIR / "jobs.json"
)

NOTES_INDEX_PATH = (
    config.NOTES_INDEX_DIR / "notes.faiss"
)

NOTES_METADATA_PATH = (
    config.NOTES_INDEX_DIR / "notes.json"
)


# ============================================================
# 6. DISPLAY CONFIGURATION
# ============================================================

print("=" * 70)
print("SMART HIRE - EVALUATION")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nJob CSV:")
print(config.JOBS_CSV)

print("\nEmbedding model:")
print(config.EMBED_MODEL)

print("\nEmbedding dimension:")
print(config.EMBED_DIM)

print("\nChat model:")
print(config.CHAT_MODEL)

print("\nTop N jobs:")
print(config.TOP_N_JOBS)

print("\nTop K notes:")
print(config.TOP_K_NOTES)


# ============================================================
# 7. CHECK API KEY
# ============================================================

api_key = os.getenv(
    config.API_KEY_ENV
)

if not api_key:
    raise ValueError(
        f"{config.API_KEY_ENV} was not found.\n"
        "Please check your .env.example file."
    )

print("\nGemini API key found.")


# ============================================================
# 8. LOAD JOB DATASET
# ============================================================

def load_jobs_for_evaluation():
    """
    Load the job CSV directly.

    We intentionally do not import load_jobs()
    because the current loader.py does not provide
    a load_jobs() function.
    """

    if not config.JOBS_CSV.exists():
        raise FileNotFoundError(
            f"Job CSV not found:\n{config.JOBS_CSV}"
        )

    jobs_df = pd.read_csv(
        config.JOBS_CSV,
        low_memory=False
    )

    print(
        f"\nLoaded {len(jobs_df)} jobs."
    )

    print(
        "Columns:",
        jobs_df.columns.tolist()
    )

    required_columns = {
        "jobtitle",
        "jobdescription"
    }

    missing = (
        required_columns
        - set(jobs_df.columns)
    )

    if missing:
        raise ValueError(
            "Required job columns are missing: "
            f"{missing}"
        )

    jobs_df = jobs_df.fillna("")

    jobs_df["jobtitle"] = (
        jobs_df["jobtitle"]
        .astype(str)
    )

    jobs_df["jobdescription"] = (
        jobs_df["jobdescription"]
        .astype(str)
    )

    # The dataset may not contain skills.
    if "skills" not in jobs_df.columns:
        jobs_df["skills"] = ""

    jobs_df["skills"] = (
        jobs_df["skills"]
        .astype(str)
    )

    # This MUST match the text used when
    # creating the FAISS job index.
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
# 9. LOAD JOB FAISS INDEX
# ============================================================

def load_job_index():

    if not JOBS_INDEX_PATH.exists():

        raise FileNotFoundError(
            f"""
Job FAISS index was not found:

{JOBS_INDEX_PATH}

Run Notebook 02 first.
"""
        )

    index = faiss.read_index(
        str(JOBS_INDEX_PATH)
    )

    print(
        f"\nJob FAISS index loaded."
    )

    print(
        "Number of vectors:",
        index.ntotal
    )

    print(
        "Vector dimension:",
        index.d
    )

    if index.d != config.EMBED_DIM:

        raise ValueError(
            f"""
Job index dimension mismatch.

FAISS index dimension:
{index.d}

Current embedding dimension:
{config.EMBED_DIM}

Rebuild the job index using the current
embedding model.
"""
        )

    return index


# ============================================================
# 10. LOAD JOB METADATA
# ============================================================

def load_job_metadata():

    if not JOBS_METADATA_PATH.exists():

        raise FileNotFoundError(
            f"""
Job metadata not found:

{JOBS_METADATA_PATH}

Run Notebook 02 again.
"""
        )

    with open(
        JOBS_METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return data


# ============================================================
# 11. JOB RETRIEVAL FUNCTION
# ============================================================

def retrieve_jobs(
    profile_text,
    jobs_df,
    job_index,
    top_n=None
):
    """
    Perform semantic job search directly using
    the FAISS index.

    This avoids depending on a specific
    search_jobs() function signature.
    """

    if top_n is None:
        top_n = config.TOP_N_JOBS

    if not profile_text.strip():
        return []

    query_embedding = embed_text(
        profile_text
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    query_embedding = (
        query_embedding
        .reshape(1, -1)
    )

    if query_embedding.shape[1] != job_index.d:

        raise ValueError(
            f"""
Embedding dimension mismatch.

Query dimension:
{query_embedding.shape[1]}

FAISS index dimension:
{job_index.d}
"""
        )

    k = min(
        top_n,
        job_index.ntotal,
        len(jobs_df)
    )

    scores, indices = (
        job_index.search(
            query_embedding,
            k
        )
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        if idx >= len(jobs_df):
            continue

        row = jobs_df.iloc[
            int(idx)
        ].to_dict()

        row["similarity"] = float(
            score
        )

        results.append(row)

    return results


# ============================================================
# 12. RETRIEVAL TEST DATA
# ============================================================

RETRIEVAL_TESTS = [

    {
        "name": "Data Scientist Candidate",

        "profile": """
Skills: Python, SQL, Machine Learning,
Statistics, Data Analysis, Pandas, NumPy.

Experience: Experience working with datasets,
statistical analysis, data cleaning and
predictive models.

Education: Computer Science / Data Science.

Target Role: Data Scientist / Data Analyst.
""",

        "expected_keywords": [
            "data scientist",
            "data analyst",
            "data science",
            "machine learning"
        ]
    },

    {
        "name": "Software Engineer Candidate",

        "profile": """
Skills: Python, Java, Data Structures,
Algorithms, Object Oriented Programming, SQL.

Experience: Experience developing software
applications, solving programming problems
and working with software systems.

Education: Computer Science.

Target Role: Software Engineer /
Backend Developer.
""",

        "expected_keywords": [
            "software engineer",
            "software developer",
            "backend",
            "developer"
        ]
    },

    {
        "name": "Frontend Developer Candidate",

        "profile": """
Skills: HTML, CSS, JavaScript, React,
responsive web development.

Experience: Experience building frontend
web applications and user interfaces.

Target Role: Frontend Developer /
React Developer.
""",

        "expected_keywords": [
            "frontend",
            "react",
            "javascript",
            "web"
        ]
    },

    {
        "name": "Cloud Engineer Candidate",

        "profile": """
Skills: AWS, Docker, Kubernetes, Linux,
cloud infrastructure and DevOps.

Experience: Experience working with
cloud systems and deployment.

Target Role: Cloud Engineer /
DevOps Engineer.
""",

        "expected_keywords": [
            "cloud",
            "aws",
            "devops"
        ]
    }
]


# ============================================================
# 13. CHECK RETRIEVAL RELEVANCE
# ============================================================

def check_retrieval_relevance(
    retrieved_jobs,
    expected_keywords
):
    """
    Returns True if at least one retrieved job
    contains an expected keyword.
    """

    for job in retrieved_jobs:

        title = str(
            job.get(
                "jobtitle",
                ""
            )
        ).lower()

        skills = str(
            job.get(
                "skills",
                ""
            )
        ).lower()

        description = str(
            job.get(
                "jobdescription",
                ""
            )
        ).lower()

        combined_text = (
            title
            + " "
            + skills
            + " "
            + description
        )

        for keyword in expected_keywords:

            if keyword.lower() in combined_text:

                return True

    return False


# ============================================================
# 14. RETRIEVAL EVALUATION
# ============================================================

def evaluate_retrieval():

    print("\n")
    print("=" * 70)
    print("1. RETRIEVAL RELEVANCE")
    print("=" * 70)

    jobs_df = load_jobs_for_evaluation()

    job_index = load_job_index()

    # Verify that the index and CSV have
    # the same number of jobs.
    if job_index.ntotal != len(jobs_df):

        raise ValueError(
            f"""
Job index / CSV size mismatch.

CSV jobs:
{len(jobs_df)}

FAISS vectors:
{job_index.ntotal}

Run Notebook 02 again so both match.
"""
        )

    results = []

    hits = 0

    for test in RETRIEVAL_TESTS:

        print("\n" + "-" * 70)

        print(
            "Test:",
            test["name"]
        )

        print(
            "Profile:",
            test["profile"].strip()
        )

        retrieved_jobs = retrieve_jobs(
            test["profile"],
            jobs_df,
            job_index,
            config.TOP_N_JOBS
        )

        hit = check_retrieval_relevance(
            retrieved_jobs,
            test["expected_keywords"]
        )

        if hit:
            hits += 1

        print("\nRetrieved jobs:")

        for rank, job in enumerate(
            retrieved_jobs,
            start=1
        ):

            print(
                f"{rank}. "
                f"{job.get('jobtitle', 'Unknown')} "
                f"(similarity="
                f"{job.get('similarity', 0):.4f})"
            )

        print(
            "\nResult:",
            "HIT" if hit else "MISS"
        )

        results.append({
            "name": test["name"],
            "hit": hit,
            "jobs": retrieved_jobs
        })

    hit_rate = (
        hits / len(RETRIEVAL_TESTS)
        if RETRIEVAL_TESTS
        else 0.0
    )

    print("\n")
    print(
        f"Retrieval Hit Rate: "
        f"{hit_rate:.3f}"
    )

    print(
        f"Retrieval Hit Rate: "
        f"{hit_rate * 100:.1f}%"
    )

    return {
        "results": results,
        "hit_rate": hit_rate
    }


# ============================================================
# 15. LOAD NOTES INDEX
# ============================================================

def load_notes_index():

    if not NOTES_INDEX_PATH.exists():

        raise FileNotFoundError(
            f"""
Career notes FAISS index was not found:

{NOTES_INDEX_PATH}

Run Notebook 03 first.
"""
        )

    index = faiss.read_index(
        str(NOTES_INDEX_PATH)
    )

    print(
        "\nCareer notes index loaded."
    )

    print(
        "Vectors:",
        index.ntotal
    )

    print(
        "Dimension:",
        index.d
    )

    if index.d != config.EMBED_DIM:

        raise ValueError(
            f"""
Career notes index dimension mismatch.

Index:
{index.d}

Current embedding dimension:
{config.EMBED_DIM}

Rebuild Notebook 03 using the current
embedding model.
"""
        )

    return index


# ============================================================
# 16. LOAD NOTE TEXTS
# ============================================================

def load_note_texts():

    if not NOTES_METADATA_PATH.exists():

        raise FileNotFoundError(
            f"""
Career notes metadata was not found:

{NOTES_METADATA_PATH}

Run Notebook 03 first.
"""
        )

    with open(
        NOTES_METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    # Handle different possible formats.

    if isinstance(data, list):

        if all(
            isinstance(item, str)
            for item in data
        ):
            texts = data

        else:

            texts = []

            for item in data:

                if isinstance(item, dict):

                    if "text" in item:
                        texts.append(
                            str(item["text"])
                        )

                    elif "page_content" in item:
                        texts.append(
                            str(
                                item[
                                    "page_content"
                                ]
                            )
                        )

    elif isinstance(data, dict):

        if "texts" in data:

            texts = data["texts"]

        elif "notes" in data:

            texts = data["notes"]

        else:

            texts = []

            for value in data.values():

                if isinstance(value, str):
                    texts.append(value)

                elif isinstance(value, dict):

                    if "text" in value:
                        texts.append(
                            str(
                                value["text"]
                            )
                        )

                    elif "page_content" in value:
                        texts.append(
                            str(
                                value[
                                    "page_content"
                                ]
                            )
                        )

    else:

        texts = []

    texts = [
        str(text)
        for text in texts
        if text is not None
        and str(text).strip()
    ]

    if not texts:

        raise ValueError(
            "No career note texts were found."
        )

    return texts


# ============================================================
# 17. RETRIEVE CAREER NOTES
# ============================================================

def retrieve_notes(
    question,
    notes_index,
    note_texts,
    top_k=None
):

    if top_k is None:
        top_k = config.TOP_K_NOTES

    if not question.strip():
        return []

    query_embedding = embed_text(
        question
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    query_embedding = (
        query_embedding
        .reshape(1, -1)
    )

    if query_embedding.shape[1] != notes_index.d:

        raise ValueError(
            f"""
Notes embedding dimension mismatch.

Query:
{query_embedding.shape[1]}

Index:
{notes_index.d}
"""
        )

    k = min(
        top_k,
        notes_index.ntotal,
        len(note_texts)
    )

    scores, indices = (
        notes_index.search(
            query_embedding,
            k
        )
    )

    retrieved = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        if idx >= len(note_texts):
            continue

        retrieved.append({
            "text": note_texts[
                int(idx)
            ],
            "score": float(score),
            "index": int(idx)
        })

    return retrieved


# ============================================================
# 18. CREATE GEMINI CHAT MODEL
# ============================================================

def create_chat_model():

    try:

        from langchain_google_genai import (
            ChatGoogleGenerativeAI
        )

    except ImportError:

        raise ImportError(
            "langchain-google-genai is not installed."
        )

    model = ChatGoogleGenerativeAI(
        model=config.CHAT_MODEL,
        temperature=0
    )

    return model


# ============================================================
# 19. ANSWER MENTOR QUESTION
# ============================================================

def answer_question(
    question,
    notes_index,
    note_texts,
    chat_model
):

    # --------------------------------------------------------
    # Input guardrail
    # --------------------------------------------------------

    allowed, message = (
        check_question(question)
    )

    if not allowed:

        return {
            "answer": message,
            "sources": []
        }

    # --------------------------------------------------------
    # Retrieve notes
    # --------------------------------------------------------

    retrieved = retrieve_notes(
        question,
        notes_index,
        note_texts,
        config.TOP_K_NOTES
    )

    if not retrieved:

        return {
            "answer":
                "I don't know based on the provided career notes.",
            "sources": []
        }

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = "\n\n---\n\n".join(
        item["text"]
        for item in retrieved
    )

    # --------------------------------------------------------
    # Build prompt
    # --------------------------------------------------------

    prompt = MENTOR_SYSTEM_PROMPT.format(
        context=context,
        question=question
    )

    # --------------------------------------------------------
    # Call Gemini
    # --------------------------------------------------------

    response = chat_model.invoke(
        prompt
    )

    # --------------------------------------------------------
    # Extract response
    # --------------------------------------------------------

    if hasattr(
        response,
        "content"
    ):

        answer = response.content

    else:

        answer = str(response)

    if not isinstance(
        answer,
        str
    ):

        answer = str(answer)

    answer = answer.strip()

    # --------------------------------------------------------
    # Output guardrail
    # --------------------------------------------------------

    answer = check_answer(
        answer
    )

    return {
        "answer": answer,
        "sources": retrieved
    }


# ============================================================
# 20. MENTOR TESTS
# ============================================================

MENTOR_TESTS = [

    {
        "question":
            "What skills should I learn for a career in data science?",

        "type":
            "answer_quality"
    },

    {
        "question":
            "How can I improve my career based on the available career notes?",

        "type":
            "answer_quality"
    }
]


# ============================================================
# 21. ANSWER QUALITY CHECK
# ============================================================

def check_answer_quality(
    answer,
    sources,
    question
):

    if not answer:

        return {
            "correctness": 0.0,
            "grounding": 0.0,
            "helpfulness": 0.0
        }

    answer_lower = answer.lower()

    # --------------------------------------------------------
    # Refusal
    # --------------------------------------------------------

    refusal = (
        "i don't know based on the provided career notes"
        in answer_lower
    )

    # --------------------------------------------------------
    # Basic correctness indicator
    # --------------------------------------------------------

    correctness = 1.0

    if refusal:
        correctness = 0.5

    # --------------------------------------------------------
    # Grounding
    # --------------------------------------------------------

    context = " ".join(
        source["text"]
        for source in sources
    ).lower()

    answer_words = set(
        re.findall(
            r"\b[a-zA-Z]{4,}\b",
            answer_lower
        )
    )

    context_words = set(
        re.findall(
            r"\b[a-zA-Z]{4,}\b",
            context
        )
    )

    if not answer_words:

        grounding = 0.0

    else:

        grounding = (
            len(
                answer_words
                & context_words
            )
            / len(answer_words)
        )

    # --------------------------------------------------------
    # Helpfulness
    # --------------------------------------------------------

    if len(answer.strip()) >= 20:
        helpfulness = 1.0
    else:
        helpfulness = 0.5

    return {
        "correctness": float(
            correctness
        ),
        "grounding": float(
            grounding
        ),
        "helpfulness": float(
            helpfulness
        )
    }


# ============================================================
# 22. ANSWER QUALITY EVALUATION
# ============================================================

def evaluate_answer_quality():

    print("\n")
    print("=" * 70)
    print("2. ANSWER QUALITY")
    print("=" * 70)

    notes_index = load_notes_index()

    note_texts = load_note_texts()

    print(
        "Career note chunks:",
        len(note_texts)
    )

    chat_model = create_chat_model()

    results = []

    for test in MENTOR_TESTS:

        print("\n" + "-" * 70)

        print(
            "Question:",
            test["question"]
        )

        result = answer_question(
            test["question"],
            notes_index,
            note_texts,
            chat_model
        )

        answer = result["answer"]

        sources = result["sources"]

        print("\nAnswer:")
        print(answer)

        quality = check_answer_quality(
            answer,
            sources,
            test["question"]
        )

        print(
            "\nCorrectness:",
            f"{quality['correctness']:.3f}"
        )

        print(
            "Grounding:",
            f"{quality['grounding']:.3f}"
        )

        print(
            "Helpfulness:",
            f"{quality['helpfulness']:.3f}"
        )

        results.append({
            "question":
                test["question"],

            "answer":
                answer,

            "correctness":
                quality["correctness"],

            "grounding":
                quality["grounding"],

            "helpfulness":
                quality["helpfulness"],

            "sources":
                sources
        })

    if results:

        correctness = float(
            np.mean([
                item["correctness"]
                for item in results
            ])
        )

        grounding = float(
            np.mean([
                item["grounding"]
                for item in results
            ])
        )

        helpfulness = float(
            np.mean([
                item["helpfulness"]
                for item in results
            ])
        )

    else:

        correctness = 0.0
        grounding = 0.0
        helpfulness = 0.0

    overall = float(
        np.mean([
            correctness,
            grounding,
            helpfulness
        ])
    )

    print("\nOverall Answer Quality")

    print(
        "Correctness:",
        f"{correctness:.3f}"
    )

    print(
        "Grounding:",
        f"{grounding:.3f}"
    )

    print(
        "Helpfulness:",
        f"{helpfulness:.3f}"
    )

    print(
        "Overall:",
        f"{overall:.3f}"
    )

    return {
        "results": results,
        "correctness": correctness,
        "grounding": grounding,
        "helpfulness": helpfulness,
        "overall": overall
    }


# ============================================================
# 23. HALLUCINATION TEST
# ============================================================

HALLUCINATION_TEST = {

    "question":
        "What is the exact salary of a software engineer at Google?",

    "expected_refusal":
        "I don't know based on the provided career notes."
}


def evaluate_hallucination():

    print("\n")
    print("=" * 70)
    print("3. HALLUCINATION CHECK")
    print("=" * 70)

    notes_index = load_notes_index()

    note_texts = load_note_texts()

    chat_model = create_chat_model()

    result = answer_question(
        HALLUCINATION_TEST["question"],
        notes_index,
        note_texts,
        chat_model
    )

    answer = result["answer"]

    expected = (
        HALLUCINATION_TEST[
            "expected_refusal"
        ].lower()
    )

    passed = (
        expected in answer.lower()
    )

    print(
        "\nQuestion:"
    )

    print(
        HALLUCINATION_TEST[
            "question"
        ]
    )

    print(
        "\nAnswer:"
    )

    print(answer)

    print(
        "\nHallucination check:",
        "PASSED" if passed
        else "REVIEW NEEDED"
    )

    return {
        "question":
            HALLUCINATION_TEST[
                "question"
            ],

        "answer":
            answer,

        "passed":
            passed
    }


# ============================================================
# 24. PROMPT COMPARISON
# ============================================================

def prompt_comparison():

    print("\n")
    print("=" * 70)
    print("4. PROMPT COMPARISON")
    print("=" * 70)

    strict_prompt = """
You are an AI Career Mentor.

Use ONLY the provided career notes.

Do not invent facts.

If the answer is not supported by the notes, say:

"I don't know based on the provided career notes."

Context:
{context}

Question:
{question}

Answer:
"""

    simple_prompt = """
You are a helpful career mentor.

Answer the question using the following context.

Context:
{context}

Question:
{question}

Answer:
"""

    print(
        "\nSTRICT GROUNDED PROMPT:"
    )

    print(strict_prompt)

    print(
        "\nSIMPLE PROMPT:"
    )

    print(simple_prompt)

    conclusion = (
        "The strict grounded prompt is preferred because "
        "it explicitly restricts the mentor to the provided "
        "career notes and provides an explicit fallback for "
        "questions that cannot be answered from those notes."
    )

    print(
        "\nConclusion:"
    )

    print(conclusion)

    return {
        "strict_prompt":
            strict_prompt,

        "simple_prompt":
            simple_prompt,

        "conclusion":
            conclusion
    }


# ============================================================
# 25. GENERATE REPORT
# ============================================================

def generate_report(
    retrieval_results,
    answer_results,
    hallucination_results,
    prompt_results
):

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = []

    report.append(
        "# SmartHire Answer Quality Evaluation"
    )

    report.append("")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    report.append(
        "## 1. Evaluation Summary"
    )

    report.append("")

    report.append(
        "This evaluation measures job retrieval relevance, "
        "mentor answer quality, grounding, helpfulness, "
        "prompt behaviour, and hallucination handling."
    )

    report.append("")

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    report.append(
        "## 2. Retrieval Relevance"
    )

    report.append("")

    hit_rate = (
        retrieval_results["hit_rate"]
    )

    report.append(
        f"**Retrieval Hit Rate:** "
        f"{hit_rate:.3f} "
        f"({hit_rate * 100:.1f}%)"
    )

    report.append("")

    report.append(
        "| Test | Result | Retrieved Jobs |"
    )

    report.append(
        "|---|---|---|"
    )

    for item in retrieval_results["results"]:

        status = (
            "PASS"
            if item["hit"]
            else "MISS"
        )

        titles = []

        for job in item["jobs"]:

            titles.append(
                str(
                    job.get(
                        "jobtitle",
                        "Unknown"
                    )
                )
            )

        titles_text = "<br>".join(
            titles
        )

        report.append(
            f"| {item['name']} | "
            f"{status} | "
            f"{titles_text} |"
        )

    report.append("")

    # --------------------------------------------------------
    # Answer quality
    # --------------------------------------------------------

    report.append(
        "## 3. Answer Quality"
    )

    report.append("")

    report.append(
        f"- **Correctness:** "
        f"{answer_results['correctness']:.3f}"
    )

    report.append(
        f"- **Grounding:** "
        f"{answer_results['grounding']:.3f}"
    )

    report.append(
        f"- **Helpfulness:** "
        f"{answer_results['helpfulness']:.3f}"
    )

    report.append(
        f"- **Overall:** "
        f"{answer_results['overall']:.3f}"
    )

    report.append("")

    report.append(
        "| Question | Correctness | "
        "Grounding | Helpfulness |"
    )

    report.append(
        "|---|---:|---:|---:|"
    )

    for item in answer_results["results"]:

        question = (
            item["question"]
            .replace("|", "\\|")
            .replace("\n", " ")
        )

        report.append(
            f"| {question} | "
            f"{item['correctness']:.3f} | "
            f"{item['grounding']:.3f} | "
            f"{item['helpfulness']:.3f} |"
        )

    report.append("")

    # --------------------------------------------------------
    # Hallucination
    # --------------------------------------------------------

    report.append(
        "## 4. Hallucination Check"
    )

    report.append("")

    status = (
        "PASSED"
        if hallucination_results["passed"]
        else "REVIEW NEEDED"
    )

    report.append(
        f"**Status:** {status}"
    )

    report.append("")

    report.append(
        "**Question:**"
    )

    report.append("")

    report.append(
        f"> {hallucination_results['question']}"
    )

    report.append("")

    report.append(
        "**Model Answer:**"
    )

    report.append("")

    report.append(
        f"> {hallucination_results['answer']}"
    )

    report.append("")

    # --------------------------------------------------------
    # Prompt comparison
    # --------------------------------------------------------

    report.append(
        "## 5. Prompt Comparison"
    )

    report.append("")

    report.append(
        prompt_results["conclusion"]
    )

    report.append("")

    report.append(
        "### Strict Grounded Prompt"
    )

    report.append("")

    report.append(
        "- Uses only the provided career notes."
    )

    report.append(
        "- Prevents unsupported facts."
    )

    report.append(
        "- Provides an explicit unknown-answer response."
    )

    report.append(
        "- Reduces hallucination risk."
    )

    report.append("")

    report.append(
        "### Simple Prompt"
    )

    report.append("")

    report.append(
        "- Provides context."
    )

    report.append(
        "- Does not explicitly enforce the same "
        "grounding restrictions."
    )

    report.append(
        "- Does not provide an explicit fallback "
        "for unsupported questions."
    )

    report.append("")

    # --------------------------------------------------------
    # Conclusion
    # --------------------------------------------------------

    report.append(
        "## 6. Overall Conclusion"
    )

    report.append("")

    report.append(
        "The SmartHire evaluation checks whether the system "
        "can retrieve relevant jobs and generate career "
        "mentor answers grounded in the available career notes."
    )

    report.append("")

    report.append(
        "The strict grounded prompt is preferred because "
        "the mentor is required to use the provided career "
        "notes and refuse unsupported questions."
    )

    report.append("")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    REPORT_PATH.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print("\n")
    print("=" * 70)
    print("EVALUATION REPORT CREATED")
    print("=" * 70)

    print(
        REPORT_PATH
    )

    return REPORT_PATH


# ============================================================
# 26. MAIN
# ============================================================

def main():

    try:

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        retrieval_results = (
            evaluate_retrieval()
        )

        # ----------------------------------------------------
        # Answer quality
        # ----------------------------------------------------

        answer_results = (
            evaluate_answer_quality()
        )

        # ----------------------------------------------------
        # Hallucination
        # ----------------------------------------------------

        hallucination_results = (
            evaluate_hallucination()
        )

        # ----------------------------------------------------
        # Prompt comparison
        # ----------------------------------------------------

        prompt_results = (
            prompt_comparison()
        )

        # ----------------------------------------------------
        # Report
        # ----------------------------------------------------

        report_path = generate_report(
            retrieval_results,
            answer_results,
            hallucination_results,
            prompt_results
        )

        print("\n")
        print("=" * 70)
        print("ALL EVALUATION STEPS COMPLETED")
        print("=" * 70)

        print(
            f"\nReport saved to:\n{report_path}"
        )

    except Exception as error:

        print("\n")
        print("=" * 70)
        print("EVALUATION FAILED")
        print("=" * 70)

        print(
            f"\nError type: "
            f"{type(error).__name__}"
        )

        print(
            f"\nError message:\n{error}"
        )

        raise


# ============================================================
# 27. RUN
# ============================================================

if __name__ == "__main__":
    main()