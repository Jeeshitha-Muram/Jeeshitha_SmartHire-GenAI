"""
SmartHire GenAI Portal - Streamlit Application

Run from the project root:

    streamlit run app/streamlit_app.py

Flow:

    Upload CV
        ↓
    Parse CV
        ↓
    Match Jobs
        ↓
    CV Suggestions
        ↓
    Career Mentor
"""

import os
import sys
import json
import tempfile
from pathlib import Path

import streamlit as st
import numpy as np
import faiss


# ============================================================
# 1. FIND PROJECT ROOT
# ============================================================

CURRENT = Path(__file__).resolve()

for folder in [CURRENT.parent] + list(CURRENT.parents):

    if (
        (folder / "src").is_dir()
        and (folder / "data").is_dir()
        and (folder / "vectorstore").is_dir()
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
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartHire GenAI",
    page_icon="💼",
    layout="wide"
)


# ============================================================
# 3. LOAD API KEY
# ============================================================

def load_api_key():
    """
    Load Gemini API key.

    Deployment:
        Uses Streamlit secrets.

    Local development:
        Uses .env file.
    """

    # --------------------------------------------------------
    # Streamlit Cloud secrets
    # --------------------------------------------------------

    try:

        if "GOOGLE_API_KEY" in st.secrets:

            api_key = st.secrets["GOOGLE_API_KEY"]

            if api_key:

                os.environ["GOOGLE_API_KEY"] = str(api_key)

                return str(api_key)

    except Exception:
        pass


    # --------------------------------------------------------
    # Local .env file
    # --------------------------------------------------------

    env_file = PROJECT_ROOT / ".env"

    if env_file.exists():

        try:

            from dotenv import load_dotenv

            load_dotenv(
                env_file,
                override=False
            )

        except ImportError:
            pass


    api_key = os.getenv(
        "GOOGLE_API_KEY"
    )

    if api_key:
        return api_key

    return None


API_KEY = load_api_key()


# ============================================================
# 4. IMPORT PROJECT MODULES
# ============================================================

from src import config

from src.parsing.loader import load_text

from src.parsing.resume_parser import parse_resume

from src.search.embed import embed_text

from src.generate.cv_suggestions import (
    generate_cv_suggestions
)

from src.mentor.rag_chain import (
    answer_question
)

from src.safety.guardrails import (
    check_question
)


# ============================================================
# 5. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .profile-box {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 15px;
    }

    .job-box {
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 6. HEADER
# ============================================================

st.markdown(
    '<div class="main-title">💼 SmartHire GenAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered resume analysis, job matching, CV improvement, '
    'and career mentoring.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# 7. CHECK API KEY
# ============================================================

if not API_KEY:

    st.error(
        "Gemini API key was not found."
    )

    st.info(
        "For local development, add GOOGLE_API_KEY "
        "to your .env file. "
        "For deployment, add it to Streamlit Secrets."
    )

    st.stop()


# ============================================================
# 8. LOAD JOB FAISS INDEX
# ============================================================

@st.cache_resource
def load_job_resources():

    index_file = (
        config.JOBS_INDEX_DIR /
        "jobs.faiss"
    )

    metadata_file = (
        config.JOBS_INDEX_DIR /
        "jobs.json"
    )

    if not index_file.exists():

        raise FileNotFoundError(
            f"Job FAISS index not found:\n"
            f"{index_file}\n\n"
            "Run Notebook 02 first."
        )

    if not metadata_file.exists():

        raise FileNotFoundError(
            f"Job metadata not found:\n"
            f"{metadata_file}\n\n"
            "Run Notebook 02 first."
        )

    # --------------------------------------------------------
    # Load FAISS
    # --------------------------------------------------------

    index = faiss.read_index(
        str(index_file)
    )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    with open(
        metadata_file,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if index.d != config.EMBED_DIM:

        raise ValueError(
            f"FAISS dimension is {index.d}, "
            f"but expected {config.EMBED_DIM}."
        )

    if index.ntotal != len(metadata):

        raise ValueError(
            "FAISS vector count does not match "
            "metadata count."
        )

    return index, metadata


# ============================================================
# 9. LOAD JOB RESOURCES
# ============================================================

try:

    job_index, job_metadata = (
        load_job_resources()
    )

except Exception as error:

    st.error(
        f"Unable to load job database:\n\n{error}"
    )

    st.stop()


# ============================================================
# 10. JOB MATCHING FUNCTION
# ============================================================

def match_jobs(
    profile: dict,
    top_n: int = 5
):
    """
    Match a parsed candidate profile
    against the job FAISS index.
    """

    skills = profile.get(
        "skills",
        []
    )

    if isinstance(skills, list):

        skills_text = ", ".join(
            str(skill)
            for skill in skills
        )

    else:

        skills_text = str(skills)

    experience = str(
        profile.get(
            "experience",
            ""
        )
    )

    education = str(
        profile.get(
            "education",
            ""
        )
    )

    target_role = str(
        profile.get(
            "target_role",
            ""
        )
    )

    candidate_text = (
        "Target Role: "
        + target_role
        + "\nSkills: "
        + skills_text
        + "\nExperience: "
        + experience
        + "\nEducation: "
        + education
    )

    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_vector = embed_text(
        candidate_text
    )

    query_vector = np.asarray(
        query_vector,
        dtype=np.float32
    ).reshape(
        1,
        -1
    )

    # --------------------------------------------------------
    # Validate embedding dimension
    # --------------------------------------------------------

    if query_vector.shape[1] != config.EMBED_DIM:

        raise ValueError(
            f"Query embedding dimension is "
            f"{query_vector.shape[1]}, "
            f"but expected {config.EMBED_DIM}."
        )

    # --------------------------------------------------------
    # Search FAISS
    # --------------------------------------------------------

    scores, indices = job_index.search(
        query_vector,
        min(
            top_n,
            job_index.ntotal
        )
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        metadata = job_metadata[
            int(idx)
        ]

        results.append(
            {
                "score": float(score),

                "jobid": metadata.get(
                    "jobid",
                    ""
                ),

                "jobtitle": metadata.get(
                    "jobtitle",
                    ""
                ),

                "skills": metadata.get(
                    "skills",
                    ""
                ),

                "jobdescription": metadata.get(
                    "jobdescription",
                    ""
                ),

                "job_text": metadata.get(
                    "job_text",
                    ""
                )
            }
        )

    return results


# ============================================================
# 11. SESSION STATE
# ============================================================

if "resume_text" not in st.session_state:
    st.session_state.resume_text = None

if "profile" not in st.session_state:
    st.session_state.profile = None

if "matched_jobs" not in st.session_state:
    st.session_state.matched_jobs = []

if "cv_suggestions" not in st.session_state:
    st.session_state.cv_suggestions = None


# ============================================================
# 12. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("SmartHire")

    st.write(
        "AI Career Assistant"
    )

    st.divider()

    st.write(
        "Job database"
    )

    st.write(
        f"📊 {job_index.ntotal:,} jobs"
    )

    st.write(
        f"🔢 {config.EMBED_DIM}-dimensional embeddings"
    )

    st.write(
        f"🤖 {config.EMBED_MODEL}"
    )

    st.divider()

    st.write(
        "Upload your CV to begin."
    )


# ============================================================
# 13. CV UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">'
    '1. Upload Your CV'
    '</div>',
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Upload your resume",
    type=[
        "pdf",
        "docx",
        "txt",
        "md"
    ]
)


if uploaded_file is not None:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    # --------------------------------------------------------
    # Parse button
    # --------------------------------------------------------

    if st.button(
        "🔍 Analyze Resume",
        type="primary"
    ):

        temp_path = None

        with st.spinner(
            "Reading and analyzing your resume..."
        ):

            try:

                # ------------------------------------------------
                # Save uploaded file temporarily
                # ------------------------------------------------

                suffix = Path(
                    uploaded_file.name
                ).suffix

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    temp_path = Path(
                        temp_file.name
                    )

                # ------------------------------------------------
                # Extract text
                # ------------------------------------------------

                resume_text = load_text(
                    temp_path
                )

                if (
                    not resume_text
                    or not resume_text.strip()
                ):

                    raise ValueError(
                        "Could not extract text "
                        "from the uploaded resume."
                    )

                # ------------------------------------------------
                # Parse resume
                # ------------------------------------------------

                profile = parse_resume(
                    resume_text
                )

                # ------------------------------------------------
                # Match jobs
                # ------------------------------------------------

                matched_jobs = match_jobs(
                    profile,
                    config.TOP_N_JOBS
                )

                # ------------------------------------------------
                # Save in session
                # ------------------------------------------------

                st.session_state.resume_text = (
                    resume_text
                )

                st.session_state.profile = (
                    profile
                )

                st.session_state.matched_jobs = (
                    matched_jobs
                )

                st.session_state.cv_suggestions = None

                st.success(
                    "Resume analysis completed."
                )

            except Exception as error:

                st.error(
                    f"Resume analysis failed:\n\n{error}"
                )

            finally:

                if temp_path is not None:

                    try:

                        if temp_path.exists():
                            temp_path.unlink()

                    except Exception:
                        pass


# ============================================================
# 14. DISPLAY PARSED PROFILE
# ============================================================

if st.session_state.profile:

    profile = (
        st.session_state.profile
    )

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '2. Parsed Candidate Profile'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="profile-box">',
            unsafe_allow_html=True
        )

        st.subheader(
            "Candidate"
        )

        st.write(
            profile.get(
                "name",
                ""
            )
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="profile-box">',
            unsafe_allow_html=True
        )

        st.subheader(
            "Target Role"
        )

        st.write(
            profile.get(
                "target_role",
                ""
            )
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            '<div class="profile-box">',
            unsafe_allow_html=True
        )

        st.subheader(
            "Skills"
        )

        skills = profile.get(
            "skills",
            []
        )

        if isinstance(skills, list):

            if skills:

                for skill in skills:

                    st.write(
                        f"• {skill}"
                    )

            else:

                st.write(
                    "No skills found."
                )

        else:

            st.write(
                skills
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    with st.expander(
        "View Experience and Education"
    ):

        st.subheader(
            "Experience"
        )

        st.write(
            profile.get(
                "experience",
                ""
            )
        )

        st.subheader(
            "Education"
        )

        st.write(
            profile.get(
                "education",
                ""
            )
        )


# ============================================================
# 15. MATCHED JOBS
# ============================================================

if st.session_state.matched_jobs:

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '3. Recommended Jobs'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Top matching jobs based on your resume:"
    )

    for rank, job in enumerate(
        st.session_state.matched_jobs,
        start=1
    ):

        with st.container(
            border=True
        ):

            st.subheader(
                f"{rank}. {job['jobtitle']}"
            )

            col1, col2 = st.columns(
                [3, 1]
            )

            with col1:

                if job["jobid"]:

                    st.write(
                        f"Job ID: {job['jobid']}"
                    )

                if job["skills"]:

                    st.write(
                        f"**Skills:** "
                        f"{job['skills']}"
                    )

            with col2:

                st.metric(
                    "Match Score",
                    f"{job['score']:.3f}"
                )

            with st.expander(
                "View Job Description"
            ):

                st.write(
                    job["jobdescription"]
                )


# ============================================================
# 16. CV SUGGESTIONS
# ============================================================

if (
    st.session_state.resume_text
    and st.session_state.matched_jobs
):

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '4. CV Improvement Suggestions'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Get suggestions for improving your CV "
        "for the top matched job."
    )

    if st.button(
        "✨ Generate CV Suggestions"
    ):

        top_job = (
            st.session_state.matched_jobs[0]
        )

        job_text = (
            top_job["job_text"]
        )

        with st.spinner(
            "Generating CV suggestions..."
        ):

            try:

                suggestions = (
                    generate_cv_suggestions(
                        st.session_state.resume_text,
                        job_text
                    )
                )

                st.session_state.cv_suggestions = (
                    suggestions
                )

            except Exception as error:

                st.error(
                    "Could not generate suggestions:\n\n"
                    f"{error}"
                )

    if st.session_state.cv_suggestions:

        st.markdown(
            "### Suggestions"
        )

        st.markdown(
            st.session_state.cv_suggestions
        )


# ============================================================
# 17. CAREER MENTOR
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    '5. Career Mentor'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Ask questions about careers using the "
    "career notes knowledge base."
)

question = st.text_input(
    "Ask your career question",
    placeholder="Enter your career question..."
)


if st.button(
    "💬Ask Mentor"
):

    if not question.strip():

        st.warning(
            "Please enter a career question."
        )

    else:

        with st.spinner(
            "Thinking..."
        ):

            try:

                # ------------------------------------------------
                # Safety check
                # ------------------------------------------------

                safety_result = check_question(
                    question
                )

                # ------------------------------------------------
                # Handle blocked questions
                # ------------------------------------------------

                if safety_result:

                    if isinstance(
                        safety_result,
                        tuple
                    ):

                        allowed = safety_result[0]

                        if not allowed:

                            message = (
                                safety_result[1]
                                if len(safety_result) > 1
                                else "This question cannot be answered."
                            )

                            st.warning(
                                message
                            )

                            st.stop()

                    elif isinstance(
                        safety_result,
                        bool
                    ):

                        if not safety_result:

                            st.warning(
                                "This question cannot be answered."
                            )

                            st.stop()

                # ------------------------------------------------
                # Ask RAG mentor
                # ------------------------------------------------

                answer = answer_question(
                    question
                )

                if answer:

                    st.markdown(
                        "### Mentor Answer"
                    )

                    st.markdown(
                        answer
                    )

                else:

                    st.info(
                        "No answer was generated."
                    )

            except Exception as error:

                st.error(
                    "Could not answer the question:\n\n"
                    f"{error}"
                )


# ============================================================
# 18. FOOTER
# ============================================================

st.divider()

st.caption(
    "SmartHire GenAI • AI-powered career assistance"
)