"""
SmartHire GenAI Portal - Streamlit Application

Run from the project root:

    streamlit run app/streamlit_app.py

Flow:

    Upload CV
        ↓
    Validate Resume
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
from html import escape

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

    Priority:
    1. Streamlit secrets
    2. Existing environment variable
    3. Existing .env.example file
    """

    # --------------------------------------------------------
    # 1. Streamlit Secrets
    # --------------------------------------------------------

    try:

        if "GOOGLE_API_KEY" in st.secrets:

            api_key = str(
                st.secrets["GOOGLE_API_KEY"]
            ).strip()

            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
                return api_key

        if "GEMINI_API_KEY" in st.secrets:

            api_key = str(
                st.secrets["GEMINI_API_KEY"]
            ).strip()

            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
                os.environ["GEMINI_API_KEY"] = api_key
                return api_key

    except Exception:
        pass


    # --------------------------------------------------------
    # 2. Existing environment variable
    # --------------------------------------------------------

    api_key = os.getenv("GOOGLE_API_KEY")

    if api_key:

        api_key = api_key.strip()

        if api_key:
            return api_key


    # --------------------------------------------------------
    # GEMINI_API_KEY
    # --------------------------------------------------------

    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:

        api_key = api_key.strip()

        if api_key:

            os.environ["GOOGLE_API_KEY"] = api_key

            return api_key


    # --------------------------------------------------------
    # 3. Load existing .env.example
    # --------------------------------------------------------

    env_example_file = PROJECT_ROOT / ".env.example"

    if env_example_file.exists():

        try:

            from dotenv import load_dotenv

            load_dotenv(
                dotenv_path=env_example_file,
                override=False
            )

        except ImportError:
            pass


    # --------------------------------------------------------
    # 4. Check environment again
    # --------------------------------------------------------

    invalid_keys = {
        "your_api_key_here",
        "your_google_api_key_here",
        "your_gemini_api_key_here",
        "your_actual_api_key",
        "replace_with_your_api_key"
    }


    api_key = os.getenv("GOOGLE_API_KEY")

    if api_key:

        api_key = api_key.strip()

        if (
            api_key
            and api_key.lower() not in invalid_keys
        ):
            return api_key


    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:

        api_key = api_key.strip()

        if (
            api_key
            and api_key.lower() not in invalid_keys
        ):

            os.environ["GOOGLE_API_KEY"] = api_key

            return api_key


    return None


API_KEY = load_api_key()


if API_KEY:
    os.environ["GOOGLE_API_KEY"] = API_KEY


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

    /* ========================================================
       GLOBAL
       ======================================================== */

    .main-title {
        font-size: 42px;
        font-weight: 750;
        letter-spacing: -1px;
        margin-bottom: 4px;
    }

    .subtitle {
        font-size: 17px;
        opacity: 0.72;
        margin-bottom: 28px;
        line-height: 1.5;
    }

    .section-title {
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.3px;
        margin-top: 18px;
        margin-bottom: 18px;
    }


    /* ========================================================
       PROFILE
       ======================================================== */

    .profile-label {
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        opacity: 0.58;
        margin-bottom: 8px;
    }

    .candidate-name {
        font-size: 30px;
        font-weight: 750;
        line-height: 1.2;
        margin-bottom: 28px;
    }

    .target-role {
        font-size: 19px;
        font-weight: 600;
        line-height: 1.4;
        margin-bottom: 12px;
    }

    .profile-divider {
        height: 1px;
        margin: 5px 0 26px 0;
        background: rgba(128, 128, 128, 0.20);
    }


    /* ========================================================
       SKILLS
       ======================================================== */

    .skills-text {
        font-size: 17px;
        line-height: 1.9;
        font-weight: 450;
        margin-top: 4px;
        margin-bottom: 12px;
    }


    /* ========================================================
       EMPTY VALUES
       ======================================================== */

    .empty-value {
        font-size: 15px;
        opacity: 0.60;
    }


    /* ========================================================
       JOB CARDS
       ======================================================== */

    .job-card {
        padding: 18px 20px;
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 14px;
        margin-bottom: 14px;
    }

    .job-title {
        font-size: 19px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .job-id {
        font-size: 13px;
        opacity: 0.62;
    }


    /* ========================================================
       MENTOR ANSWER
       ======================================================== */

    .mentor-answer {
        font-size: 16px;
        line-height: 1.7;
        margin-top: 8px;
        margin-bottom: 18px;
    }

    .mentor-answer h1,
    .mentor-answer h2,
    .mentor-answer h3 {
        margin-top: 18px;
        margin-bottom: 8px;
    }

    .mentor-answer ul,
    .mentor-answer ol {
        margin-top: 5px;
        margin-bottom: 12px;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.16);
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        border-radius: 9px;
        font-weight: 650;
        min-height: 42px;
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    [data-testid="stFileUploader"] {
        margin-bottom: 10px;
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
        "Please configure GOOGLE_API_KEY using "
        "Streamlit Secrets, an environment variable, "
        "or your existing .env.example file."
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
# 10. RESUME VALIDATION FUNCTION
# ============================================================

def is_likely_resume(resume_text: str):

    """
    Check whether extracted text looks like a resume/CV.
    """

    if not resume_text:

        return (
            False,
            "The uploaded file contains no readable text."
        )


    text = resume_text.lower().strip()


    if len(text) < 100:

        return (
            False,
            "The uploaded document does not contain enough text "
            "to be recognized as a resume."
        )


    # --------------------------------------------------------
    # Resume-related keywords
    # --------------------------------------------------------

    resume_keywords = [

        "resume",
        "curriculum vitae",
        "cv",
        "career objective",
        "professional summary",
        "profile summary",
        "work experience",
        "professional experience",
        "employment history",
        "education",
        "skills",
        "technical skills",
        "projects",
        "certifications",
        "certificate",
        "achievements",
        "internship",
        "internships",
        "experience",
        "contact",
        "email",
        "phone",
        "linkedin",
        "github"
    ]


    matched_keywords = [
        keyword
        for keyword in resume_keywords
        if keyword in text
    ]


    # --------------------------------------------------------
    # Important resume sections
    # --------------------------------------------------------

    section_keywords = [

        "education",
        "skills",
        "experience",
        "work experience",
        "professional experience",
        "projects",
        "certifications",
        "career objective",
        "professional summary",
        "employment history"
    ]


    matched_sections = [
        section
        for section in section_keywords
        if section in text
    ]


    # --------------------------------------------------------
    # Resume identification rules
    # --------------------------------------------------------

    if (
        len(matched_keywords) >= 3
        and len(matched_sections) >= 2
    ):

        return True, ""


    contact_present = (
        "@" in text
        or "phone" in text
        or "mobile" in text
        or "linkedin" in text
    )


    if (
        contact_present
        and len(matched_sections) >= 2
        and len(matched_keywords) >= 4
    ):

        return True, ""


    return (
        False,
        "The uploaded document does not appear to be a resume/CV. "
        "Please upload a valid resume in PDF or DOCX format."
    )


# ============================================================
# 11. JOB MATCHING FUNCTION
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
# 12. CAREER MENTOR QUESTION VALIDATION
# ============================================================

def is_career_question(question: str):
    """
    Determine whether the user's question is related to
    careers, jobs, resumes, skills, interviews, education,
    professional development, or career planning.

    Irrelevant questions such as:
        - What is your name?
        - Tell me a joke
        - What is the weather?
        - Who is the president?
        - How are you?

    are rejected before the RAG system is called.
    """

    if not question:
        return False


    question = question.lower().strip()


    # --------------------------------------------------------
    # Remove common punctuation
    # --------------------------------------------------------

    normalized_question = (
        question
        .replace("?", " ")
        .replace("!", " ")
        .replace(".", " ")
        .replace(",", " ")
        .replace(":", " ")
        .replace(";", " ")
    )


    # --------------------------------------------------------
    # Career-related keywords
    # --------------------------------------------------------

    career_keywords = [

        # Career
        "career",
        "careers",
        "job",
        "jobs",
        "occupation",
        "profession",
        "professional",
        "work",
        "workplace",
        "employment",
        "employability",

        # Resume / CV
        "resume",
        "cv",
        "curriculum vitae",
        "cover letter",
        "application",
        "job application",
        "portfolio",
        "linkedin",
        "github",

        # Skills
        "skill",
        "skills",
        "technical skill",
        "soft skill",
        "programming",
        "coding",
        "python",
        "java",
        "javascript",
        "machine learning",
        "artificial intelligence",
        "ai",
        "data science",
        "data analyst",
        "data analytics",
        "deep learning",
        "nlp",
        "cloud",
        "aws",
        "azure",
        "gcp",
        "sql",
        "database",
        "devops",
        "software development",
        "software engineer",
        "developer",

        # Interviews
        "interview",
        "interviews",
        "interview preparation",
        "interview questions",
        "hr interview",
        "technical interview",

        # Education / learning
        "course",
        "courses",
        "certification",
        "certifications",
        "certificate",
        "degree",
        "education",
        "college",
        "university",
        "learning",
        "learn",
        "study",
        "training",

        # Career development
        "experience",
        "internship",
        "internships",
        "intern",
        "promotion",
        "salary",
        "salary negotiation",
        "career growth",
        "career path",
        "career goal",
        "career goals",
        "career change",
        "switch career",
        "switching career",
        "professional growth",
        "professional development",

        # Job search
        "job search",
        "job searching",
        "job opportunity",
        "job opportunities",
        "job role",
        "job roles",
        "vacancy",
        "vacancies",
        "hiring",
        "recruitment",
        "recruiter",
        "recruiters",

        # Specific career actions
        "how to get a job",
        "how to get hired",
        "how to prepare",
        "what should i learn",
        "what should i improve",
        "what skills should i improve",
        "what skills should i learn"
    ]


    # --------------------------------------------------------
    # Check for career keywords
    # --------------------------------------------------------

    for keyword in career_keywords:

        if keyword in normalized_question:

            return True


    # --------------------------------------------------------
    # Career question patterns
    # --------------------------------------------------------

    career_patterns = [

        "how can i improve my resume",
        "how can i improve my cv",
        "how do i improve my resume",
        "how do i improve my cv",
        "what should i put on my resume",
        "what should i put on my cv",
        "how can i get hired",
        "how can i prepare for an interview",
        "what should i learn for",
        "what skills are required for",
        "what skills do i need for",
        "how do i become a",
        "how can i become a",
        "is this a good career",
        "which career should i choose",
        "which job should i choose",
        "how do i find a job",
        "how can i find a job"
    ]


    for pattern in career_patterns:

        if pattern in normalized_question:

            return True


    return False


# ============================================================
# 13. SESSION STATE
# ============================================================

if "resume_text" not in st.session_state:
    st.session_state.resume_text = None


if "profile" not in st.session_state:
    st.session_state.profile = None


if "matched_jobs" not in st.session_state:
    st.session_state.matched_jobs = []


if "cv_suggestions" not in st.session_state:
    st.session_state.cv_suggestions = None


if "mentor_answer" not in st.session_state:
    st.session_state.mentor_answer = None


if "mentor_question" not in st.session_state:
    st.session_state.mentor_question = None


if "mentor_question_allowed" not in st.session_state:
    st.session_state.mentor_question_allowed = False


# ============================================================
# 14. SIDEBAR
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
# 15. CV UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📄 1. Upload Your CV'
    '</div>',
    unsafe_allow_html=True
)


st.info(
    "Only resumes/CVs in PDF or DOCX format are accepted. "
    "Other documents such as reports, assignments, invoices, "
    "or books will be rejected."
)


uploaded_file = st.file_uploader(
    "Upload your resume",
    type=[
        "pdf",
        "docx"
    ],
    help="Accepted formats: PDF and DOCX resumes only."
)


if uploaded_file is not None:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )


    if st.button(
        "Analyze Resume",
        type="primary"
    ):

        temp_path = None


        with st.spinner(
            "Reading and analyzing your resume..."
        ):

            try:

                # ------------------------------------------------
                # Validate file extension
                # ------------------------------------------------

                suffix = Path(
                    uploaded_file.name
                ).suffix.lower()


                if suffix not in {
                    ".pdf",
                    ".docx"
                }:

                    raise ValueError(
                        "Only PDF and DOCX resume files are accepted."
                    )


                # ------------------------------------------------
                # Save uploaded file temporarily
                # ------------------------------------------------

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
                        "Could not extract text from the uploaded "
                        "PDF/DOCX file. Please upload a readable resume."
                    )


                # ------------------------------------------------
                # Validate resume
                # ------------------------------------------------

                is_resume, validation_message = (
                    is_likely_resume(
                        resume_text
                    )
                )


                if not is_resume:

                    st.error(
                        "❌ This document was not accepted."
                    )

                    st.warning(
                        validation_message
                    )

                    st.info(
                        "Please upload a genuine resume/CV "
                        "in PDF or DOCX format."
                    )


                    # Clear previous resume data

                    st.session_state.resume_text = None
                    st.session_state.profile = None
                    st.session_state.matched_jobs = []
                    st.session_state.cv_suggestions = None
                    st.session_state.mentor_answer = None
                    st.session_state.mentor_question = None
                    st.session_state.mentor_question_allowed = False

                    st.stop()


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
                # Save session state
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

                st.session_state.mentor_answer = None

                st.session_state.mentor_question = None

                st.session_state.mentor_question_allowed = False


                st.success(
                    "✅ Resume analysis completed successfully."
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
# 16. DISPLAY PARSED PROFILE
# ============================================================

if st.session_state.profile:

    profile = st.session_state.profile

    st.divider()


    st.markdown(
        '<div class="section-title">'
        '👤 2. Parsed Candidate Profile'
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # PROFILE INFORMATION
    # ========================================================

    col1, col2 = st.columns(
        [1, 1.35],
        gap="large"
    )


    # ========================================================
    # LEFT SIDE
    # ========================================================

    with col1:

        st.markdown(
            '<div class="profile-label">Candidate</div>',
            unsafe_allow_html=True
        )


        candidate_name = str(
            profile.get(
                "name",
                ""
            )
        ).strip()


        if candidate_name:

            st.markdown(
                f'<div class="candidate-name">'
                f'{escape(candidate_name)}'
                f'</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="empty-value">'
                'Candidate name not found'
                '</div>',
                unsafe_allow_html=True
            )


        st.markdown(
            '<div class="profile-divider"></div>',
            unsafe_allow_html=True
        )


        st.markdown(
            '<div class="profile-label">Target Role</div>',
            unsafe_allow_html=True
        )


        target_role = str(
            profile.get(
                "target_role",
                ""
            )
        ).strip()


        if target_role:

            st.markdown(
                f'<div class="target-role">'
                f'{escape(target_role)}'
                f'</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="empty-value">'
                'Target role not found'
                '</div>',
                unsafe_allow_html=True
            )


    # ========================================================
    # RIGHT SIDE
    # ========================================================

    with col2:

        st.markdown(
            '<div class="profile-label">Skills</div>',
            unsafe_allow_html=True
        )


        skills = profile.get(
            "skills",
            []
        )


        if isinstance(
            skills,
            list
        ):

            clean_skills = [
                str(skill).strip()
                for skill in skills
                if str(skill).strip()
            ]

        else:

            clean_skills = [
                skill.strip()
                for skill in str(
                    skills
                ).split(",")
                if skill.strip()
            ]


        if clean_skills:

            skills_text = " • ".join(
                escape(skill)
                for skill in clean_skills
            )


            st.markdown(
                f'<div class="skills-text">'
                f'{skills_text}'
                f'</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="empty-value">'
                'No skills found.'
                '</div>',
                unsafe_allow_html=True
            )


    # ========================================================
    # EXPERIENCE & EDUCATION
    # ========================================================

    with st.expander(
        "View Experience and Education"
    ):

        st.subheader(
            "Experience"
        )


        experience = str(
            profile.get(
                "experience",
                ""
            )
        ).strip()


        if experience:

            st.write(
                experience
            )

        else:

            st.caption(
                "No experience information found."
            )


        st.subheader(
            "Education"
        )


        education = str(
            profile.get(
                "education",
                ""
            )
        ).strip()


        if education:

            st.write(
                education
            )

        else:

            st.caption(
                "No education information found."
            )


# ============================================================
# 17. MATCHED JOBS
# ============================================================

if st.session_state.matched_jobs:

    st.divider()


    st.markdown(
        '<div class="section-title">'
        '🎯 3. Recommended Jobs'
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
# 18. CV SUGGESTIONS
# ============================================================

if (
    st.session_state.resume_text
    and st.session_state.matched_jobs
):

    st.divider()


    st.markdown(
        '<div class="section-title">'
        '✍️ 4. CV Improvement Suggestions'
        '</div>',
        unsafe_allow_html=True
    )


    st.write(
        "Get suggestions for improving your CV "
        "for the top matched job."
    )


    if st.button(
        "Generate CV Suggestions"
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
# 19. CAREER MENTOR
# ============================================================

st.divider()


# IMPORTANT:
# Number 5 has been removed from the heading.

st.markdown(
    '<div class="section-title">'
    '💬 Career Mentor'
    '</div>',
    unsafe_allow_html=True
)


st.write(
    "Ask questions about careers using the "
    "career notes knowledge base."
)


question = st.text_input(
    "Ask your career question",
    placeholder=(
        "Example: What skills should I improve "
        "for an AI Engineer role?"
    )
)


if st.button(
    "Ask Mentor"
):

    # --------------------------------------------------------
    # Clear previous mentor answer
    # --------------------------------------------------------

    st.session_state.mentor_answer = None

    st.session_state.mentor_question = None

    st.session_state.mentor_question_allowed = False


    # --------------------------------------------------------
    # Empty question
    # --------------------------------------------------------

    if not question.strip():

        st.error(
            "❌ Please enter a career-related question."
        )


    else:

        clean_question = question.strip()


        # ----------------------------------------------------
        # STEP 1: Career relevance check
        # ----------------------------------------------------

        if not is_career_question(
            clean_question
        ):

            st.error(
                "❌ Invalid question. "
                "Please ask a question related to careers, "
                "jobs, resumes, skills, interviews, "
                "education, or professional development."
            )

            st.info(
                "Example: "
                "What skills should I learn for an AI Engineer role?"
            )


            # IMPORTANT:
            # Do NOT call answer_question()
            # Do NOT show resources.

            st.session_state.mentor_answer = None

            st.session_state.mentor_question = None

            st.session_state.mentor_question_allowed = False


        else:

            # ------------------------------------------------
            # STEP 2: Existing safety check
            # ------------------------------------------------

            with st.spinner(
                "Thinking..."
            ):

                try:

                    safety_result = check_question(
                        clean_question
                    )


                    # ------------------------------------------------
                    # Handle safety result
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
                                    else (
                                        "This question cannot "
                                        "be answered."
                                    )
                                )


                                st.error(
                                    f"❌ {message}"
                                )


                                st.session_state.mentor_answer = None

                                st.session_state.mentor_question = None

                                st.session_state.mentor_question_allowed = False

                                st.stop()


                        elif isinstance(
                            safety_result,
                            bool
                        ):

                            if not safety_result:

                                st.error(
                                    "❌ This question cannot "
                                    "be answered."
                                )


                                st.session_state.mentor_answer = None

                                st.session_state.mentor_question = None

                                st.session_state.mentor_question_allowed = False

                                st.stop()


                    # ------------------------------------------------
                    # STEP 3: Ask RAG mentor
                    # ------------------------------------------------

                    answer = answer_question(
                        clean_question
                    )


                    if answer:

                        st.session_state.mentor_answer = (
                            answer
                        )

                        st.session_state.mentor_question = (
                            clean_question
                        )

                        st.session_state.mentor_question_allowed = True


                    else:

                        st.session_state.mentor_answer = None

                        st.session_state.mentor_question = None

                        st.session_state.mentor_question_allowed = False

                        st.info(
                            "No answer was generated."
                        )


                except Exception as error:

                    st.session_state.mentor_answer = None

                    st.session_state.mentor_question = None

                    st.session_state.mentor_question_allowed = False

                    st.error(
                        "Could not answer the question:\n\n"
                        f"{error}"
                    )


# ============================================================
# 20. DISPLAY MENTOR ANSWER
# ============================================================

if (
    st.session_state.mentor_answer
    and st.session_state.mentor_question_allowed
):

    st.markdown(
        "### Mentor Answer"
    )


    st.markdown(
        st.session_state.mentor_answer
    )


    # ========================================================
    # RECOMMENDED RESOURCES
    # ========================================================
    #
    # Resources are displayed ONLY when:
    #
    # 1. The question passed career validation
    # 2. The safety check passed
    # 3. A mentor answer was generated
    #
    # Irrelevant questions will NEVER reach this section.
    # ========================================================

    st.markdown(
        "### 📚 Recommended Resources"
    )


    st.caption(
        "Useful resources to strengthen your skills "
        "and prepare for your target career."
    )


    resources = [

        {
            "title": "Python Documentation",
            "url": "https://docs.python.org/3/"
        },

        {
            "title": "Scikit-learn User Guide",
            "url": "https://scikit-learn.org/stable/user_guide.html"
        },

        {
            "title": "Kaggle Learn",
            "url": "https://www.kaggle.com/learn"
        },

        {
            "title": "GitHub Skills",
            "url": "https://skills.github.com/"
        },

        {
            "title": "Hugging Face Learn",
            "url": "https://huggingface.co/learn"
        }
    ]


    for resource in resources:

        st.markdown(
            f"- [{resource['title']}]({resource['url']})"
        )


# ============================================================
# 21. FOOTER
# ============================================================

st.divider()


st.caption(
    "SmartHire GenAI • AI-powered career assistance"
)