import numpy as np
from sentence_transformers import SentenceTransformer

from src import config


MODEL_NAME = config.EMBED_MODEL

model = SentenceTransformer(MODEL_NAME)

EMBEDDING_DIMENSION = config.EMBED_DIM


def embed_text(text: str) -> np.ndarray:
    """
    Convert a single text into a normalized embedding vector.
    """

    if text is None:
        raise ValueError("Text cannot be None.")

    text = str(text).strip()

    if not text:
        return np.zeros(
            EMBEDDING_DIMENSION,
            dtype=np.float32
        )

    embedding = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    if embedding.shape != (EMBEDDING_DIMENSION,):
        raise ValueError(
            f"Expected embedding shape "
            f"({EMBEDDING_DIMENSION},), "
            f"but got {embedding.shape}"
        )

    return embedding


def embed_texts(
    texts: list[str],
    batch_size: int = 32
) -> np.ndarray:
    """
    Convert multiple texts into normalized embeddings.
    """

    if texts is None:
        raise ValueError("Texts cannot be None.")

    if len(texts) == 0:
        return np.empty(
            (0, EMBEDDING_DIMENSION),
            dtype=np.float32
        )

    cleaned_texts = [
        "" if text is None else str(text).strip()
        for text in texts
    ]

    embeddings = model.encode(
        cleaned_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    expected_shape = (
        len(cleaned_texts),
        EMBEDDING_DIMENSION
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Expected embedding shape "
            f"{expected_shape}, "
            f"but got {embeddings.shape}"
        )

    return embeddings


def cosine_similarity(
    vector_a,
    vector_b
) -> float:
    """
    Calculate cosine similarity between two vectors.
    """

    vector_a = np.asarray(
        vector_a,
        dtype=np.float32
    ).reshape(-1)

    vector_b = np.asarray(
        vector_b,
        dtype=np.float32
    ).reshape(-1)

    if vector_a.shape[0] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"vector_a must have dimension "
            f"{EMBEDDING_DIMENSION}"
        )

    if vector_b.shape[0] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"vector_b must have dimension "
            f"{EMBEDDING_DIMENSION}"
        )

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(
        np.dot(vector_a, vector_b)
        / (norm_a * norm_b)
    )