
"""
vector_store.py

Modul bersama untuk mengakses ChromaDB dan menjalankan operasi
embedding serta cosine similarity.

Digunakan oleh:
- test_retrieval.py
- cross_chapter_retriever.py
- Script eksperimen batch berikutnya.

Jalankan script retrieval dari root project RAG.
"""

import os
from pathlib import Path

import chromadb
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# KONFIGURASI
# ============================================================

load_dotenv()

CHROMA_DIR = Path(
    os.getenv("CHROMA_DIR", "data/chroma_db")
)

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "openstax_books",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small",
)

DEFAULT_BOOK_TITLE = os.getenv(
    "DEFAULT_BOOK_TITLE",
    "Psychology 2e",
)


# ============================================================
# OPENAI CLIENT
# ============================================================

def open_openai_client():
    """Membuat OpenAI client setelah memeriksa API key."""

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY tidak ditemukan. "
            "Periksa file .env di root project."
        )

    return OpenAI(api_key=api_key)


# ============================================================
# CHROMADB
# ============================================================

def open_collection():
    """
    Membuka collection ChromaDB yang sudah ada.

    Fungsi ini tidak menjalankan embedding ulang dan tidak
    membuat collection baru.
    """

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Folder ChromaDB tidak ditemukan: {CHROMA_DIR}. "
            "Periksa lokasi database dan jalankan dari root project."
        )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME
        )
    except Exception as error:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' tidak ditemukan "
            f"di '{CHROMA_DIR}'. Periksa konfigurasi embedder."
        ) from error

    return collection


def check_metadata(collection, book_title):
    """
    Memastikan buku tersedia dan semua chunk buku tersebut
    memiliki chapter_number berupa integer.

    Hasil pemeriksaan digunakan untuk memastikan filter
    jarak nomor bab dapat diterapkan dengan benar.
    """

    data = collection.get(
        where={"book_title": book_title},
        include=["metadatas"],
    )

    if not data["ids"]:
        raise ValueError(
            f"Tidak ditemukan chunk untuk buku '{book_title}'. "
            "Pastikan nama buku sama persis dengan metadata "
            "book_title di ChromaDB."
        )

    invalid_chapters = []

    for vector_id, metadata in zip(
        data["ids"],
        data["metadatas"],
    ):
        chapter = metadata.get("chapter_number")

        # bool tidak diterima sebagai nomor bab.
        if not isinstance(chapter, int) or isinstance(chapter, bool):
            invalid_chapters.append(
                (vector_id, chapter)
            )

    if invalid_chapters:
        examples = invalid_chapters[:5]

        raise TypeError(
            "Ditemukan metadata chapter_number yang bukan integer. "
            f"Contoh vector_id dan nilainya: {examples}. "
            "Periksa metadata saat parsing dan embedding."
        )

    return len(data["ids"])


# ============================================================
# QUERY EMBEDDING
# ============================================================

def embed_query(client, text):
    """
    Mengubah satu query menjadi embedding menggunakan model
    yang sama dengan model embedding dokumen.

    Mengembalikan:
    - embedding sebagai NumPy array
    - jumlah token aktual untuk query tersebut
    """

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
    )

    vector = np.asarray(
        response.data[0].embedding,
        dtype=np.float32,
    )

    tokens = response.usage.total_tokens

    return vector, tokens


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vector_a, vector_b):
    """
    Menghitung cosine similarity.

    vector_a:
        Satu vektor dengan bentuk (dimensi,).

    vector_b:
        Satu vektor dengan bentuk (dimensi,), atau
        sekumpulan vektor dengan bentuk (jumlah_vektor, dimensi).

    Hasil:
        float jika vector_b satu vektor;
        NumPy array jika vector_b berisi banyak vektor.
    """

    vector_a = np.asarray(
        vector_a,
        dtype=np.float32,
    )

    vector_b = np.asarray(
        vector_b,
        dtype=np.float32,
    )

    if vector_a.ndim != 1:
        raise ValueError(
            "vector_a harus berupa satu vektor satu dimensi."
        )

    if vector_b.ndim not in (1, 2):
        raise ValueError(
            "vector_b harus berupa satu vektor atau matriks dua dimensi."
        )

    norm_a = np.linalg.norm(vector_a)

    if vector_b.ndim == 1:
        norm_b = np.linalg.norm(vector_b)
        denominator = norm_a * norm_b

        if denominator == 0:
            return 0.0

        return float(
            np.dot(vector_a, vector_b) / denominator
        )

    # Setiap baris pada vector_b merupakan satu embedding.
    norm_b = np.linalg.norm(
        vector_b,
        axis=1,
    )

    denominator = norm_a * norm_b

    similarities = np.zeros(
        vector_b.shape[0],
        dtype=np.float32,
    )

    valid = denominator > 0

    similarities[valid] = (
        vector_b[valid] @ vector_a
    ) / denominator[valid]

    return similarities
