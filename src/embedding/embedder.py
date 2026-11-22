
"""
embedder.py

Membuat embedding untuk chunk buku OpenStax menggunakan
text-embedding-3-small dan menyimpannya ke ChromaDB lokal.

Fitur:
- Memproses teks dalam batch.
- Menggunakan satu collection untuk semua buku.
- Menyimpan metadata book_title dan metadata sumber lainnya.
- Mengestimasi token dan biaya sebelum embedding.
- Mencatat token aktual dari respons API.
- Melewati chunk yang sudah tersimpan dan belum berubah.
- Memperbarui embedding jika teks atau metadata berubah.
- Mendukung penambahan buku baru, seperti Concepts of Biology.

Jalankan dari root project:
    python -m src.embedder
"""

import hashlib
import json
import os
import time
from pathlib import Path

import chromadb
import tiktoken
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    OpenAI,
    RateLimitError,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Membaca API key dari file .env.
load_dotenv()

# Model embedding yang digunakan.
EMBEDDING_MODEL = "text-embedding-3-small"

# Semua buku OpenStax disimpan dalam collection yang sama.
COLLECTION_NAME = "openstax_books"

# Folder database lokal.
PERSIST_DIRECTORY = Path("data/chroma_db")

# Maksimal jumlah teks dalam satu request embedding.
BATCH_SIZE = 100

# Jumlah percobaan ulang jika terjadi gangguan sementara.
MAX_RETRIES = 5

# Jeda awal retry dalam detik, menggunakan exponential backoff.
RETRY_BASE_SECONDS = 2

# Harga standar per satu juta token input.
# Perbarui nilai ini jika harga resmi berubah.
PRICE_PER_MILLION_TOKENS = 0.02

# Encoding untuk memperkirakan jumlah token teks.
TOKEN_ENCODING = "cl100k_base"
encoding = tiktoken.get_encoding(TOKEN_ENCODING)


# ============================================================
# BOOK CONFIGURATION
# ============================================================

# Saat ini hanya Psychology 2e.
# Tambahkan konfigurasi Concepts of Biology nanti.
#
# Setiap buku menggunakan id_prefix yang berbeda agar ID
# chunk dari dua buku tidak saling bertabrakan.

BOOKS = [
    {
        "book_title": "Psychology 2e",
        "chunks_path": Path("data/psychology/chunks.json"),
        "id_prefix": "psychology_2e",
    },

    # Contoh konfigurasi untuk buku kedua:
    #
    # {
    #     "book_title": "Concepts of Biology",
    #     "chunks_path": Path("data/biology/chunks.json"),
    #     "id_prefix": "concepts_of_biology",
    # },
]


# ============================================================
# TOKEN ESTIMATION
# ============================================================

def count_tokens(text):
    """Memperkirakan jumlah token teks menggunakan tiktoken."""
    return len(encoding.encode(text))


def estimate_cost(total_tokens):
    """Menghitung perkiraan biaya embedding dalam USD."""
    return (
        total_tokens / 1_000_000
    ) * PRICE_PER_MILLION_TOKENS


# ============================================================
# METADATA NORMALIZATION
# ============================================================

def make_chroma_safe(value):
    """
    Memastikan nilai metadata kompatibel dengan ChromaDB.
    Nilai sederhana dipertahankan; nilai kompleks diubah
    menjadi string JSON.
    """

    if value is None:
        return ""

    if isinstance(value, (str, int, float, bool)):
        return value

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
    )


def normalize_chunk(chunk, book_title):
    """
    Menormalisasi struktur satu chunk.

    Mendukung struktur:
    {
        "chunk_id": "...",
        "text": "...",
        "metadata": {...}
    }

    Metadata disimpan sebagai nilai sederhana yang
    kompatibel dengan ChromaDB.
    """

    if not isinstance(chunk, dict):
        raise ValueError("Format chunk harus berupa dictionary.")

    text = chunk.get("text", "")

    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            "Ditemukan chunk tanpa teks yang valid."
        )

    metadata = dict(chunk.get("metadata") or {})

    # Menggabungkan field di luar metadata apabila tersedia.
    for key, value in chunk.items():
        if key not in ("text", "metadata"):
            metadata.setdefault(key, value)

    # Mengambil ID asli chunk.
    chunk_id = (
        chunk.get("chunk_id")
        or metadata.get("chunk_id")
        or metadata.get("id")
    )

    if not chunk_id:
        raise ValueError(
            f"Chunk tanpa chunk_id: {text[:60]!r}"
        )

    chunk_id = str(chunk_id)

    # Menambahkan judul buku secara konsisten.
    metadata.pop("id", None)
    metadata.pop("chunk_id", None)
    metadata["book_title"] = book_title
    metadata["chunk_id"] = chunk_id

    # Memastikan nomor bab berupa integer jika memungkinkan.
    if metadata.get("chapter_number") is not None:
        try:
            metadata["chapter_number"] = int(
                metadata["chapter_number"]
            )
        except (ValueError, TypeError):
            metadata["chapter_number"] = str(
                metadata["chapter_number"]
            )

    # Memastikan semua metadata aman untuk ChromaDB.
    clean_metadata = {
        key: make_chroma_safe(value)
        for key, value in metadata.items()
    }

    return chunk_id, text, clean_metadata


# ============================================================
# LOAD & PREPARE CHUNKS
# ============================================================

def load_book_chunks(book_config):
    """Membaca dan memvalidasi file chunks.json."""

    chunks_path = book_config["chunks_path"]

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"File chunk tidak ditemukan: {chunks_path}"
        )

    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not isinstance(chunks, list):
        raise ValueError(
            f"Isi {chunks_path} harus berupa list JSON."
        )

    prepared = []

    for chunk in chunks:
        chunk_id, text, metadata = normalize_chunk(
            chunk,
            book_config["book_title"],
        )

        # Prefix buku mencegah benturan ID lintas buku.
        vector_id = (
            f"{book_config['id_prefix']}_{chunk_id}"
        )

        # Hash mendeteksi perubahan teks atau metadata.
        # Jika sumber berubah dengan ID yang sama, embedding
        # akan dibuat kembali.
        hash_input = json.dumps(
            {
                "text": text,
                "metadata": metadata,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

        content_hash = hashlib.sha256(
            hash_input.encode("utf-8")
        ).hexdigest()

        metadata["content_hash"] = content_hash

        prepared.append({
            "vector_id": vector_id,
            "chunk_id": chunk_id,
            "text": text,
            "metadata": metadata,
        })

    return prepared


def load_all_books():
    """Memuat semua buku yang tercantum dalam BOOKS."""

    all_items = []

    for book_config in BOOKS:
        print(
            f"\nMembaca buku: {book_config['book_title']}"
        )

        items = load_book_chunks(book_config)

        print(f"Jumlah chunk: {len(items)}")

        all_items.extend(items)

    if not all_items:
        raise ValueError(
            "Tidak ada chunk yang dapat diproses."
        )

    all_ids = [
        item["vector_id"]
        for item in all_items
    ]

    if len(all_ids) != len(set(all_ids)):
        raise ValueError(
            "Ditemukan ID vector duplikat. "
            "Periksa chunk_id dan id_prefix setiap buku."
        )

    return all_items


# ============================================================
# CHECK EXISTING VECTORS
# ============================================================

def get_existing_metadata(collection, vector_ids):
    """
    Membaca metadata vector yang sudah tersimpan.
    Pengambilan dilakukan bertahap untuk menghindari
    permintaan get dengan daftar ID yang terlalu besar.
    """

    existing = {}

    for start in range(0, len(vector_ids), 200):
        batch_ids = vector_ids[start:start + 200]

        result = collection.get(
            ids=batch_ids,
            include=["metadatas"],
        )

        for vector_id, metadata in zip(
            result["ids"],
            result["metadatas"],
        ):
            existing[vector_id] = metadata or {}

    return existing


def find_pending_items(items, existing_metadata):
    """
    Memilih chunk yang belum ada atau sudah berubah.
    Chunk yang sama persis tidak perlu di-embedding ulang.
    """

    pending = []

    for item in items:
        stored_metadata = existing_metadata.get(
            item["vector_id"]
        )

        if stored_metadata is None:
            pending.append(item)
            continue

        old_hash = stored_metadata.get("content_hash")
        new_hash = item["metadata"]["content_hash"]

        if old_hash != new_hash:
            pending.append(item)

    return pending


# ============================================================
# OPENAI EMBEDDING
# ============================================================

def embed_batch(client, texts):
    """
    Mengirim satu batch teks ke Embeddings API.
    Mengembalikan daftar vector dan token aktual dari API.
    """

    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
            )

            # Memastikan urutan vector sesuai urutan input.
            ordered = sorted(
                response.data,
                key=lambda item: item.index,
            )

            vectors = [
                item.embedding
                for item in ordered
            ]

            actual_tokens = (
                response.usage.total_tokens
                if response.usage
                else 0
            )

            if len(vectors) != len(texts):
                raise RuntimeError(
                    "Jumlah embedding tidak sesuai "
                    "dengan jumlah teks input."
                )

            return vectors, actual_tokens

        except RateLimitError as error:
            message = str(error).lower()
            error_code = str(
                getattr(error, "code", "")
            ).lower()

            # Kredit habis tidak akan selesai dengan retry.
            if (
                "insufficient_quota" in message
                or "credit_balance_exhausted" in message
                or error_code == "insufficient_quota"
            ):
                raise RuntimeError(
                    "Kredit API tidak cukup. "
                    "Periksa saldo API sebelum melanjutkan."
                ) from error

            if attempt == MAX_RETRIES - 1:
                raise

            wait = RETRY_BASE_SECONDS * (2 ** attempt)

            print(
                f"Rate limit. Percobaan ulang dalam "
                f"{wait} detik..."
            )

            time.sleep(wait)

        except APIConnectionError:
            if attempt == MAX_RETRIES - 1:
                raise

            wait = RETRY_BASE_SECONDS * (2 ** attempt)

            print(
                f"Koneksi API bermasalah. "
                f"Mencoba ulang dalam {wait} detik..."
            )

            time.sleep(wait)

        except APIStatusError as error:
            # Retry hanya untuk gangguan server sementara.
            if error.status_code < 500:
                raise

            if attempt == MAX_RETRIES - 1:
                raise

            wait = RETRY_BASE_SECONDS * (2 ** attempt)

            print(
                f"Server API bermasalah. "
                f"Mencoba ulang dalam {wait} detik..."
            )

            time.sleep(wait)

    raise RuntimeError(
        "Embedding gagal setelah beberapa percobaan."
    )


# ============================================================
# MAIN EMBEDDING PROCESS
# ============================================================

def main():

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY tidak ditemukan. "
            "Periksa file .env."
        )

    print("=" * 60)
    print("EMBEDDING BUKU OPENSTAX")
    print("=" * 60)

    # Membaca semua chunk dari konfigurasi buku.
    items = load_all_books()

    # Menyiapkan database lokal dan satu collection.
    chroma = chromadb.PersistentClient(
        path=str(PERSIST_DIRECTORY)
    )

    collection = chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Memeriksa data yang sudah tersimpan.
    vector_ids = [
        item["vector_id"]
        for item in items
    ]

    existing_metadata = get_existing_metadata(
        collection,
        vector_ids,
    )

    pending = find_pending_items(
        items,
        existing_metadata,
    )

    # Estimasi biaya hanya untuk chunk baru atau berubah.
    estimated_tokens = sum(
        count_tokens(item["text"])
        for item in pending
    )

    estimated_cost = estimate_cost(estimated_tokens)

    print("\n" + "=" * 60)
    print("ESTIMASI SEBELUM EMBEDDING")
    print("=" * 60)
    print(f"Total chunk sumber    : {len(items)}")
    print(
        f"Sudah tersimpan      : "
        f"{len(items) - len(pending)}"
    )
    print(f"Perlu di-embedding   : {len(pending)}")
    print(f"Ukuran batch          : {BATCH_SIZE}")
    print(f"Estimasi token        : {estimated_tokens:,}")
    print(f"Estimasi biaya        : ${estimated_cost:.6f} USD")

    if not pending:
        print("\nSemua chunk sudah tersimpan dan tidak berubah.")
        print(f"Total di ChromaDB: {collection.count()}")
        return

    print(
        "\nEstimasi token dapat berbeda dari pemakaian aktual."
    )

    # Meminta konfirmasi sebelum menggunakan kredit API.
    try:
        answer = input(
            "\nLanjutkan embedding? Ketik y untuk melanjutkan "
            "atau n untuk membatalkan: "
        ).strip().lower()
    except EOFError:
        answer = "n"

    if answer not in ("y", "yes", "ya"):
        print("Embedding dibatalkan. Tidak ada request embedding dikirim.")
        return

    # Membuat client setelah konfirmasi.
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    total_actual_tokens = 0
    total_processed = 0

    print("\nMemulai embedding...")

    # Setiap request memproses hingga 100 teks.
    for start in range(0, len(pending), BATCH_SIZE):

        batch = pending[start:start + BATCH_SIZE]

        texts = [
            item["text"]
            for item in batch
        ]

        vectors, actual_tokens = embed_batch(
            client,
            texts,
        )

        total_actual_tokens += actual_tokens

        # Menyimpan vector, teks, ID, dan metadata.
        # Upsert juga memperbarui data bila ID sudah ada.
        collection.upsert(
            ids=[
                item["vector_id"]
                for item in batch
            ],
            embeddings=vectors,
            documents=texts,
            metadatas=[
                item["metadata"]
                for item in batch
            ],
        )

        total_processed += len(batch)

        actual_cost_so_far = estimate_cost(
            total_actual_tokens
        )

        print(
            f"Progress: {total_processed}/{len(pending)} "
            f"| Token aktual: {total_actual_tokens:,} "
            f"| Estimasi biaya aktual: "
            f"${actual_cost_so_far:.6f}"
        )

    print("\n" + "=" * 60)
    print("EMBEDDING SELESAI")
    print("=" * 60)
    print(f"Model                : {EMBEDDING_MODEL}")
    print(f"Collection           : {COLLECTION_NAME}")
    print(f"Database             : {PERSIST_DIRECTORY}")
    print(f"Chunk baru/berubah   : {total_processed}")
    print(f"Token estimasi       : {estimated_tokens:,}")
    print(f"Token aktual API     : {total_actual_tokens:,}")
    print(
        f"Estimasi biaya aktual: "
        f"${estimate_cost(total_actual_tokens):.6f} USD"
    )
    print(f"Total di ChromaDB    : {collection.count()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
