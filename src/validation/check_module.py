
import json
import random
from collections import Counter
from pathlib import Path

import tiktoken


# ============================================================
# CONFIGURATION
# ============================================================

# File hasil chunking yang akan divalidasi.
CHUNKS_FILE = Path("data/psychology/chunks.json")

# Batas token sesuai pengaturan chunker.py.
MAX_TOKENS = 512

# Jumlah chunk acak yang ditampilkan untuk pemeriksaan manual.
SAMPLE_SIZE = 5

# Seed agar sampel yang dipilih tetap sama saat script diulang.
RANDOM_SEED = 42


# ============================================================
# TOKENIZER
# ============================================================

# Menggunakan tokenizer yang sama dengan chunker.py.
encoding = tiktoken.get_encoding("cl100k_base")


def token_length(text):
    """Menghitung jumlah token pada teks."""
    return len(encoding.encode(text))


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Membaca seluruh chunk dari file JSON.
    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        print("File chunks.json kosong.")
        raise SystemExit(1)

    # --------------------------------------------------------
    # 1. TOTAL CHUNK DAN PANJANG TOKEN
    # --------------------------------------------------------

    token_counts = [
        token_length(chunk.get("text", ""))
        for chunk in chunks
    ]

    print("=" * 60)
    print("VALIDASI HASIL CHUNKING PSYCHOLOGY 2e")
    print("=" * 60)

    print(f"Total chunk              : {len(chunks)}")
    print(f"Token terpendek          : {min(token_counts)}")
    print(f"Token terpanjang         : {max(token_counts)}")
    print(
        f"Chunk melebihi {MAX_TOKENS} token : "
        f"{sum(n > MAX_TOKENS for n in token_counts)}"
    )

    # --------------------------------------------------------
    # 2. JUMLAH CHUNK PER BAB
    # --------------------------------------------------------

    chapter_counts = Counter(
        chunk.get("metadata", {}).get("chapter_number")
        for chunk in chunks
    )

    print("\nJUMLAH CHUNK PER BAB")

    for chapter in sorted(
        chapter_counts,
        key=lambda value: (
            value is None,
            value if value is not None else 0
        )
    ):
        print(f"Bab {chapter}: {chapter_counts[chapter]} chunk")

    # --------------------------------------------------------
    # 3. VALIDASI METADATA
    # --------------------------------------------------------

    # Nama field disesuaikan dengan struktur chunks.json saat ini.
    required_fields = [
        "book",
        "chapter_number",
        "chapter_title",
        "module_id",
        "module_title",
        "section_title",
        "source_file",
    ]

    missing_metadata = []

    for chunk in chunks:
        metadata = chunk.get("metadata", {})

        missing = [
            field
            for field in required_fields
            if metadata.get(field) is None
            or metadata.get(field) == ""
        ]

        # chunk_id berada di tingkat utama objek chunk.
        if not chunk.get("chunk_id"):
            missing.append("chunk_id")

        if not chunk.get("text", "").strip():
            missing.append("text")

        if missing:
            missing_metadata.append(
                (chunk.get("chunk_id", "TANPA ID"), missing)
            )

    print("\nVALIDASI METADATA")
    print(f"Chunk dengan field kosong: {len(missing_metadata)}")

    for chunk_id, missing in missing_metadata[:10]:
        print(f"{chunk_id}: {', '.join(missing)}")

    # --------------------------------------------------------
    # 4. VALIDASI ID DUPLIKAT
    # --------------------------------------------------------

    chunk_ids = [chunk.get("chunk_id") for chunk in chunks]
    id_counts = Counter(chunk_ids)

    duplicate_ids = [
        chunk_id
        for chunk_id, count in id_counts.items()
        if chunk_id is None or count > 1
    ]

    print(f"\nJumlah ID duplikat/kosong: {len(duplicate_ids)}")

    if duplicate_ids:
        print("Contoh ID bermasalah:", duplicate_ids[:10])

    # --------------------------------------------------------
    # 5. LIMA CHUNK ACAK UNTUK PEMERIKSAAN MANUAL
    # --------------------------------------------------------

    random.seed(RANDOM_SEED)
    samples = random.sample(
        chunks,
        min(SAMPLE_SIZE, len(chunks))
    )

    print("\nLIMA CHUNK ACAK UNTUK DIPERIKSA MANUAL")

    for index, chunk in enumerate(samples, start=1):
        metadata = chunk.get("metadata", {})

        print("\n" + "-" * 60)
        print(f"Sampel       : {index}")
        print(f"Chunk ID     : {chunk.get('chunk_id')}")
        print(f"Jumlah token : {token_length(chunk.get('text', ''))}")
        print(f"Bab          : {metadata.get('chapter_number')}")
        print(f"Judul bab    : {metadata.get('chapter_title')}")
        print(f"Modul        : {metadata.get('module_id')}")
        print(f"Bagian       : {metadata.get('section_title')}")
        print("Teks:")
        print(chunk.get("text", ""))

    print("\n" + "=" * 60)
    print("VALIDASI OTOMATIS SELESAI")
    print("Sampel teks tetap perlu diperiksa secara manual.")
    print("=" * 60)
