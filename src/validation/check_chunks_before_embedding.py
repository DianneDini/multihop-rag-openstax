
from pathlib import Path
import json
import tiktoken

# Membaca hasil chunking terbaru.
file_path = Path("data/psychology/chunks.json")

with file_path.open("r", encoding="utf-8") as file:
    chunks = json.load(file)

# Menggunakan encoding yang sama dengan chunker.py.
encoding = tiktoken.get_encoding("cl100k_base")


def token_length(text):
    return len(encoding.encode(text))


# ------------------------------------------------------------
# 1. Memeriksa chunk yang sangat pendek.
# ------------------------------------------------------------

short_chunks = [
    chunk
    for chunk in chunks
    if token_length(chunk["text"]) < 50
]

print("=" * 60)
print("PEMERIKSAAN CHUNK PENDEK")
print("=" * 60)
print(f"Jumlah chunk di bawah 50 token: {len(short_chunks)}")

for chunk in short_chunks:
    metadata = chunk["metadata"]

    print("\n" + "-" * 60)
    print("Chunk ID:", chunk["chunk_id"])
    print("Jumlah token:", token_length(chunk["text"]))
    print("Bab:", metadata.get("chapter_number"))
    print("Modul:", metadata.get("module_id"))
    print("Bagian:", metadata.get("section_title"))
    print("Teks:", chunk["text"])


# ------------------------------------------------------------
# 2. Memeriksa chunk glossary.
# ------------------------------------------------------------

glossary_chunks = [
    chunk
    for chunk in chunks
    if chunk["metadata"].get("section_title") == "Glossary"
]

print("\n" + "=" * 60)
print("PEMERIKSAAN CHUNK GLOSSARY")
print("=" * 60)
print(f"Jumlah chunk glossary: {len(glossary_chunks)}")

for chunk in glossary_chunks:
    metadata = chunk["metadata"]

    print("\n" + "-" * 60)
    print("Chunk ID:", chunk["chunk_id"])
    print("Jumlah token:", token_length(chunk["text"]))
    print("Modul:", metadata.get("module_id"))
    print("Bagian:", metadata.get("section_title"))
    print("Teks:", chunk["text"])
