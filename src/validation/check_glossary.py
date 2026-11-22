
from pathlib import Path
import json

# Membaca hasil parsing buku Psychology 2e.
file_path = Path("data/psychology/parsed_records.json")

with file_path.open("r", encoding="utf-8") as file:
    records = json.load(file)

# Istilah yang ingin diperiksa.
terms = [
    "dysgraphia",
    "dyslexia",
    "range of reaction"
]

print("=" * 60)
print("PEMERIKSAAN HASIL PARSING GLOSSARY")
print("=" * 60)

# Memeriksa record yang metadata section-nya adalah Glossary.
for term in terms:
    matches = [
        record
        for record in records
        if record.get("metadata", {}).get("section_title") == "Glossary"
        and term.lower() in record.get("text", "").lower()
    ]

    print(f"\nIstilah: {term}")
    print(f"Jumlah record ditemukan: {len(matches)}")

    for record in matches:
        print("Teks:", record["text"])
        print(
            "Metadata:",
            json.dumps(
                record["metadata"],
                ensure_ascii=False,
                indent=2
            )
        )
