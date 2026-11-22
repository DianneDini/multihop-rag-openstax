
"""
calibrate_threshold.py

Kalibrasi awal ambang cosine similarity untuk Metode Y.

- Menguji beberapa target konsep menggunakan query yang sama
  dengan eksperimen Metode Y.
- Memilih anchor teratas berdasarkan query.
- Mencari partner dari bab dengan jarak minimal 3.
- Menghitung cosine similarity antara anchor dan partner.
- Membandingkan hasil dengan beberapa kandidat ambang.
- Menyimpan hasil ke CSV untuk pemeriksaan manual.

Jalankan dari root project:
    python -m src.retrieval.cross_chapter.calibrate_threshold

Pilih buku:
    python -m src.retrieval.cross_chapter.calibrate_threshold \
        --book-title "Psychology 2e"
"""

import argparse
import csv
import re
from pathlib import Path

from src.retrieval.vector_store import (
    DEFAULT_BOOK_TITLE,
    open_collection,
    open_openai_client,
    check_metadata,
    cosine_similarity,
)

from src.retrieval.cross_chapter.cross_chapter_retriever import (
    MIN_CHAPTER_DISTANCE,
    retrieve_cross_chapter,
)


# ============================================================
# TARGET KONSEP UNTUK PILOT
# ============================================================
# Contoh awal untuk Psychology 2e.
# Sebaiknya ganti dengan 5-10 target konsep dari daftar target
# penelitianmu yang sudah ditetapkan.

TARGET_CONCEPTS = [
    "Chronic Stress",
    "Classical Conditioning",
    "Working Memory",
    "Long-Term Memory",
    "Observational Learning",
    "Sleep and Memory",
    "Emotion and Memory",
    "Fight-or-Flight Response",
]


# Kandidat ambang yang akan diperiksa.
SIMILARITY_THRESHOLDS = [0.40, 0.45, 0.50, 0.70]


# ============================================================
# PEMBANTU
# ============================================================

def format_threshold(value):
    """Mengubah 0.40 menjadi label 0.40 yang aman untuk CSV."""

    return f"{value:.2f}"


def get_chapter(chunk):
    """Mengambil nomor bab jika chunk tersedia."""

    if chunk is None:
        return None

    return chunk.get("chapter_number")


def evaluate_thresholds(chapter_distance, similarity):
    """
    Mengevaluasi pasangan pada setiap kandidat ambang.

    Pasangan valid untuk suatu ambang jika:
    - pasangan ditemukan;
    - jarak nomor bab minimal 3;
    - similarity antarchunk mencapai ambang tersebut.
    """

    results = {}

    for threshold in SIMILARITY_THRESHOLDS:
        label = format_threshold(threshold)

        valid = (
            similarity is not None
            and chapter_distance is not None
            and chapter_distance >= MIN_CHAPTER_DISTANCE
            and similarity >= threshold
        )

        results[f"valid_at_{label}"] = valid

    return results


# ============================================================
# KALIBRASI
# ============================================================

def calibrate(book_title, output_path):
    """Menjalankan retrieval untuk seluruh target konsep pilot."""

    collection = open_collection()

    total_book_chunks = check_metadata(
        collection,
        book_title,
    )

    client = open_openai_client()

    rows = []

    print("=" * 70)
    print("KALIBRASI COSINE SIMILARITY")
    print("=" * 70)
    print(f"Buku              : {book_title}")
    print(f"Jumlah chunk buku : {total_book_chunks}")
    print(f"Jumlah target     : {len(TARGET_CONCEPTS)}")
    print(f"Jarak bab minimum : {MIN_CHAPTER_DISTANCE}")
    print(f"Kandidat ambang   : {SIMILARITY_THRESHOLDS}")
    print("=" * 70)

    for index, query in enumerate(TARGET_CONCEPTS, start=1):
        print(f"\n[{index}/{len(TARGET_CONCEPTS)}] Target: {query}")

        row = {
            "target_concept": query,
            "book_title": book_title,
            "status": "ok",
            "anchor_vector_id": "",
            "anchor_chapter": "",
            "anchor_query_similarity": "",
            "partner_vector_id": "",
            "partner_chapter": "",
            "partner_query_similarity": "",
            "chapter_distance": "",
            "cosine_similarity_pair": "",
            "manual_relevance": "",
            "manual_notes": "",
            "anchor_text": "",
            "partner_text": "",
            "error": "",
        }

        try:
            # Gunakan fungsi Metode Y yang sama dengan eksperimen.
            anchor, partner, _ = retrieve_cross_chapter(
                client=client,
                collection=collection,
                query=query,
                book_title=book_title,
            )

            if anchor is None:
                row["status"] = "no_anchor"

            else:
                row["anchor_vector_id"] = anchor["vector_id"]
                row["anchor_chapter"] = anchor["chapter_number"]
                row["anchor_query_similarity"] = round(
                    anchor["query_similarity"], 4
                )
                row["anchor_text"] = anchor["text"]

            if partner is None:
                row["status"] = "no_partner"
                print("  Partner tidak ditemukan.")

            else:
                chapter_distance = abs(
                    anchor["chapter_number"]
                    - partner["chapter_number"]
                )

                pair_similarity = cosine_similarity(
                    anchor["embedding"],
                    partner["embedding"],
                )

                row["partner_vector_id"] = partner["vector_id"]
                row["partner_chapter"] = partner["chapter_number"]
                row["partner_query_similarity"] = round(
                    partner["query_similarity"], 4
                )
                row["chapter_distance"] = chapter_distance
                row["cosine_similarity_pair"] = round(
                    pair_similarity, 4
                )
                row["partner_text"] = partner["text"]

                # Menilai pasangan pada setiap kandidat ambang.
                row.update(
                    evaluate_thresholds(
                        chapter_distance,
                        pair_similarity,
                    )
                )

                print(
                    f"  Bab anchor/partner : "
                    f"{anchor['chapter_number']} / "
                    f"{partner['chapter_number']}"
                )
                print(
                    f"  Jarak bab          : {chapter_distance}"
                )
                print(
                    f"  Similarity A-B     : {pair_similarity:.4f}"
                )

                for threshold in SIMILARITY_THRESHOLDS:
                    label = format_threshold(threshold)
                    print(
                        f"  Valid pada {label}      : "
                        f"{row[f'valid_at_{label}']}"
                    )

                print(
                    "  Pemeriksaan manual : perlu dilakukan"
                )

        except Exception as error:
            row["status"] = "error"
            row["error"] = str(error)
            print(f"  ERROR: {error}")

        rows.append(row)

    # --------------------------------------------------------
    # SIMPAN KE CSV
    # --------------------------------------------------------

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_fields = [
        "target_concept",
        "book_title",
        "status",
        "anchor_vector_id",
        "anchor_chapter",
        "anchor_query_similarity",
        "partner_vector_id",
        "partner_chapter",
        "partner_query_similarity",
        "chapter_distance",
        "cosine_similarity_pair",
    ]

    threshold_fields = [
        f"valid_at_{format_threshold(value)}"
        for value in SIMILARITY_THRESHOLDS
    ]

    remaining_fields = [
        "manual_relevance",
        "manual_notes",
        "anchor_text",
        "partner_text",
        "error",
    ]

    fieldnames = (
        base_fields
        + threshold_fields
        + remaining_fields
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 70)
    print("KALIBRASI SELESAI")
    print("=" * 70)
    print(f"Target diuji : {len(rows)}")
    print(f"File hasil   : {output_path.resolve()}")
    print(
        "\nBuka CSV untuk membaca pasangan dan mengisi "
        "manual_relevance serta manual_notes."
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pilot kalibrasi cosine similarity Metode Y."
    )

    parser.add_argument(
        "--book-title",
        default=DEFAULT_BOOK_TITLE,
        help="Nama buku sesuai metadata ChromaDB.",
    )

    parser.add_argument(
        "--output",
        default="data/psychology/calibration_results.csv",
        help="Lokasi file CSV hasil kalibrasi.",
    )

    args = parser.parse_args()

    if not TARGET_CONCEPTS:
        raise ValueError("TARGET_CONCEPTS belum diisi.")

    calibrate(
        book_title=args.book_title,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
