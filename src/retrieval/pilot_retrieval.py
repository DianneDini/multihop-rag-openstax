
"""
pilot_retrieval.py

Jalankan dari root project:
    python -m src.retrieval.pilot_retrieval
"""

import csv
import json
from pathlib import Path

from src.retrieval.vector_store import (
    open_collection,
    open_openai_client,
    check_metadata,
)
from src.retrieval.test_retrieval import (
    retrieve_conventional,
    evaluate_pair as evaluate_x,
)
from src.retrieval.cross_chapter.cross_chapter_retriever import (
    retrieve_cross_chapter,
    evaluate_pair as evaluate_y,
)

CONCEPTS_PATH = Path("data/psychology/target_concepts.json")
OUTPUT_PATH = Path("data/psychology/pilot_retrieval_results.csv")


def get_field(chunk, field):
    if chunk is None:
        return ""
    if field == "vector_id":
        return chunk.get("vector_id", "")
    if field == "text":
        return chunk.get("text", "")
    if field in ("chapter_number", "query_similarity"):
        value = chunk.get(field)
        if value is None:
            return ""
        return round(float(value), 6) if field == "query_similarity" else value
    return chunk.get("metadata", {}).get(field, "")


def make_row(concept, method, a, b, evaluation, tokens):
    evaluation = evaluation or {}

    return {
        "target_concept": concept["target_concept"],
        "book_title": concept.get("book_title", "Psychology 2e"),
        "method": method,
        "chunk_a_id": get_field(a, "vector_id"),
        "chunk_a_chapter": get_field(a, "chapter_number"),
        "chunk_a_chapter_title": get_field(a, "chapter_title"),
        "chunk_a_module": get_field(a, "module_title"),
        "chunk_a_section": get_field(a, "section_title"),
        "chunk_a_query_similarity": get_field(a, "query_similarity"),
        "chunk_a_text": get_field(a, "text"),
        "chunk_b_id": get_field(b, "vector_id"),
        "chunk_b_chapter": get_field(b, "chapter_number"),
        "chunk_b_chapter_title": get_field(b, "chapter_title"),
        "chunk_b_module": get_field(b, "module_title"),
        "chunk_b_section": get_field(b, "section_title"),
        "chunk_b_query_similarity": get_field(b, "query_similarity"),
        "chunk_b_text": get_field(b, "text"),
        "chapter_distance": evaluation.get("chapter_distance", ""),
        "pair_cosine_similarity": evaluation.get("pair_similarity", ""),
        "chapter_distance_at_least_3": evaluation.get("chapter_condition", False),
        "pair_similarity_at_least_0_5": evaluation.get("similarity_condition", False),
        "is_valid": evaluation.get("is_valid", False),
        "query_tokens_reported": tokens,
    }


def main():
    if not CONCEPTS_PATH.exists():
        raise FileNotFoundError(
            f"File target konsep tidak ditemukan: {CONCEPTS_PATH}"
        )

    with CONCEPTS_PATH.open("r", encoding="utf-8") as f:
        concepts = json.load(f)

    if not isinstance(concepts, list) or not concepts:
        raise ValueError("File target_concepts.json kosong atau formatnya salah.")

    collection = open_collection()
    client = open_openai_client()
    rows = []

    print("=" * 60, flush=True)
    print("PILOT RETRIEVAL: METODE X DAN METODE Y", flush=True)
    print(f"Jumlah target konsep: {len(concepts)}", flush=True)
    print("=" * 60, flush=True)

    for i, concept in enumerate(concepts, start=1):
        query = concept["target_concept"].strip()
        book = concept.get("book_title", "Psychology 2e").strip()

        print(f"\n[{i}/{len(concepts)}] {query}", flush=True)
        check_metadata(collection, book)

        # Metode X: top-2 tanpa batasan jarak bab.
        chunks_x, tokens_x = retrieve_conventional(
            client, collection, query, book
        )

        if len(chunks_x) >= 2:
            xa, xb = chunks_x[0], chunks_x[1]
            eval_x = evaluate_x(xa, xb)
        else:
            xa = chunks_x[0] if chunks_x else None
            xb, eval_x = None, None

        rows.append(make_row(
            concept, "X - Conventional", xa, xb, eval_x, tokens_x
        ))

        # Metode Y: anchor dan partner dengan jarak bab minimal 3.
        ya, yb, tokens_y = retrieve_cross_chapter(
            client, collection, query, book
        )
        eval_y = evaluate_y(ya, yb)

        rows.append(make_row(
            concept, "Y - Cross-Chapter", ya, yb, eval_y, tokens_y
        ))

        x_sim = "N/A" if eval_x is None else f"{eval_x['pair_similarity']:.4f}"
        y_sim = "N/A" if eval_y is None else f"{eval_y['pair_similarity']:.4f}"

        print(
            f"  Metode X: bab {get_field(xa, 'chapter_number')} dan "
            f"{get_field(xb, 'chapter_number')}; similarity={x_sim}",
            flush=True,
        )
        print(
            f"  Metode Y: bab {get_field(ya, 'chapter_number')} dan "
            f"{get_field(yb, 'chapter_number')}; similarity={y_sim}",
            flush=True,
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    valid_x = sum(
        row["is_valid"] is True
        for row in rows
        if row["method"] == "X - Conventional"
    )
    valid_y = sum(
        row["is_valid"] is True
        for row in rows
        if row["method"] == "Y - Cross-Chapter"
    )

    print("\nPILOT SELESAI", flush=True)
    print(f"Jumlah baris CSV: {len(rows)}", flush=True)
    print(f"Pasangan valid Metode X: {valid_x}", flush=True)
    print(f"Pasangan valid Metode Y: {valid_y}", flush=True)
    print(f"File hasil: {OUTPUT_PATH.resolve()}", flush=True)


if __name__ == "__main__":
    main()
