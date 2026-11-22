"""
test_prompt_iteration.py

Uji coba satu versi prompt pada beberapa target konsep sebelum
eksperimen penuh (iterasi prompt Fase 3).

Untuk setiap target konsep:
    retrieval lintas bab -> evaluasi pasangan -> Non-CoT dan CoT

Output (per versi prompt, agar iterasi dapat dibandingkan):
    data/psychology/prompt_iterations/<versi>/<konsep>.json
    data/psychology/prompt_iterations/<versi>/summary.csv
    data/psychology/prompt_iterations/<versi>/review.md

review.md berisi soal, kunci jawaban, dan penalaran CoT dalam
format yang mudah dibaca untuk review manual.

Jalankan dari root project:
    python -m src.generation.test_prompt_iteration
    python -m src.generation.test_prompt_iteration \
        --concepts "Stress" "Memory" --book "Psychology 2e"
"""

import argparse
import csv
import os
import re
import time
from pathlib import Path

from google import genai

from src.retrieval.vector_store import (
    DEFAULT_BOOK_TITLE,
    open_collection,
    open_openai_client,
)
from src.generation.generator import (
    PROMPT_VERSION,
    VARIANTS,
    describe_settings,
    run_pipeline,
    save_json,
)


DEFAULT_CONCEPTS = [
    "Naturalistic Observation",
    "Natural Selection",
    "Endocrine System",
    "Insomnia",
    "Cognition",
]

# Jeda antar target konsep untuk menjaga batas rate free tier.
DELAY_BETWEEN_CONCEPTS = float(os.getenv("ITERATION_DELAY", "10"))

OUTPUT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "psychology"
    / "prompt_iterations"
)

SUMMARY_FIELDS = [
    "concept",
    "variant",
    "skipped_reason",
    "chapter_a",
    "chapter_b",
    "chapter_distance",
    "pair_similarity",
    "partner_query_percentile",
    "is_valid",
    "is_relevant",
    "status",
    "question_word_count",
    "connection_type",
    "quality_flags",
    "output_tokens",
    "thinking_tokens",
    "error",
    "question",
]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def summary_rows(concept: str, output: dict) -> list:
    retrieval = output["retrieval"]
    pair = retrieval.get("pair_evaluation") or {}
    source_a = retrieval.get("source_a") or {}
    source_b = retrieval.get("source_b") or {}

    base = {
        "concept": concept,
        "skipped_reason": output.get("skipped_reason") or "",
        "chapter_a": source_a.get("chapter_number"),
        "chapter_b": source_b.get("chapter_number"),
        "chapter_distance": pair.get("chapter_distance"),
        "pair_similarity": (
            round(pair["pair_similarity"], 4)
            if pair.get("pair_similarity") is not None else None
        ),
        "partner_query_percentile": (
            round(pair["partner_query_percentile"], 1)
            if pair.get("partner_query_percentile") is not None else None
        ),
        "is_valid": pair.get("is_valid"),
        "is_relevant": pair.get("is_relevant"),
    }

    if not output["results"]:
        return [{**base, "variant": "-"}]

    rows = []

    for result in output["results"]:
        usage = result.get("usage") or {}
        reasoning = result.get("reasoning") or {}

        rows.append({
            **base,
            "variant": result["variant"],
            "status": result["status"],
            "question_word_count": result.get("question_word_count"),
            "connection_type": reasoning.get("connection_type", ""),
            "quality_flags": ";".join(result.get("quality_flags") or []),
            "output_tokens": usage.get("output_tokens"),
            "thinking_tokens": usage.get("thinking_tokens"),
            "error": result.get("error") or "",
            "question": result.get("question", ""),
        })

    return rows


def review_section(concept: str, output: dict) -> str:
    retrieval = output["retrieval"]
    pair = retrieval.get("pair_evaluation") or {}
    a = retrieval.get("source_a") or {}
    b = retrieval.get("source_b") or {}

    lines = [
        f"## {concept}",
        "",
        f"- Chunk A: Bab {a.get('chapter_number')} - {a.get('section')}",
        f"- Chunk B: Bab {b.get('chapter_number')} - {b.get('section')}",
        f"- Similarity A-B: {pair.get('pair_similarity')}",
        f"- is_valid: {pair.get('is_valid')} | "
        f"is_relevant: {pair.get('is_relevant')}",
        "",
    ]

    if output.get("skipped_reason"):
        lines += [f"**Dilewati:** {output['skipped_reason']}", ""]
        return "\n".join(lines)

    for result in output["results"]:
        lines += [f"### {result['variant'].upper()} ({result['status']})", ""]

        if result.get("error"):
            lines += [f"Error: {result['error']}", ""]
            continue

        if result.get("reasoning"):
            lines.append("**Penalaran:**")
            for key, value in result["reasoning"].items():
                lines.append(f"- *{key}*: {value}")
            lines.append("")

        lines += [
            f"**Soal:** {result['question']}",
            "",
            f"**Kunci jawaban:** {result['answer_key']}",
            "",
        ]

        if result.get("quality_flags"):
            lines += [f"**Flags:** {', '.join(result['quality_flags'])}", ""]

        lines += [
            "**Review manual:** multi-hop (butuh A dan B)? __ | "
            "hubungan didukung teks? __ | halusinasi? __ | catatan: __",
            "",
        ]

    return "\n".join(lines)


def print_overview(rows: list):
    generated = [r for r in rows if r.get("status")]

    print("\n" + "=" * 60)
    print("RINGKASAN")
    print("=" * 60)

    for variant in VARIANTS:
        subset = [r for r in generated if r["variant"] == variant]
        success = [r for r in subset if r["status"] == "success"]
        flagged = [r for r in success if r["quality_flags"]]

        print(
            f"{variant:<8}: {len(success)}/{len(subset)} success, "
            f"{len([r for r in subset if r['status'] == 'insufficient_sources'])} "
            f"insufficient, {len(flagged)} ber-flag"
        )

    skipped = {r["concept"] for r in rows if r["skipped_reason"]}
    if skipped:
        print(f"Dilewati : {sorted(skipped)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Uji coba versi prompt pada beberapa target konsep."
    )
    parser.add_argument("--concepts", nargs="+", default=DEFAULT_CONCEPTS)
    parser.add_argument("--book", default=DEFAULT_BOOK_TITLE)
    parser.add_argument("--require-relevant", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY belum tersedia. Periksa file .env.")

    out_dir = OUTPUT_ROOT / PROMPT_VERSION
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"UJI ITERASI PROMPT {PROMPT_VERSION}")
    print("=" * 60)
    print(f"Pengaturan : {describe_settings()}")
    print(f"Buku       : {args.book}")
    print(f"Konsep     : {args.concepts}")

    collection = open_collection()
    openai_client = open_openai_client()
    gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    rows = []
    sections = [
        f"# Review prompt {PROMPT_VERSION}",
        "",
        f"Pengaturan: `{describe_settings()}`",
        "",
    ]

    for index, concept in enumerate(args.concepts):
        print(f"\n### [{index + 1}/{len(args.concepts)}] {concept}")

        try:
            output = run_pipeline(
                query=concept,
                book_title=args.book,
                require_relevant=args.require_relevant,
                collection=collection,
                openai_client=openai_client,
                gemini_client=gemini_client,
                verbose=True,
            )
        except Exception as exc:  # retrieval gagal, lanjut konsep lain
            print(f"GAGAL: {exc}")
            rows.append({
                "concept": concept,
                "variant": "-",
                "skipped_reason": "error",
                "error": str(exc),
            })
            continue

        if output["skipped_reason"]:
            print(f"Dilewati: {output['skipped_reason']}")

        save_json(output, out_dir / f"{slugify(concept)}.json")
        rows.extend(summary_rows(concept, output))
        sections.append(review_section(concept, output))

        if index < len(args.concepts) - 1:
            time.sleep(DELAY_BETWEEN_CONCEPTS)

    with (out_dir / "summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=SUMMARY_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "review.md").write_text(
        "\n".join(sections), encoding="utf-8"
    )

    print_overview(rows)
    print(f"\nHasil disimpan di: {out_dir.resolve()}")
    print("Baca review.md untuk review manual setiap soal.")


if __name__ == "__main__":
    main()