"""
generator.py

Fase 3: Pembangkitan soal multi-hop dari dua chunk lintas bab
dengan dua varian prompting (Skenario Eksperimen 2):

    non_cot : prompting langsung (baseline)
    cot     : Chain-of-Thought eksplisit. Model menuliskan langkah
              penalaran (konsep, informasi unik, hubungan, rancangan
              soal, uji satu sumber) di field JSON SEBELUM soal dan
              kunci jawaban.

Kedua varian memakai pasangan chunk, model, temperature, konfigurasi
thinking, dan aturan soal yang sama. Satu-satunya perbedaan adalah
ada/tidaknya langkah penalaran bertahap.

Riwayat versi prompt:
    v3 : CoT implisit (penalaran tidak ditulis)
    v4 : CoT eksplisit, aturan anti-halusinasi relasional
    v5 : aturan anti-penjajaran fakta, jenis hubungan spesifik,
         uji satu sumber (single_source_check)
    Few-shot tersedia sebagai opsi (USE_FEW_SHOT=true) tetapi
    NONAKTIF secara default dan belum menjadi bagian v5.

Jalankan dari root project:
    python -m src.generation.generator --query "Stress" --book "Psychology 2e"
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, ValidationError

from src.retrieval.vector_store import (
    DEFAULT_BOOK_TITLE,
    open_collection,
    open_openai_client,
)
from src.retrieval.cross_chapter.cross_chapter_retriever import (
    DEFAULT_QUERY,
    retrieve_cross_chapter,
    evaluate_pair,
)


# ============================================================
# 1. KONFIGURASI
# ============================================================

load_dotenv()


def _env(name):
    """Nilai .env; None jika tidak ada atau kosong."""

    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


GENERATOR_MODEL = _env("GENERATOR_MODEL") or "gemini-3.5-flash-lite"
QUESTION_LANGUAGE = _env("QUESTION_LANGUAGE") or "English"
GENERATION_TEMPERATURE = _env("GENERATION_TEMPERATURE")

# Isi SALAH SATU di .env (lihat dokumentasi resmi Gemini API).
THINKING_BUDGET = _env("THINKING_BUDGET")   # Gemini 2.5, contoh: 0
THINKING_LEVEL = _env("THINKING_LEVEL")     # Gemini 3.x, contoh: minimal

# Contoh few-shot (dipakai KEDUA varian jika aktif). Default NONAKTIF;
# aktifkan dengan USE_FEW_SHOT=true hanya jika diputuskan menjadi
# bagian desain eksperimen dan dijelaskan di Bab 3.
USE_FEW_SHOT = (_env("USE_FEW_SHOT") or "false").lower() == "true"

# Retry untuk batas rate free tier (HTTP 429 / RESOURCE_EXHAUSTED).
MAX_RETRIES = int(_env("GENERATION_MAX_RETRIES") or 3)
RETRY_BASE_DELAY = float(_env("GENERATION_RETRY_DELAY") or 20)

PROMPT_VERSION = "v5"

MAX_QUESTION_WORDS = 40

OUTPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "psychology"
    / "generated_questions_sample.json"
)

VARIANTS = ("non_cot", "cot")


# ============================================================
# 2. SKEMA OUTPUT
# ============================================================

class NonCoTQuestion(BaseModel):
    status: Literal["success", "insufficient_sources"]
    question: str
    answer_key: str


CONNECTION_TYPES = (
    "qualifies",
    "explains the mechanism of",
    "is an example or application of",
    "contrasts with",
    "extends",
)

# "none" hanya untuk status insufficient_sources (tidak ada hubungan).
CONNECTION_TYPE_OPTIONS = CONNECTION_TYPES + ("none",)


class CoTQuestion(BaseModel):
    # Urutan field disengaja: penalaran ditulis sebelum soal final.
    status: Literal["success", "insufficient_sources"]
    concept_a: str
    concept_b: str
    unique_to_a: str
    unique_to_b: str
    connection_type: Literal[CONNECTION_TYPE_OPTIONS]
    connection: str
    question_design: str
    single_source_check: str
    question: str
    answer_key: str


SCHEMAS = {
    "non_cot": NonCoTQuestion,
    "cot": CoTQuestion,
}

REASONING_FIELDS = (
    "concept_a",
    "concept_b",
    "unique_to_a",
    "unique_to_b",
    "connection_type",
    "connection",
    "question_design",
    "single_source_check",
)


class GenerationError(Exception):
    """Error saat proses pembangkitan soal."""


# ============================================================
# 3. ATURAN KUALITAS SOAL (SAMA UNTUK KEDUA VARIAN)
# ============================================================

def build_shared_rules(language: str) -> str:
    types = ", ".join(f'"{t}"' for t in CONNECTION_TYPES)

    return f"""
GENERAL RULES

1. Write all output fields in {language}.

2. Generate exactly one open-ended analytical question.

3. The question must require information from BOTH sources.
   It must not be fully answerable from either source alone.
   Every part of the question must depend on the combination,
   not on one source with a detail from the other attached.

4. The question must require analysis, explanation, comparison,
   application, or evaluation rather than simple recall.

5. Each source must contribute information that the other source
   does not contain. If one source only repeats the other, return
   status "insufficient_sources".

6. The relationship between the sources must be specific: one
   source {types} the other. Sharing a topic is NOT a relationship.
   If the only link is a shared topic, return
   status "insufficient_sources".

7. Do not merely place two facts side by side. The answer key must
   state a conclusion that cannot be reached from either source
   alone, and show how both sources lead to it.

8. Use only facts, terms, and examples that appear in the sources.
   Do not introduce concepts or terminology from outside the sources.
   Do not exaggerate what the sources say.

9. Do not state or imply that one concept causes, explains, or
   leads to another unless the sources explicitly support that
   relationship.

10. The question must be self-contained, at most {MAX_QUESTION_WORDS}
    words, and understandable without reading the sources.
    Do not mention "Source A", "Source B", "the text", or "the passage".
    Do not tell the learner how to structure the answer or which
    ideas to combine.

11. Source labels may be used in the answer key only.

12. If the sources do not support a coherent analytical question
    requiring both sources, return:
    status = "insufficient_sources"
    question = ""
    answer_key = ""

13. If status is "success", question and answer_key must both
    contain meaningful text.

14. Return only the structured JSON output required by the schema.
"""


# ============================================================
# 4. CONTOH FEW-SHOT (SAMA UNTUK KEDUA VARIAN)
# ============================================================
# Topik sengaja di luar Psychology 2e dan Concepts of Biology agar
# model tidak menyalin isi contoh ke soal dari dataset penelitian.

FEW_SHOT_SOURCES = """
EXAMPLE SOURCE A
Chapter: 2
Section: Scarcity and Choice

Every choice involves an opportunity cost: the value of the best
alternative that is given up. For example, an hour spent working at
a part-time job is an hour that cannot be spent studying.

EXAMPLE SOURCE B
Chapter: 9
Section: Labor Markets

When the wage paid for an hour of work increases, workers earn more
for each hour they give to their job. Many workers respond to higher
wages by choosing to work more hours.
"""

FEW_SHOT_FINAL = {
    "status": "success",
    "question": (
        "If a student's hourly wage at a part-time job rises, why does "
        "an hour of studying become more costly, and how might this "
        "change the way the student divides their time?"
    ),
    "answer_key": (
        "Source A defines opportunity cost as the value of the best "
        "alternative given up, and gives working as the alternative to "
        "studying. Source B states that a higher wage increases what is "
        "earned per hour and that many workers respond by working more "
        "hours. Combining both: because the alternative given up is now "
        "worth more, each hour of studying costs the student more in "
        "forgone earnings, so the student may shift some time from "
        "studying to working. Neither source alone reaches this "
        "conclusion."
    ),
}

FEW_SHOT_REASONING = {
    "concept_a": "Opportunity cost: the value of the best alternative "
                 "given up; working is the alternative to studying.",
    "concept_b": "Higher wages increase earnings per hour, and many "
                 "workers respond by working more hours.",
    "unique_to_a": "The definition of opportunity cost and the example "
                   "that working is the alternative to studying.",
    "unique_to_b": "The effect of a wage increase on earnings per hour "
                   "and on hours worked.",
    "connection_type": "extends",
    "connection": "Source B describes a change (a higher wage) in the "
                  "value of the alternative that Source A uses to "
                  "define the opportunity cost of studying.",
    "question_design": "The learner must apply the definition from A to "
                       "the wage change in B to infer that studying "
                       "becomes more costly. Bloom level: apply/analyze.",
    "single_source_check": "A alone does not mention wage changes; B "
                           "alone does not mention studying or "
                           "opportunity cost. The question cannot be "
                           "answered from either source alone.",
}


def build_few_shot_block(variant: str) -> str:
    if not USE_FEW_SHOT:
        return ""

    if variant == "cot":
        example = {
            "status": FEW_SHOT_FINAL["status"],
            **FEW_SHOT_REASONING,
            "question": FEW_SHOT_FINAL["question"],
            "answer_key": FEW_SHOT_FINAL["answer_key"],
        }
    else:
        example = FEW_SHOT_FINAL

    return f"""
EXAMPLE (different subject; shows the expected quality only.
Do not reuse its topic or wording.)
{FEW_SHOT_SOURCES}
EXAMPLE OUTPUT
{json.dumps(example, indent=2)}
"""


# ============================================================
# 5. SUMBER INPUT (SAMA UNTUK KEDUA VARIAN)
# ============================================================

def get_chunk_text(chunk: dict) -> str:
    text = chunk.get("text", "")

    if not isinstance(text, str) or not text.strip():
        raise GenerationError("Chunk tidak memiliki teks yang valid.")

    return text.strip()


def build_sources_block(chunk_a: dict, chunk_b: dict) -> str:
    text_a = get_chunk_text(chunk_a)
    text_b = get_chunk_text(chunk_b)

    return f"""
SOURCE A
Chapter: {chunk_a.get("chapter_number", "Unknown")}
Section: {chunk_a.get("metadata", {}).get("section_title", "Unknown")}

{text_a}


SOURCE B
Chapter: {chunk_b.get("chapter_number", "Unknown")}
Section: {chunk_b.get("metadata", {}).get("section_title", "Unknown")}

{text_b}
"""


# ============================================================
# 6. PROMPT NON-COT
# ============================================================

def build_non_cot_prompt(chunk_a: dict, chunk_b: dict, language: str) -> str:
    return f"""
You are an educational question-generation assistant.

Your task is to generate one high-quality multi-hop analytical
question using the two supplied textbook excerpts.

Read both sources and directly produce the final question and
its answer key.

OUTPUT FIELDS
- status
- question
- answer_key

{build_shared_rules(language)}
{build_few_shot_block("non_cot")}
NOW THE ACTUAL SOURCES
{build_sources_block(chunk_a, chunk_b)}

Generate the final output now.
"""


# ============================================================
# 7. PROMPT COT (PENALARAN EKSPLISIT)
# ============================================================

def build_cot_prompt(chunk_a: dict, chunk_b: dict, language: str) -> str:
    types = ", ".join(f'"{t}"' for t in CONNECTION_TYPES)

    return f"""
You are an educational question-generation assistant.

Your task is to generate one high-quality multi-hop analytical
question using the two supplied textbook excerpts.

Work through the following steps IN ORDER and write each step in
its JSON field before writing the final question and answer key.

Step 1 - Identify key concepts
  concept_a: the key concept in Source A and its supporting facts.
  concept_b: the key concept in Source B and its supporting facts.

Step 2 - Identify unique contributions
  unique_to_a: information needed for the question that appears
               ONLY in Source A.
  unique_to_b: information needed for the question that appears
               ONLY in Source B.
  If either is empty, return status "insufficient_sources".

Step 3 - Determine the relationship
  connection_type: one of {types}, or "none".
  connection: how Source B relates to Source A (or the reverse),
  stated only as far as both sources explicitly support it.
  "Both sources discuss X" is a shared topic, not a relationship;
  in that case set connection_type "none" and return status
  "insufficient_sources".

Step 4 - Design the question
  question_design: a draft question, how it forces the learner to
  use unique_to_a AND unique_to_b, and the targeted Bloom level
  (apply, analyze, evaluate, or create).

Step 5 - Check the draft
  single_source_check: try to answer the draft using ONLY Source A,
  then ONLY Source B. State what is missing each time. If either
  source alone is enough, revise the draft before writing the
  final question.

Step 6 - Write the final output
  question: the final (revised) question.
  answer_key: the answer key, ending with the conclusion that only
  the combination of both sources supports.

{build_shared_rules(language)}
{build_few_shot_block("cot")}
NOW THE ACTUAL SOURCES
{build_sources_block(chunk_a, chunk_b)}

Generate the output now.
"""


PROMPT_BUILDERS = {
    "non_cot": build_non_cot_prompt,
    "cot": build_cot_prompt,
}


# ============================================================
# 8. KONFIGURASI GENERASI (SAMA UNTUK KEDUA VARIAN)
# ============================================================

def get_temperature():
    if GENERATION_TEMPERATURE is None:
        return None

    try:
        return float(GENERATION_TEMPERATURE)
    except ValueError as exc:
        raise GenerationError(
            "GENERATION_TEMPERATURE harus berupa angka."
        ) from exc


def get_thinking_config():
    if THINKING_BUDGET and THINKING_LEVEL:
        raise GenerationError(
            "Isi salah satu saja: THINKING_BUDGET atau THINKING_LEVEL."
        )

    if THINKING_BUDGET:
        try:
            return {"thinking_budget": int(THINKING_BUDGET)}
        except ValueError as exc:
            raise GenerationError(
                "THINKING_BUDGET harus bilangan bulat."
            ) from exc

    if THINKING_LEVEL:
        return {"thinking_level": THINKING_LEVEL}

    return None


def build_generation_config(variant: str) -> dict:
    config = {
        "response_mime_type": "application/json",
        "response_schema": SCHEMAS[variant],
    }

    temperature = get_temperature()
    if temperature is not None:
        config["temperature"] = temperature

    thinking = get_thinking_config()
    if thinking is not None:
        config["thinking_config"] = thinking

    return config


def describe_settings() -> dict:
    return {
        "model": GENERATOR_MODEL,
        "prompt_version": PROMPT_VERSION,
        "language": QUESTION_LANGUAGE,
        "temperature": get_temperature(),
        "thinking_config": get_thinking_config(),
        "few_shot": USE_FEW_SHOT,
        "max_question_words": MAX_QUESTION_WORDS,
    }


# ============================================================
# 9. VALIDASI DAN PEMERIKSAAN KUALITAS
# ============================================================

def validate_generated_question(result, variant: str):
    """Validasi struktural; gagal -> GenerationError."""

    if result.status == "success":
        required = ["question", "answer_key"]

        if variant == "cot":
            required += list(REASONING_FIELDS)

        empty = [f for f in required if not getattr(result, f).strip()]

        if empty:
            raise GenerationError(
                f"Status success tetapi field kosong: {empty}"
            )

    elif result.question.strip() or result.answer_key.strip():
        raise GenerationError(
            "Untuk insufficient_sources, question dan answer_key "
            "harus kosong."
        )

    return result


SOURCE_LABEL_PATTERN = re.compile(
    r"\b(source\s*[ab]|sumber\s*[ab]|the\s+(text|passage|excerpt)s?)\b",
    re.IGNORECASE,
)

ANSWER_HINT_PATTERN = re.compile(
    r"\b(in your answer|your answer should|integrate|combine)\b",
    re.IGNORECASE,
)

# Kata dari contoh few-shot; jika muncul, model menyalin contoh.
FEW_SHOT_LEAK_PATTERN = re.compile(
    r"\b(opportunity cost|wage|part-time job)\b",
    re.IGNORECASE,
)


def check_question_quality(question: str, answer_key: str) -> list:
    """
    Pemeriksaan otomatis sederhana. Soal TIDAK ditolak; flag
    dicatat untuk analisis kelemahan dan validasi manual.
    """

    if not question.strip():
        return []

    flags = []

    if SOURCE_LABEL_PATTERN.search(question):
        flags.append("mentions_source_label")

    if ANSWER_HINT_PATTERN.search(question):
        flags.append("answer_structure_hint")

    if len(question.split()) > MAX_QUESTION_WORDS:
        flags.append("too_long")

    if USE_FEW_SHOT and FEW_SHOT_LEAK_PATTERN.search(
        question + " " + answer_key
    ):
        flags.append("few_shot_leak")

    return flags


# ============================================================
# 10. INFORMASI UNTUK FILE OUTPUT
# ============================================================

def summarize_chunk(chunk: dict) -> dict:
    metadata = chunk.get("metadata", {})

    return {
        "vector_id": chunk.get("vector_id"),
        "chapter_number": chunk.get("chapter_number"),
        "chapter_title": metadata.get("chapter_title"),
        "section": metadata.get("section_title"),
        "query_similarity": chunk.get("query_similarity"),
        "query_percentile": chunk.get("query_percentile"),
        "text": chunk.get("text", ""),
    }


def summarize_pair_evaluation(evaluation: dict) -> dict:
    """Menyimpan seluruh hasil evaluasi retrieval apa adanya."""

    return {
        key: value
        for key, value in evaluation.items()
        if isinstance(value, (bool, int, float, str, dict, type(None)))
    }


def read_usage(response) -> dict:
    usage = getattr(response, "usage_metadata", None)

    return {
        "input_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
        "thinking_tokens": getattr(usage, "thoughts_token_count", None) or 0,
        "total_tokens": getattr(usage, "total_token_count", None),
    }


def get_finish_reason(response):
    candidates = getattr(response, "candidates", None)

    if not candidates or candidates[0].finish_reason is None:
        return None

    reason = candidates[0].finish_reason
    return getattr(reason, "name", str(reason))


# ============================================================
# 11. GENERASI SATU VARIAN
# ============================================================

def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def call_model(client, prompt: str, variant: str):
    """Memanggil Gemini dengan retry untuk batas rate."""

    for attempt in range(MAX_RETRIES + 1):
        try:
            return client.models.generate_content(
                model=GENERATOR_MODEL,
                contents=prompt,
                config=build_generation_config(variant),
            )
        except Exception as exc:
            if not _is_rate_limit(exc) or attempt == MAX_RETRIES:
                raise

            delay = RETRY_BASE_DELAY * (attempt + 1)
            print(f"  Rate limit, menunggu {delay:.0f} detik...")
            time.sleep(delay)


def generate_question(
    client,
    chunk_a: dict,
    chunk_b: dict,
    variant: Literal["non_cot", "cot"],
    query: str,
) -> dict:
    schema = SCHEMAS[variant]
    prompt = PROMPT_BUILDERS[variant](
        chunk_a=chunk_a,
        chunk_b=chunk_b,
        language=QUESTION_LANGUAGE,
    )

    try:
        response = call_model(client, prompt, variant)

        if not response.text:
            raise GenerationError(
                "Model tidak menghasilkan teks respons. "
                f"finish_reason={get_finish_reason(response)}"
            )

        try:
            parsed = schema.model_validate_json(response.text)
        except ValidationError as exc:
            raise GenerationError(
                f"Output tidak sesuai skema: {exc}"
            ) from exc

        parsed = validate_generated_question(parsed, variant)
        data = parsed.model_dump()

        flags = check_question_quality(data["question"], data["answer_key"])

        # Aturan deterministik: jika penalaran CoT sendiri menyatakan
        # tidak ada hubungan, soal dianggap insufficient_sources.
        # Output asli disimpan untuk analisis kepatuhan instruksi.
        overridden_output = None

        if (
            variant == "cot"
            and data["status"] == "success"
            and data["connection_type"] == "none"
        ):
            overridden_output = {
                "status": data["status"],
                "question": data["question"],
                "answer_key": data["answer_key"],
            }
            data["status"] = "insufficient_sources"
            data["question"] = ""
            data["answer_key"] = ""
            flags = ["overridden_connection_none"]

        reasoning = (
            {field: data[field] for field in REASONING_FIELDS}
            if variant == "cot"
            else None
        )

        return {
            "variant": variant,
            "query": query,
            "model_requested": GENERATOR_MODEL,
            "model_version": getattr(response, "model_version", None),
            **describe_settings(),
            # Field yang dinilai judge (posisi sama untuk kedua varian)
            "status": data["status"],
            "question": data["question"],
            "answer_key": data["answer_key"],
            # Penalaran CoT (tidak diberikan ke judge)
            "reasoning": reasoning,
            "overridden_output": overridden_output,
            "quality_flags": flags,
            "question_word_count": len(data["question"].split()),
            "finish_reason": get_finish_reason(response),
            "usage": read_usage(response),
            "prompt": prompt,
            "raw_response": response.text,
            "error": None,
        }

    except GenerationError:
        raise

    except Exception as exc:
        raise GenerationError(
            f"Generasi varian {variant} gagal: {exc}"
        ) from exc


def error_result(variant: str, query: str, message: str) -> dict:
    return {
        "variant": variant,
        "query": query,
        "model_requested": GENERATOR_MODEL,
        **describe_settings(),
        "status": "generation_error",
        "question": "",
        "answer_key": "",
        "reasoning": None,
        "quality_flags": [],
        "usage": None,
        "error": message,
    }


# ============================================================
# 12. PIPELINE SATU TARGET KONSEP
# ============================================================

def run_pipeline(
    query: str,
    book_title: str,
    variants=VARIANTS,
    require_relevant: bool = False,
    collection=None,
    openai_client=None,
    gemini_client=None,
    verbose: bool = True,
) -> dict:
    """
    Retrieval -> evaluasi pasangan -> generasi setiap varian.

    Selalu mengembalikan dict output. Jika pasangan tidak lolos,
    'skipped_reason' terisi dan 'results' kosong.
    """

    collection = collection or open_collection()
    openai_client = openai_client or open_openai_client()

    chunk_a, chunk_b, query_tokens = retrieve_cross_chapter(
        client=openai_client,
        collection=collection,
        query=query,
        book_title=book_title,
    )

    output = {
        "experiment": {
            "created_at": datetime.now().astimezone().isoformat(),
            "query": query,
            "book_title": book_title,
            "query_tokens": query_tokens,
            **describe_settings(),
            "variants": list(variants),
        },
        "retrieval": {
            "pair_evaluation": None,
            "source_a": summarize_chunk(chunk_a) if chunk_a else None,
            "source_b": summarize_chunk(chunk_b) if chunk_b else None,
        },
        "skipped_reason": None,
        "results": [],
    }

    pair_evaluation = evaluate_pair(chunk_a, chunk_b)

    if pair_evaluation is None:
        output["skipped_reason"] = "no_partner"
        return output

    output["retrieval"]["pair_evaluation"] = summarize_pair_evaluation(
        pair_evaluation
    )

    if verbose:
        print("\n--- RETRIEVAL VALIDATION ---")
        print(json.dumps(
            output["retrieval"]["pair_evaluation"],
            indent=2,
            ensure_ascii=False,
        ))

    if not pair_evaluation.get("is_valid", False):
        output["skipped_reason"] = "pair_not_valid"
        return output

    if not pair_evaluation.get("is_relevant", True):
        if require_relevant:
            output["skipped_reason"] = "pair_not_relevant"
            return output
        if verbose:
            print("PERINGATAN: is_relevant=False, generasi tetap dijalankan.")

    gemini_client = gemini_client or genai.Client(
        api_key=os.getenv("GOOGLE_API_KEY")
    )

    for variant in variants:
        if verbose:
            print(f"\n--- GENERATING: {variant.upper()} ---")

        try:
            result = generate_question(
                client=gemini_client,
                chunk_a=chunk_a,
                chunk_b=chunk_b,
                variant=variant,
                query=query,
            )
        except GenerationError as exc:
            result = error_result(variant, query, str(exc))

        output["results"].append(result)

        if verbose:
            print_result(result)

    return output


def print_result(result: dict):
    print(f"Status   : {result['status']}")

    if result.get("question"):
        print(f"Question : {result['question']}")
        print(f"Words    : {result.get('question_word_count')}")

    if result.get("reasoning"):
        print(f"Relation : {result['reasoning']['connection_type']}")

    if result.get("quality_flags"):
        print(f"Flags    : {result['quality_flags']}")

    if result.get("usage"):
        print(f"Usage    : {result['usage']}")
        if result["usage"]["thinking_tokens"]:
            print(
                "PERINGATAN: model memakai thinking tokens. Atur "
                "THINKING_BUDGET/THINKING_LEVEL agar perbandingan adil."
            )

    if result.get("error"):
        print(f"Error    : {result['error']}")


def save_json(data: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    return path


# ============================================================
# 13. ARGUMENT COMMAND LINE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate multi-hop questions using Non-CoT and CoT."
    )
    parser.add_argument("--variant", choices=["both", *VARIANTS],
                        default="both",
                        help="Metode generasi yang dijalankan.")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY,
                        help="Target konsep untuk retrieval.")
    parser.add_argument("--book", type=str, default=DEFAULT_BOOK_TITLE,
                        help="Judul buku yang digunakan saat retrieval.")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH),
                        help="Lokasi file JSON hasil generasi.")
    parser.add_argument("--require-relevant", action="store_true",
                        help="Lewati generasi jika is_relevant=False.")
    return parser.parse_args()


# ============================================================
# 14. MAIN
# ============================================================

def main():
    args = parse_args()

    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY belum tersedia. Periksa file .env.")

    settings = describe_settings()

    print("=" * 60)
    print("MULTI-HOP QUESTION GENERATION")
    print("=" * 60)
    print(f"Model       : {settings['model']}")
    print(f"Prompt      : {settings['prompt_version']}")
    print(f"Query       : {args.query}")
    print(f"Book        : {args.book}")
    print(f"Language    : {settings['language']}")
    print(f"Temperature : {settings['temperature'] or 'SDK default'}")
    print(f"Thinking    : {settings['thinking_config'] or 'model default'}")
    print(f"Few-shot    : {settings['few_shot']}")

    variants = list(VARIANTS) if args.variant == "both" else [args.variant]

    output = run_pipeline(
        query=args.query,
        book_title=args.book,
        variants=variants,
        require_relevant=args.require_relevant,
    )

    if output["skipped_reason"]:
        print(f"\nGenerasi dilewati: {output['skipped_reason']}")

    path = save_json(output, args.output)

    print("\n" + "=" * 60)
    print(f"Hasil disimpan di: {path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()