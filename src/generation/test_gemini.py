"""
test_gemini.py
Uji pertama Fase 3: memastikan GOOGLE_API_KEY, nama model, dan akses API
berfungsi, serta mencatat versi model yang benar-benar menjawab.

Memakai prompt netral (BUKAN teks OpenStax), jadi aman dijalankan sebelum
billing aktif.

Jalankan dari root project:
    python src/test_gemini.py
"""

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
# Nama model diatur lewat .env supaya mudah diganti tanpa mengubah kode.

GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gemini-3.5-flash-lite")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.8-flash")

TEST_PROMPT = "Jawab dengan satu kalimat pendek: apa ibu kota Indonesia?"


# ============================================================
# PANGGIL MODEL
# ============================================================

def run_model(client, model, prompt):
    """Mengirim satu prompt dan mengembalikan jawaban beserta metadata."""
    response = client.models.generate_content(model=model, contents=prompt)
    usage = response.usage_metadata

    return {
        "model_requested": model,
        # Versi model yang benar-benar menjawab. Simpan ini di output
        # eksperimen, karena nama model bisa diarahkan ke versi lain.
        "model_version": response.model_version,
        "text": (response.text or "").strip(),
        "input_tokens": usage.prompt_token_count or 0,
        "output_tokens": usage.candidates_token_count or 0,
        "thinking_tokens": usage.thoughts_token_count or 0,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY tidak ditemukan. Periksa file .env."
        )

    client = genai.Client(api_key=api_key)

    models = [GENERATOR_MODEL]
    if JUDGE_MODEL != GENERATOR_MODEL:
        models.append(JUDGE_MODEL)

    for model in models:
        print()
        print("=" * 60)
        print(f"MODEL: {model}")
        print("=" * 60)

        try:
            result = run_model(client, model, TEST_PROMPT)
        except Exception as error:
            print(f"GAGAL: {error}")
            continue

        print(f"Versi model yang menjawab : {result['model_version']}")
        print(f"Jawaban                   : {result['text']}")
        print(f"Token input               : {result['input_tokens']}")
        print(f"Token output              : {result['output_tokens']}")
        print(f"Token thinking            : {result['thinking_tokens']}")


if __name__ == "__main__":
    main()