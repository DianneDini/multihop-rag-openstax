"""
cross_chapter_retriever.py

Metode Y: Cross-Chapter Retrieval (Skenario Eksperimen 1).

Alur:
1. Mengubah satu target konsep menjadi embedding query.
2. Anchor (Chunk A): chunk dengan similarity tertinggi terhadap query
   dari buku yang dipilih (top-1).
3. Filter metadata: kandidat partner berasal dari buku yang sama dan
   berjarak minimal MIN_CHAPTER_DISTANCE bab dari anchor.
4. Partner (Chunk B): dipilih dari kandidat lolos filter menurut
   aturan PARTNER_SELECTION (lihat di bawah).
5. Evaluasi pasangan, dilaporkan sebagai DUA aspek terpisah:
   a. Validitas teknis (is_valid): jarak bab dan cosine similarity
      A-B terhadap PAIR_SIMILARITY_THRESHOLD.
   b. Relevansi query (is_relevant): apakah anchor DAN partner
      termasuk chunk yang paling relevan terhadap target konsep
      di dalam buku tersebut.

Aturan pemilihan partner (PARTNER_SELECTION):
- "rank_then_evaluate" (default): ambil kandidat dengan ranking
  tertinggi, lalu catat validitasnya. Pasangan bisa is_valid=False.
  Ini sesuai rancangan Skenario 1 di proposal.
- "threshold_filtered": di antara kandidat yang similarity A-B-nya
  >= ambang, ambil yang ranking-nya tertinggi. Jika tidak ada yang
  lolos, kembali ke kandidat ranking tertinggi dan ditandai.
  Jika dipakai, Bab 3 proposal harus menyebutkan aturan ini.

Status ambang:
- PAIR_SIMILARITY_THRESHOLD = 0.5 adalah ambang yang SEDANG DIUJI,
  belum hasil kalibrasi. Opsi --calibrate hanya memberikan data
  pembanding (sebaran similarity pasangan acak lintas bab), bukan
  ambang optimal secara otomatis. Keputusan akhir perlu dianalisis
  dan dijustifikasi di Bab 3.

Jalankan dari root project:
    python -m src.retrieval.cross_chapter.cross_chapter_retriever \
        --query "Stress" --book-title "Psychology 2e"

Data pembanding ambang:
    python -m src.retrieval.cross_chapter.cross_chapter_retriever \
        --calibrate --book-title "Psychology 2e"
"""

import argparse
import os

import numpy as np

from src.retrieval.vector_store import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_BOOK_TITLE,
    open_collection,
    open_openai_client,
    check_metadata,
    embed_query,
    cosine_similarity,
)


# ============================================================
# KONFIGURASI METODE Y
# ============================================================

def _env_float(name, default=None):
    """Membaca angka dari .env; None jika kosong."""

    value = os.getenv(name)

    if value is None or not value.strip():
        return default

    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{name} harus berupa angka.") from error


# Target konsep (istilah), bukan kalimat tanya, sesuai proposal.
DEFAULT_QUERY = "Stress"

MIN_CHAPTER_DISTANCE = 3

# Ambang similarity antarchunk yang sedang diuji (belum kalibrasi).
PAIR_SIMILARITY_THRESHOLD = _env_float("PAIR_SIMILARITY_THRESHOLD", 0.5)
THRESHOLD_STATUS = os.getenv("THRESHOLD_STATUS", "tested")

# Acuan ranking kandidat partner:
#   "query"  : relevansi terhadap query (sesuai proposal; default)
#   "anchor" : kemiripan dengan anchor (pembanding)
PARTNER_RANKING = os.getenv("PARTNER_RANKING", "query")

# Aturan pemilihan partner: "rank_then_evaluate" | "threshold_filtered"
PARTNER_SELECTION = os.getenv("PARTNER_SELECTION", "rank_then_evaluate")

# Jumlah kandidat lintas bab yang diambil dari ChromaDB.
PARTNER_CANDIDATES = int(os.getenv("PARTNER_CANDIDATES", "20"))

# Relevansi query (dinilai terpisah dari is_valid).
# Default relatif: chunk dianggap relevan jika similarity-nya
# terhadap query termasuk persentil teratas di bukunya.
QUERY_RELEVANCE_PERCENTILE = _env_float("QUERY_RELEVANCE_PERCENTILE", 95.0)
# Opsional: ambang absolut. Jika diisi, menggantikan kriteria persentil.
QUERY_RELEVANCE_THRESHOLD = _env_float("QUERY_RELEVANCE_THRESHOLD")

# Parameter data pembanding ambang (--calibrate).
CALIBRATION_PAIRS = 2000
CALIBRATION_SEED = 42

# Section non-materi yang dikecualikan dari anchor dan partner.
# Glossary/Summary hanya mengulang definisi; Critical Thinking
# Questions adalah baseline Eksperimen 3 sehingga tidak boleh
# menjadi sumber soal. Sesuaikan dengan nilai section_title di DB.
EXCLUDED_SECTIONS = [
    s.strip()
    for s in os.getenv(
        "EXCLUDED_SECTIONS",
        "Glossary,Key Terms,Summary,Review Questions,"
        "Critical Thinking Questions,Personal Application Questions,"
        "References",
    ).split(",")
    if s.strip()
]

VALID_RANKINGS = ("query", "anchor")
VALID_SELECTIONS = ("rank_then_evaluate", "threshold_filtered")


# ============================================================
# UTILITAS
# ============================================================

def get_chapter_number(metadata):
    """Mengambil nomor bab dari metadata sebagai integer."""

    try:
        return int(metadata.get("chapter_number"))
    except (TypeError, ValueError):
        return None


def normalize_rows(matrix):
    """Normalisasi L2 per baris."""

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def build_book_filter(book_title):
    """Buku yang dipilih, tanpa section non-materi."""

    if not EXCLUDED_SECTIONS:
        return {"book_title": {"$eq": book_title}}

    return {
        "$and": [
            {"book_title": {"$eq": book_title}},
            {"section_title": {"$nin": EXCLUDED_SECTIONS}},
        ]
    }


def build_partner_filter(book_title, anchor_chapter):
    """Kandidat: buku sama, bukan section non-materi,
    |bab - bab anchor| >= MIN_CHAPTER_DISTANCE."""

    conditions = [{"book_title": {"$eq": book_title}}]

    if EXCLUDED_SECTIONS:
        conditions.append({"section_title": {"$nin": EXCLUDED_SECTIONS}})

    return {
        "$and": [
            *conditions,
            {
                "$or": [
                    {
                        "chapter_number": {
                            "$lte": anchor_chapter - MIN_CHAPTER_DISTANCE
                        }
                    },
                    {
                        "chapter_number": {
                            "$gte": anchor_chapter + MIN_CHAPTER_DISTANCE
                        }
                    },
                ]
            },
        ]
    }


# ============================================================
# EMBEDDING SATU BUKU (UNTUK RELEVANSI & KALIBRASI)
# ============================================================

_BOOK_CACHE = {}


def load_book_embeddings(collection, book_title):
    """
    Mengambil seluruh embedding (ternormalisasi) dan nomor bab
    satu buku. Disimpan di cache agar tidak diambil berulang
    saat menjalankan banyak target konsep.
    """

    if book_title in _BOOK_CACHE:
        return _BOOK_CACHE[book_title]

    data = collection.get(
        where=build_book_filter(book_title),
        include=["embeddings", "metadatas"],
    )

    chapters = [get_chapter_number(m) for m in data["metadatas"]]
    keep = [i for i, c in enumerate(chapters) if c is not None]

    if len(keep) < 2:
        raise ValueError(
            f"Chunk dengan chapter_number valid pada '{book_title}' "
            "kurang dari 2."
        )

    embeddings = np.asarray(data["embeddings"], dtype=np.float32)[keep]

    book = {
        "embeddings": normalize_rows(embeddings),
        "chapters": np.asarray([chapters[i] for i in keep]),
    }

    _BOOK_CACHE[book_title] = book
    return book


def query_percentile(query_similarity, book_query_sims):
    """Persentase chunk di buku yang similarity-nya <= chunk ini."""

    return float((book_query_sims <= query_similarity).mean() * 100)


# ============================================================
# REPRESENTASI CHUNK
# ============================================================

def make_chunk(vector_id, text, metadata, embedding, query_vector,
               book_query_sims):
    """Menyusun representasi chunk yang dipakai modul lain."""

    chapter = get_chapter_number(metadata)

    if chapter is None:
        raise ValueError(
            f"Metadata chapter_number tidak valid pada {vector_id}."
        )

    embedding = np.asarray(embedding, dtype=np.float32)
    query_similarity = float(cosine_similarity(query_vector, embedding))

    return {
        "vector_id": vector_id,
        "text": text,
        "metadata": metadata,
        "embedding": embedding,
        "chapter_number": chapter,
        "query_similarity": query_similarity,
        "query_percentile": query_percentile(
            query_similarity, book_query_sims
        ),
    }


# ============================================================
# CROSS-CHAPTER RETRIEVAL
# ============================================================

def select_partner(candidate_embeddings, reference, anchor_embedding,
                   selection, threshold):
    """
    Memilih indeks partner dari kandidat.

    Return: (indeks, info) dengan info berisi aturan yang dipakai
    dan apakah terjadi fallback.
    """

    embeddings = [
        np.asarray(e, dtype=np.float32) for e in candidate_embeddings
    ]

    rank_scores = np.asarray([
        float(cosine_similarity(reference, e)) for e in embeddings
    ])
    order = np.argsort(-rank_scores)

    info = {
        "selection": selection,
        "n_candidates": len(embeddings),
        "n_candidates_above_threshold": None,
        "fallback": False,
    }

    if selection == "rank_then_evaluate":
        return int(order[0]), info

    # threshold_filtered
    pair_scores = np.asarray([
        float(cosine_similarity(anchor_embedding, e)) for e in embeddings
    ])
    passing = [int(i) for i in order if pair_scores[i] >= threshold]
    info["n_candidates_above_threshold"] = len(passing)

    if passing:
        return passing[0], info

    info["fallback"] = True
    return int(order[0]), info


def retrieve_cross_chapter(
    client,
    collection,
    query,
    book_title,
    partner_ranking=PARTNER_RANKING,
    partner_selection=PARTNER_SELECTION,
    threshold=PAIR_SIMILARITY_THRESHOLD,
):
    """
    Mengambil pasangan (anchor, partner) untuk satu target konsep.

    Return: (anchor, partner, query_tokens). partner bernilai None
    jika tidak ada kandidat yang memenuhi batasan jarak bab.
    """

    if partner_ranking not in VALID_RANKINGS:
        raise ValueError(f"partner_ranking harus salah satu {VALID_RANKINGS}.")

    if partner_selection not in VALID_SELECTIONS:
        raise ValueError(
            f"partner_selection harus salah satu {VALID_SELECTIONS}."
        )

    # 1. Embedding query dan sebaran relevansi query di buku.
    query_vector, query_tokens = embed_query(client, query)

    book = load_book_embeddings(collection, book_title)
    unit_query = query_vector / max(np.linalg.norm(query_vector), 1e-12)
    book_query_sims = book["embeddings"] @ unit_query

    # 2. Anchor: top-1 terhadap query.
    anchor_result = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=1,
        where=build_book_filter(book_title),
        include=["documents", "metadatas", "embeddings"],
    )

    if not anchor_result["ids"][0]:
        raise ValueError(
            f"Tidak ditemukan chunk untuk buku '{book_title}'. "
            "Periksa metadata book_title di ChromaDB."
        )

    anchor = make_chunk(
        anchor_result["ids"][0][0],
        anchor_result["documents"][0][0],
        anchor_result["metadatas"][0][0],
        anchor_result["embeddings"][0][0],
        query_vector,
        book_query_sims,
    )

    # 3. Kandidat lintas bab.
    reference = (
        query_vector if partner_ranking == "query" else anchor["embedding"]
    )

    partner_result = collection.query(
        query_embeddings=[reference.tolist()],
        n_results=PARTNER_CANDIDATES,
        where=build_partner_filter(book_title, anchor["chapter_number"]),
        include=["documents", "metadatas", "embeddings"],
    )

    if not partner_result["ids"][0]:
        return anchor, None, query_tokens

    # 4. Pilih partner sesuai aturan.
    best, selection_info = select_partner(
        partner_result["embeddings"][0],
        reference,
        anchor["embedding"],
        partner_selection,
        threshold,
    )

    partner = make_chunk(
        partner_result["ids"][0][best],
        partner_result["documents"][0][best],
        partner_result["metadatas"][0][best],
        partner_result["embeddings"][0][best],
        query_vector,
        book_query_sims,
    )
    partner["partner_ranking"] = partner_ranking
    partner["selection_info"] = selection_info

    return anchor, partner, query_tokens


# ============================================================
# EVALUASI PASANGAN
# ============================================================

def is_chunk_relevant(chunk):
    """Kriteria relevansi query untuk satu chunk."""

    if QUERY_RELEVANCE_THRESHOLD is not None:
        return chunk["query_similarity"] >= QUERY_RELEVANCE_THRESHOLD

    return chunk["query_percentile"] >= QUERY_RELEVANCE_PERCENTILE


def describe_relevance_rule():
    if QUERY_RELEVANCE_THRESHOLD is not None:
        return f"query_similarity >= {QUERY_RELEVANCE_THRESHOLD}"

    return f"query_percentile >= {QUERY_RELEVANCE_PERCENTILE}"


def evaluate_pair(anchor, partner, threshold=PAIR_SIMILARITY_THRESHOLD):
    """
    Mengevaluasi pasangan dalam dua aspek TERPISAH.

    1. Validitas teknis (is_valid), sesuai definisi proposal:
       jarak bab >= MIN_CHAPTER_DISTANCE dan cosine similarity
       A-B >= threshold.
    2. Relevansi query (is_relevant): anchor dan partner sama-sama
       memenuhi kriteria relevansi terhadap target konsep.

    is_valid_and_relevant disediakan sebagai ringkasan, tetapi
    keduanya sebaiknya dilaporkan terpisah di hasil eksperimen.
    """

    if anchor is None or partner is None:
        return None

    chapter_distance = abs(
        anchor["chapter_number"] - partner["chapter_number"]
    )
    pair_similarity = float(
        cosine_similarity(anchor["embedding"], partner["embedding"])
    )

    chapter_condition = chapter_distance >= MIN_CHAPTER_DISTANCE
    similarity_condition = pair_similarity >= threshold
    is_valid = chapter_condition and similarity_condition

    anchor_relevant = bool(is_chunk_relevant(anchor))
    partner_relevant = bool(is_chunk_relevant(partner))
    is_relevant = anchor_relevant and partner_relevant

    return {
        # Aspek 1: validitas teknis
        "chapter_distance": int(chapter_distance),
        "pair_similarity": pair_similarity,
        "threshold": float(threshold),
        "threshold_status": THRESHOLD_STATUS,
        "chapter_condition": bool(chapter_condition),
        "similarity_condition": bool(similarity_condition),
        "is_valid": bool(is_valid),
        # Aspek 2: relevansi query
        "relevance_rule": describe_relevance_rule(),
        "anchor_query_similarity": anchor["query_similarity"],
        "partner_query_similarity": partner["query_similarity"],
        "anchor_query_percentile": anchor["query_percentile"],
        "partner_query_percentile": partner["query_percentile"],
        "anchor_relevant": anchor_relevant,
        "partner_relevant": partner_relevant,
        "is_relevant": bool(is_relevant),
        # Ringkasan
        "is_valid_and_relevant": bool(is_valid and is_relevant),
        # Aturan pemilihan partner
        "partner_ranking": partner.get("partner_ranking"),
        "selection_info": partner.get("selection_info"),
    }


# ============================================================
# DATA PEMBANDING AMBANG
# ============================================================

def calibrate_threshold(
    collection,
    book_title,
    n_pairs=CALIBRATION_PAIRS,
    seed=CALIBRATION_SEED,
):
    """
    Sebaran cosine similarity pasangan chunk lintas bab
    (jarak >= MIN_CHAPTER_DISTANCE) yang diambil acak.

    Hasilnya adalah DATA PEMBANDING: menunjukkan seberapa mirip
    pasangan yang tidak dipilih berdasarkan konsep. Persentil tinggi
    (misalnya p95) dapat menjadi salah satu pertimbangan ambang,
    tetapi bukan ambang optimal secara otomatis.
    """

    book = load_book_embeddings(collection, book_title)
    embeddings, chapters = book["embeddings"], book["chapters"]

    rng = np.random.default_rng(seed)
    n = len(embeddings)
    sims = []
    attempts = 0

    while len(sims) < n_pairs and attempts < n_pairs * 50:
        attempts += 1
        i, j = rng.integers(0, n, size=2)

        if abs(int(chapters[i]) - int(chapters[j])) < MIN_CHAPTER_DISTANCE:
            continue

        sims.append(float(embeddings[i] @ embeddings[j]))

    sims = np.asarray(sims)

    return {
        "book_title": book_title,
        "n_chunks": int(n),
        "n_pairs": int(len(sims)),
        "seed": seed,
        "mean": float(sims.mean()),
        "std": float(sims.std()),
        "p50": float(np.percentile(sims, 50)),
        "p90": float(np.percentile(sims, 90)),
        "p95": float(np.percentile(sims, 95)),
        "p99": float(np.percentile(sims, 99)),
        "max": float(sims.max()),
        "share_ge_0.5": float((sims >= 0.5).mean()),
        "share_ge_0.7": float((sims >= 0.7).mean()),
        "current_threshold": float(PAIR_SIMILARITY_THRESHOLD),
        "current_threshold_percentile": float(
            (sims < PAIR_SIMILARITY_THRESHOLD).mean() * 100
        ),
    }


# ============================================================
# MENAMPILKAN HASIL
# ============================================================

def print_chunk(label, chunk):
    """Menampilkan metadata dan teks chunk."""

    if chunk is None:
        print(f"\n{label}: Tidak ditemukan.")
        return

    metadata = chunk["metadata"]

    print(f"\n--- {label} ---")
    print(f"Vector ID        : {chunk['vector_id']}")
    print(f"Chunk ID         : {metadata.get('chunk_id', '-')}")
    print(f"Bab              : {chunk['chapter_number']}")
    print(f"Judul bab        : {metadata.get('chapter_title', '-')}")
    print(f"Modul            : {metadata.get('module_title', '-')}")
    print(f"Bagian           : {metadata.get('section_title', '-')}")
    print(f"Similarity query : {chunk['query_similarity']:.4f}")
    print(f"Persentil query  : {chunk['query_percentile']:.1f}")
    print(f"Key metadata     : {sorted(metadata.keys())}")

    print("\nTeks chunk:")
    print(chunk["text"][:500])

    if len(chunk["text"]) > 500:
        print("... [teks dilanjutkan]")


def print_evaluation(evaluation):
    print("\n" + "=" * 65)
    print("EVALUASI PASANGAN METODE Y")
    print("=" * 65)

    if evaluation is None:
        print(
            "Tidak ditemukan kandidat partner yang memenuhi "
            "batasan jarak bab."
        )
        return

    e = evaluation
    print("[1] Validitas teknis")
    print(f"    Jarak antarbab          : {e['chapter_distance']}")
    print(f"    Cosine similarity A-B   : {e['pair_similarity']:.4f}")
    print(f"    Jarak bab >= {MIN_CHAPTER_DISTANCE}          : "
          f"{e['chapter_condition']}")
    print(f"    Similarity >= {e['threshold']:.2f} "
          f"({e['threshold_status']}) : {e['similarity_condition']}")
    print(f"    is_valid                : {e['is_valid']}")

    print("[2] Relevansi query")
    print(f"    Aturan                  : {e['relevance_rule']}")
    print(f"    Anchor  sim / persentil : "
          f"{e['anchor_query_similarity']:.4f} / "
          f"{e['anchor_query_percentile']:.1f} -> {e['anchor_relevant']}")
    print(f"    Partner sim / persentil : "
          f"{e['partner_query_similarity']:.4f} / "
          f"{e['partner_query_percentile']:.1f} -> {e['partner_relevant']}")
    print(f"    is_relevant             : {e['is_relevant']}")

    print("[3] Pemilihan partner")
    print(f"    Ranking / aturan        : {e['partner_ranking']} / "
          f"{e['selection_info']['selection']}")
    print(f"    Info                    : {e['selection_info']}")

    print(f"\nis_valid_and_relevant     : {e['is_valid_and_relevant']}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Metode Y: Cross-Chapter Retrieval."
    )
    parser.add_argument("--query", default=DEFAULT_QUERY,
                        help="Satu target konsep, misalnya 'Memory'.")
    parser.add_argument("--book-title", default=DEFAULT_BOOK_TITLE,
                        help="Nama buku sesuai metadata book_title.")
    parser.add_argument("--partner-ranking", choices=VALID_RANKINGS,
                        default=PARTNER_RANKING)
    parser.add_argument("--partner-selection", choices=VALID_SELECTIONS,
                        default=PARTNER_SELECTION)
    parser.add_argument("--calibrate", action="store_true",
                        help="Tampilkan sebaran similarity pasangan acak.")
    parser.add_argument("--list-sections", action="store_true",
                        help="Tampilkan jumlah chunk per section_title.")
    args = parser.parse_args()

    collection = open_collection()
    book_chunk_count = check_metadata(collection, args.book_title)

    if args.list_sections:
        from collections import Counter

        data = collection.get(
            where={"book_title": args.book_title},
            include=["metadatas"],
        )
        counts = Counter(
            m.get("section_title", "-") for m in data["metadatas"]
        )

        print(f"Section di '{args.book_title}' "
              f"(dikecualikan: {EXCLUDED_SECTIONS})\n")
        for title, n in counts.most_common():
            mark = "  [EXCLUDED]" if title in EXCLUDED_SECTIONS else ""
            print(f"{n:>4}  {title}{mark}")
        return

    if args.calibrate:
        stats = calibrate_threshold(collection, args.book_title)

        print("=" * 65)
        print("DATA PEMBANDING AMBANG (PASANGAN ACAK LINTAS BAB)")
        print("=" * 65)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key:<30}: {value:.4f}")
            else:
                print(f"{key:<30}: {value}")
        print("=" * 65)
        print("Catatan: ini data pembanding, bukan ambang optimal "
              "otomatis. Analisis dan justifikasi di Bab 3.")
        return

    client = open_openai_client()

    anchor, partner, query_tokens = retrieve_cross_chapter(
        client=client,
        collection=collection,
        query=args.query,
        book_title=args.book_title,
        partner_ranking=args.partner_ranking,
        partner_selection=args.partner_selection,
    )

    print("=" * 65)
    print("METODE Y: CROSS-CHAPTER RETRIEVAL")
    print("=" * 65)
    print(f"Database         : {CHROMA_DIR}")
    print(f"Collection       : {COLLECTION_NAME}")
    print(f"Buku             : {args.book_title}")
    print(f"Jumlah chunk     : {book_chunk_count}")
    print(f"Target konsep    : {args.query}")
    print(f"Kandidat partner : {PARTNER_CANDIDATES}")

    print_chunk("Chunk A (Anchor)", anchor)
    print_chunk("Chunk B (Cross-Chapter)", partner)
    print_evaluation(evaluate_pair(anchor, partner))

    print(f"\nToken query: {query_tokens}")
    print("=" * 65)


if __name__ == "__main__":
    main()