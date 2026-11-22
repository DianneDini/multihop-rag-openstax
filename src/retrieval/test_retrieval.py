"""
test_retrieval.py

Metode X: Conventional Retrieval (Skenario Eksperimen 1).

- Satu target konsep sebagai query, dibatasi pada satu buku.
- Mengambil dua chunk teratas berdasarkan relevansi query,
  TANPA batasan jarak bab.
- Evaluasi pasangan memakai fungsi dan kriteria yang SAMA dengan
  Metode Y (jarak bab, ambang similarity, relevansi query).

Agar perbandingan adil, seluruh pengaturan bersama diimpor dari
cross_chapter_retriever.py (satu sumber konfigurasi):
    - EXCLUDED_SECTIONS (Glossary, dll. tidak ikut diambil)
    - MIN_CHAPTER_DISTANCE, PAIR_SIMILARITY_THRESHOLD
    - kriteria relevansi query
    - representasi chunk dan evaluate_pair()
Satu-satunya perbedaan dengan Metode Y adalah tidak adanya
batasan jarak bab saat retrieval.

Jalankan dari root project:
    python -m src.retrieval.test_retrieval --query "Stress"
    python -m src.retrieval.test_retrieval \
        --query "Natural Selection" --book-title "Concepts of Biology"
"""

import argparse

import numpy as np

from src.retrieval.vector_store import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_BOOK_TITLE,
    open_collection,
    open_openai_client,
    check_metadata,
    embed_query,
)
from src.retrieval.cross_chapter.cross_chapter_retriever import (
    DEFAULT_QUERY,
    EXCLUDED_SECTIONS,
    MIN_CHAPTER_DISTANCE,
    PAIR_SIMILARITY_THRESHOLD,
    build_book_filter,
    load_book_embeddings,
    make_chunk,
    evaluate_pair,
    print_chunk,
)


# ============================================================
# KONFIGURASI METODE X
# ============================================================

TOP_K = 2


# ============================================================
# CONVENTIONAL RETRIEVAL
# ============================================================

def retrieve_conventional(client, collection, query, book_title):
    """
    Mengambil dua chunk teratas berdasarkan relevansi query dari
    satu buku, tanpa batasan jarak bab. Section non-materi
    (EXCLUDED_SECTIONS) dikecualikan, sama seperti Metode Y.

    Return: (chunks, query_tokens)
    """

    query_vector, query_tokens = embed_query(client, query)

    # Sebaran relevansi query di buku (untuk persentil relevansi,
    # dihitung dengan cara yang sama seperti Metode Y).
    book = load_book_embeddings(collection, book_title)
    unit_query = query_vector / max(np.linalg.norm(query_vector), 1e-12)
    book_query_sims = book["embeddings"] @ unit_query

    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=TOP_K,
        where=build_book_filter(book_title),
        include=["documents", "metadatas", "embeddings"],
    )

    if not results["ids"][0]:
        raise ValueError(
            f"Tidak ditemukan chunk untuk buku '{book_title}'. "
            "Periksa nilai book_title dalam metadata ChromaDB."
        )

    chunks = [
        make_chunk(
            vector_id,
            document,
            metadata,
            embedding,
            query_vector,
            book_query_sims,
        )
        for vector_id, document, metadata, embedding in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["embeddings"][0],
        )
    ]

    return chunks, query_tokens


# ============================================================
# OUTPUT
# ============================================================

def print_evaluation(evaluation):
    print("\n" + "=" * 65)
    print("EVALUASI PASANGAN METODE X")
    print("=" * 65)

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
    print(f"    Chunk 1 sim / persentil : "
          f"{e['anchor_query_similarity']:.4f} / "
          f"{e['anchor_query_percentile']:.1f} -> {e['anchor_relevant']}")
    print(f"    Chunk 2 sim / persentil : "
          f"{e['partner_query_similarity']:.4f} / "
          f"{e['partner_query_percentile']:.1f} -> {e['partner_relevant']}")
    print(f"    is_relevant             : {e['is_relevant']}")

    print(f"\nis_valid_and_relevant     : {e['is_valid_and_relevant']}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Metode X: Conventional Retrieval."
    )
    parser.add_argument("--query", "--question", dest="query",
                        default=DEFAULT_QUERY,
                        help="Satu target konsep.")
    parser.add_argument("--book-title", default=DEFAULT_BOOK_TITLE,
                        help="Nama buku sesuai metadata book_title.")
    args = parser.parse_args()

    collection = open_collection()
    chunk_count = check_metadata(collection, args.book_title)
    client = open_openai_client()

    chunks, query_tokens = retrieve_conventional(
        client, collection, args.query, args.book_title
    )

    print("=" * 65)
    print("METODE X: CONVENTIONAL RETRIEVAL")
    print("=" * 65)
    print(f"Database          : {CHROMA_DIR}")
    print(f"Collection        : {COLLECTION_NAME}")
    print(f"Buku              : {args.book_title}")
    print(f"Jumlah chunk buku : {chunk_count}")
    print(f"Section excluded  : {EXCLUDED_SECTIONS}")
    print(f"Target konsep     : {args.query}")
    print(f"Ambang similarity : {PAIR_SIMILARITY_THRESHOLD}")
    print(f"Chunk terpilih    : {len(chunks)}")

    for index, chunk in enumerate(chunks, start=1):
        print_chunk(f"Chunk {index}", chunk)

    if len(chunks) == 2:
        print_evaluation(evaluate_pair(chunks[0], chunks[1]))
    else:
        print("\nPasangan belum bisa dievaluasi karena "
              "jumlah chunk kurang dari dua.")

    print(f"\nToken query: {query_tokens}")
    print("=" * 65)


if __name__ == "__main__":
    main()