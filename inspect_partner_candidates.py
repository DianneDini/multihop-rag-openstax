
from src.retrieval.vector_store import (
    open_collection,
    open_openai_client,
    embed_query,
    cosine_similarity,
)

import numpy as np

QUERY = "naturalistic observation"
BOOK_TITLE = "Psychology 2e"
ANCHOR_CHAPTER = 2
MIN_CHAPTER_DISTANCE = 3

client = open_openai_client()
collection = open_collection()

query_vector, _ = embed_query(client, QUERY)

anchor_filter = {
    "$and": [
        {"book_title": {"$eq": BOOK_TITLE}},
        {"chapter_number": {"$eq": ANCHOR_CHAPTER}},
    ]
}

# Ambil kandidat dari seluruh bab yang memenuhi batas jarak.
partner_filter = {
    "$and": [
        {"book_title": {"$eq": BOOK_TITLE}},
        {
            "$or": [
                {
                    "chapter_number": {
                        "$lte": ANCHOR_CHAPTER - MIN_CHAPTER_DISTANCE
                    }
                },
                {
                    "chapter_number": {
                        "$gte": ANCHOR_CHAPTER + MIN_CHAPTER_DISTANCE
                    }
                },
            ]
        },
    ]
}

# Ambil banyak kandidat agar ranking dapat diperiksa.
results = collection.query(
    query_embeddings=[query_vector.tolist()],
    n_results=30,
    where=partner_filter,
    include=["documents", "metadatas", "embeddings", "distances"],
)

candidates = []

for i, chunk_id in enumerate(results["ids"][0]):
    metadata = results["metadatas"][0][i]
    chapter = int(metadata["chapter_number"])
    embedding = np.asarray(
        results["embeddings"][0][i],
        dtype=np.float32,
    )

    query_similarity = cosine_similarity(query_vector, embedding)

    candidates.append({
        "id": chunk_id,
        "chapter": chapter,
        "section": metadata.get("section_title", ""),
        "query_similarity": query_similarity,
        "text": results["documents"][0][i],
    })

candidates.sort(
    key=lambda item: item["query_similarity"],
    reverse=True,
)

for rank, item in enumerate(candidates, start=1):
    print(
        f"\nRank: {rank}"
        f"\nID: {item['id']}"
        f"\nBab: {item['chapter']}"
        f"\nSection: {item['section']}"
        f"\nQuery similarity: {item['query_similarity']:.4f}"
        f"\nTeks: {item['text'][:350]}"
    )
    print("-" * 60)
