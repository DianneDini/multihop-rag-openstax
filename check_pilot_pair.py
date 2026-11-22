
from src.retrieval.vector_store import open_collection

collection = open_collection()

chunk_id = "psychology_2e_chunk_00249"

result = collection.get(
    ids=[chunk_id],
    include=["metadatas", "documents"]
)

if not result["ids"]:
    print("Chunk tidak ditemukan di ChromaDB.")
else:
    metadata = result["metadatas"][0]
    document = result["documents"][0]

    print("ID:", result["ids"][0])
    print("Bab:", metadata.get("chapter_number"))
    print("Judul bab:", metadata.get("chapter_title"))
    print("Modul:", metadata.get("module_title"))
    print("Section:", metadata.get("section_title"))
    print("Panjang teks:", len(document), "karakter")

    print("\n=== TEKS LENGKAP CHUNK BAB 6 ===\n")
    print(document)
