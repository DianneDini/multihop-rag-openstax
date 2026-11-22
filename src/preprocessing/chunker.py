from pathlib import Path
import json
import tiktoken

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# CONFIGURATION
# ============================================================
# Menentukan folder tempat data hasil parsing dan hasil chunking
# akan disimpan.

OUTPUT_DIR = Path("data/psychology")

# File input berisi hasil parsing dari parser.py.

INPUT_FILE = OUTPUT_DIR / "parsed_records.json"

# File output yang berisi hasil chunking.

OUTPUT_FILE = OUTPUT_DIR / "chunks.json"


# ============================================================
# CHUNKING CONFIGURATION
# ============================================================
# Ukuran maksimum setiap chunk ditentukan dalam token.
# Nilai ini digunakan sebagai batas ukuran teks yang akan
# diproses oleh text splitter.

CHUNK_SIZE = 512

# Sebagian token dari chunk sebelumnya digunakan kembali
# pada chunk berikutnya agar konteks antar-chunk tetap
# terjaga.

CHUNK_OVERLAP = 128

# Encoding yang digunakan untuk menghitung jumlah token.
# Encoding ini digunakan agar ukuran chunk dapat dihitung
# berdasarkan token, bukan jumlah karakter.

TOKEN_ENCODING = "cl100k_base"


# ============================================================
# TOKENIZER
# ============================================================
# Membuat tokenizer yang digunakan untuk menghitung jumlah
# token pada setiap teks.

encoding = tiktoken.get_encoding(TOKEN_ENCODING)


def token_length(text):
    """
    Menghitung jumlah token pada teks.
    """

    return len(
        encoding.encode(text)
    )


# ============================================================
# LOAD PARSED DATA
# ============================================================
# Membaca hasil parsing yang sebelumnya sudah disimpan oleh
# parser.py dalam bentuk JSON.
#
# Data ini menjadi input untuk proses chunking.

def load_records():
    """
    Membaca hasil parsing dari parsed_records.json.
    """

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        records = json.load(file)

    return records


# ============================================================
# CREATE TEXT SPLITTER
# ============================================================
# Membuat text splitter yang digunakan untuk membagi teks
# menjadi beberapa chunk.
#
# Ukuran chunk dan overlap dihitung menggunakan token_length,
# sehingga batas 512 dan overlap 128 mengacu pada token.

def create_splitter():
    """
    Membuat RecursiveCharacterTextSplitter
    dengan ukuran berbasis token.
    """

    splitter = RecursiveCharacterTextSplitter(

        # Batas maksimum ukuran sebuah chunk.

        chunk_size=CHUNK_SIZE,

        # Jumlah token yang digunakan kembali antara
        # dua chunk yang berurutan.

        chunk_overlap=CHUNK_OVERLAP,

        # Menggunakan jumlah token sebagai dasar perhitungan
        # ukuran chunk.

        length_function=token_length,

        # Pemisah diprioritaskan dari struktur yang lebih besar
        # ke yang lebih kecil.
        #
        # Tujuannya agar pemotongan sebisa mungkin tetap
        # mengikuti paragraf, baris, dan batas kalimat.

        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
            ""
        ]
    )

    return splitter



# ============================================================
# GROUP RECORDS BY MODULE AND SECTION
# ============================================================
# Record dikelompokkan berdasarkan bab, module, dan section.
# Tujuannya agar teks dari bagian yang berbeda, termasuk
# glossary, tidak digabungkan dalam kelompok yang sama.

def group_records_by_module_and_section(records):
    """
    Mengelompokkan record berdasarkan bab, module,
    dan section tempat teks berasal.
    """

    grouped = {}

    for record in records:
        metadata = record["metadata"]

        # Identitas kelompok berdasarkan lokasi teks.
        key = (
            metadata["chapter_number"],
            metadata["module_id"],
            metadata.get("section_title", "Unknown")
        )

        if key not in grouped:
            grouped[key] = {
                "metadata": metadata.copy(),
                "texts": []
            }

        # Menambahkan teks ke kelompok section yang sesuai.
        grouped[key]["texts"].append(record["text"])

    return grouped

# ============================================================
# CREATE CHUNKS
# ============================================================
# Setelah paragraph dikelompokkan berdasarkan module,
# seluruh teks dalam satu module digabungkan dan kemudian
# dipecah menjadi beberapa chunk.

def create_chunks(records):

    # Membuat splitter dengan konfigurasi yang sudah ditentukan.

    splitter = create_splitter()

    # Mengelompokkan record berdasarkan chapter dan module.

    grouped_sections = group_records_by_module_and_section(records)

    all_chunks = []

    # Digunakan untuk memberikan ID unik pada setiap chunk.

    chunk_counter = 1

    # --------------------------------------------------------
    # Memproses module sesuai urutan data input.
    # --------------------------------------------------------
    # Dengan demikian satu module diproses sampai selesai
    # sebelum berpindah ke module berikutnya.

    
    for section_key, section_data in grouped_sections.items():

        # Metadata diambil dari section yang sedang diproses.
        metadata = section_data["metadata"]

        # Menggabungkan teks hanya dari section yang sama.
        section_text = "\n\n".join(
            section_data["texts"]
        )

        # Membagi teks section menjadi beberapa chunk.
        chunks = splitter.split_text(
            section_text
        )

        for chunk in chunks:

            # Mengabaikan chunk yang kosong.

            if not chunk.strip():
                continue

            # Membuat struktur data untuk satu chunk.
            # Metadata asli dari module tetap disimpan agar
            # sumber setiap chunk masih dapat diketahui.

            chunk_data = {

                "chunk_id":
                    f"chunk_{chunk_counter:05d}",

                "text":
                    chunk,

                "metadata":
                    metadata.copy()
            }

            all_chunks.append(
                chunk_data
            )

            chunk_counter += 1

    return all_chunks


# ============================================================
# SAVE CHUNKS
# ============================================================
# Menyimpan seluruh hasil chunking ke file JSON.
# File ini nantinya menjadi input untuk tahap berikutnya.

def save_chunks(chunks):

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# VALIDATION
# ============================================================
# Melakukan pengecekan sederhana terhadap hasil chunking.
#
# Yang diperiksa adalah jumlah chunk, jumlah token minimum,
# jumlah token maksimum, dan apakah ada chunk yang melebihi
# batas 512 token.

def validate_chunks(chunks):

    print()
    print("=" * 60)
    print("VALIDASI CHUNKS")
    print("=" * 60)

    token_counts = []

    for chunk in chunks:

        # Menghitung jumlah token pada setiap chunk.

        count = token_length(
            chunk["text"]
        )

        token_counts.append(
            count
        )

    print(
        f"Jumlah chunk       : {len(chunks)}"
    )

    print(
        f"Token minimum      : {min(token_counts)}"
    )

    print(
        f"Token maksimum     : {max(token_counts)}"
    )

    # Menghitung jumlah chunk yang melewati batas
    # 512 token.

    chunks_over_limit = sum(
        count > CHUNK_SIZE
        for count in token_counts
    )

    print(
        f"Chunk > 512 token  : "
        f"{chunks_over_limit}"
    )


# ============================================================
# MAIN
# ============================================================
# Bagian utama program menjalankan proses chunking dari awal
# sampai penyimpanan hasil.
#
# Alurnya:
# 1. Membaca hasil parsing
# 2. Mengelompokkan data berdasarkan module
# 3. Membuat chunk
# 4. Menyimpan hasil
# 5. Melakukan validasi hasil

if __name__ == "__main__":

    print("=" * 60)
    print("CHUNKING PSYCHOLOGY 2e")
    print("=" * 60)

    print()

    print("Membaca hasil parsing...")

    # Membaca data hasil parser.py.

    records = load_records()

    print(
        f"Total record input : {len(records)}"
    )

    print()

    # Menampilkan konfigurasi chunking yang digunakan.

    print(
        f"Chunk size         : "
        f"{CHUNK_SIZE} token"
    )

    print(
        f"Chunk overlap      : "
        f"{CHUNK_OVERLAP} token"
    )

    print()

    print(
        "Mengelompokkan record berdasarkan module..."
    )

    # Mengelompokkan paragraph berdasarkan module sebelum
    # proses chunking dilakukan.

    grouped_sections = group_records_by_module_and_section(
        records
    )

    print(
        f"Total kelompok    : "
        f"{len(grouped_sections)}"
    )

    print()

    print("Membuat chunks...")

    # Membuat chunk dari seluruh module.

    chunks = create_chunks(
        records
    )

    print()

    print("=" * 60)
    print("SELESAI")
    print("=" * 60)

    print(
        f"Total chunk        : "
        f"{len(chunks)}"
    )

    # Menyimpan hasil chunking ke file JSON.

    save_chunks(
        chunks
    )

    print(
        f"File output        : "
        f"{OUTPUT_FILE}"
    )

    # Melakukan pengecekan ukuran token pada setiap chunk.

    validate_chunks(
        chunks
    )

    print()
    print("=" * 60)
    print("CONTOH 4 CHUNK PERTAMA")
    print("=" * 60)

    # Menampilkan empat chunk pertama sebagai contoh untuk
    # memeriksa hasil chunking secara langsung.

    for chunk in chunks[:4]:

        print()

        print(
            f"Chunk ID: "
            f"{chunk['chunk_id']}"
        )

        # Menampilkan jumlah token dari chunk.

        print(
            f"Jumlah token: "
            f"{token_length(chunk['text'])}"
        )

        print()

        print("Text:")

        # Menampilkan isi lengkap chunk agar dapat dicek
        # apakah teks terpotong dengan benar.

        print(
            f"{chunk['text']}"
        )

        print()

        print("Metadata:")

        # Menampilkan metadata sumber dari chunk.

        for key, value in chunk["metadata"].items():

            print(
                f"  {key}: {value}"
            )