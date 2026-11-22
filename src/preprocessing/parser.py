from pathlib import Path
from lxml import etree
import json


# ============================================================
# CONFIGURATION
# ============================================================
# Menentukan lokasi file CNXML yang akan diproses.
# Untuk saat ini dataset yang digunakan adalah Psychology 2e.

DATA_DIR = Path("data/psychology")

# Namespace yang digunakan dalam struktur file CNXML.
# Namespace diperlukan agar elemen XML dapat dikenali
# saat dilakukan proses parsing.

CNX_NS = "http://cnx.rice.edu/cnxml"
MD_NS = "http://cnx.rice.edu/mdml"
COL_NS = "http://cnx.rice.edu/collxml"

NS = {
    "cnx": CNX_NS,
    "md": MD_NS,
    "col": COL_NS
}


# ============================================================
# TEXT CLEANING
# ============================================================
# Fungsi ini digunakan untuk mengambil seluruh teks dari
# suatu elemen XML.
#
# Setelah teks diambil, spasi yang berlebihan dirapikan agar
# hasil yang disimpan lebih mudah digunakan pada tahap
# preprocessing dan chunking.


import re

def clean_text(element):
    text = " ".join(element.itertext())

    # Merapikan spasi berlebih.
    text = re.sub(r"\s+", " ", text)

    # Menghilangkan spasi sebelum tanda baca.
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    # Menghapus tanda kurung yang tidak memiliki isi.
    text = re.sub(r"\s*\(\s*\)", "", text)

    return text.strip()


# ============================================================
# REFERENCE DETECTION
# ============================================================
# Bagian References tidak digunakan sebagai materi utama
# untuk knowledge base.
#
# Fungsi ini digunakan untuk mengecek apakah suatu section
# merupakan bagian References berdasarkan class atau judulnya.

def is_reference_section(section):
    """
    Mengecek apakah section merupakan bagian References.
    """

    section_class = (
        section.get("class") or ""
    ).lower()

    title_element = section.find(
        f"{{{CNX_NS}}}title"
    )

    if title_element is not None:
        title = clean_text(
            title_element
        ).lower()
    else:
        title = ""

    return (
        "references" in section_class
        or title == "references"
    )


# ============================================================
# SECTION PATH
# ============================================================
# Fungsi ini mengambil informasi hierarki section tempat
# suatu paragraph berada.
#
# Informasi ini nantinya disimpan sebagai metadata sehingga
# lokasi teks dalam buku tetap dapat diketahui.

def get_section_path(para, module_title):
    """
    Mengambil hierarki section tempat paragraph berada.

    Contoh:
    5.1 Sensation > Absolute Threshold
    """

    sections = list(
        para.iterancestors(
            f"{{{CNX_NS}}}section"
        )
    )

    # Urutan ancestor awalnya dari section paling dalam
    # ke section paling luar, sehingga dibalik agar menjadi
    # urutan hierarki dari luar ke dalam.

    sections.reverse()

    titles = []

    for section in sections:

        title_element = section.find(
            f"{{{CNX_NS}}}title"
        )

        if title_element is not None:

            title = clean_text(
                title_element
            )

            if title:
                titles.append(title)

    if titles:
        return " > ".join(titles)

    # Jika paragraph tidak berada dalam section,
    # module title digunakan sebagai informasi section.

    return module_title


# ============================================================
# PARSE ONE MODULE
# ============================================================
# Fungsi ini melakukan parsing terhadap satu file CNXML.
#
# Hasil parsing berupa kumpulan record yang berisi:
# - teks dari paragraph
# - informasi chapter
# - informasi module
# - lokasi section
# - nama file sumber
#
# Bagian seperti summary, review questions, critical thinking,
# personal application, dan references tidak dimasukkan sebagai
# materi utama knowledge base.

def parse_module(
    file_path,
    chapter_number,
    chapter_title
):
    """
    Parsing satu file CNXML menjadi
    kumpulan record teks + metadata.

    Hanya mengambil materi utama dari <content>.
    Bagian summary, review questions, critical thinking,
    dan personal application tidak dimasukkan.
    """

    file_path = Path(file_path)

    # Membuat parser XML dengan pengaturan untuk mengabaikan
    # whitespace kosong yang tidak diperlukan.

    parser = etree.XMLParser(
        remove_blank_text=True
    )

    tree = etree.parse(
        str(file_path),
        parser
    )

    root = tree.getroot()

    # --------------------------------------------------------
    # Module ID
    # --------------------------------------------------------
    # Mengambil content-id dari metadata CNXML.
    # Jika tidak ditemukan, nama file digunakan sebagai
    # alternatif module ID.

    content_id_element = root.find(
        f".//{{{MD_NS}}}content-id"
    )

    if content_id_element is not None:
        module_id = content_id_element.text
    else:
        module_id = file_path.stem

    # --------------------------------------------------------
    # Module Title
    # --------------------------------------------------------
    # Mengambil judul module dari elemen title pada dokumen.

    title_element = root.find(
        f"{{{CNX_NS}}}title"
    )

    if title_element is not None:
        module_title = clean_text(
            title_element
        )
    else:
        module_title = module_id

    records = []

    # --------------------------------------------------------
    # Find main content
    # --------------------------------------------------------
    # Mengambil bagian content yang berisi isi utama module.

    content = root.find(
        f"{{{CNX_NS}}}content"
    )

    if content is None:
        return records

    # --------------------------------------------------------
    # Section yang tidak masuk knowledge base
    # --------------------------------------------------------
    # Section berikut tidak digunakan sebagai materi utama
    # untuk knowledge base RAG.
    #
    # Review Questions dan Critical Thinking Questions
    # nantinya dapat digunakan terpisah sebagai bagian evaluasi
    # atau pembanding penelitian.

    excluded_classes = {
    "summary",
    "review-questions",
    "critical-thinking",
    "personal-application",
    "references"
    }

    # --------------------------------------------------------
    # Find all paragraphs
    # --------------------------------------------------------
    # Mengambil semua paragraph yang berada di dalam content.
    # Setelah itu setiap paragraph akan diperiksa terlebih dahulu
    # untuk menentukan apakah termasuk materi utama atau tidak.

    paragraphs = content.xpath(
        ".//cnx:para",
        namespaces=NS
    )

    for para in paragraphs:

        # ----------------------------------------------------
        # Check ancestor sections
        # ----------------------------------------------------
        # Mengecek section tempat paragraph berada.
        # Informasi ini digunakan untuk mengetahui apakah
        # paragraph termasuk bagian yang perlu dikeluarkan.

        ancestor_sections = list(
            para.iterancestors(
                f"{{{CNX_NS}}}section"
            )
        )

        # ----------------------------------------------------
        # Skip excluded sections
        # ----------------------------------------------------
        # Paragraph yang berada di dalam section tertentu
        # tidak dimasukkan ke hasil parsing utama.

        skip = False

        for section in ancestor_sections:

            section_class = (
                section.get("class") or ""
            ).lower()

            section_classes = set(
                section_class.split()
            )

            # Mengambil judul section jika tersedia.
            # Ini membantu mendeteksi References meskipun
            # section tersebut tidak memiliki class references.

            title_element = section.find(
                f"{{{CNX_NS}}}title"
            )

            section_title = ""

            if title_element is not None:
                section_title = clean_text(
                    title_element
                ).lower()

            if section_classes.intersection(
                excluded_classes
            ):
                skip = True
                break

            if section_title == "references":
                skip = True
                break

        if skip:
            continue

        # ----------------------------------------------------
        # Skip paragraphs inside Link to Learning note
        # ----------------------------------------------------
        # Link to Learning hanya berfungsi sebagai referensi
        # tambahan, misalnya link menuju video.
        #
        # Bagian ini tidak digunakan sebagai materi utama
        # knowledge base.

        ancestor_notes = para.xpath(
            "ancestor::cnx:note",
            namespaces=NS
        )

        skip_note = False

        for note in ancestor_notes:

            note_class = (
                note.get("class") or ""
            ).lower()

            if "link-to-learning" in note_class:
                skip_note = True
                break

        if skip_note:
            continue

        # ----------------------------------------------------
        # Extract text
        # ----------------------------------------------------
        # Mengambil isi teks dari paragraph yang sudah lolos
        # dari proses penyaringan.

        text = clean_text(
            para
        )

        if not text:
            continue

        # ----------------------------------------------------
        # Section path
        # ----------------------------------------------------
        # Menentukan lokasi section dari paragraph untuk
        # digunakan sebagai metadata.

        section_path = get_section_path(
            para,
            module_title
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------
        # Setiap paragraph disimpan bersama informasi lokasi
        # agar hubungan antara teks dengan sumbernya tetap
        # dapat dilacak setelah masuk ke tahap berikutnya.

        record = {
            "text": text,

            "metadata": {
                "book": "Psychology 2e",
                "chapter_number": chapter_number,
                "chapter_title": chapter_title,
                "module_id": module_id,
                "module_title": module_title,
                "section_title": section_path,
                "source_file": file_path.name
            }
        }

        
        records.append(record)

    # --------------------------------------------------------
    # Extract glossary definitions
    # --------------------------------------------------------
    # Mengambil definisi istilah dari bagian glossary.
    # Glossary berada di luar elemen content sehingga
    # perlu diproses secara terpisah.

    glossary = root.find(
        f"{{{CNX_NS}}}glossary"
    )

    if glossary is not None:

        definitions = glossary.findall(
            f"{{{CNX_NS}}}definition"
        )

        for definition in definitions:

            # Mengambil istilah dan arti dari setiap definisi.
            term_element = definition.find(
                f"{{{CNX_NS}}}term"
            )

            meaning_element = definition.find(
                f"{{{CNX_NS}}}meaning"
            )

            # Melewati definisi jika istilah atau artinya
            # tidak ditemukan.
            if term_element is None or meaning_element is None:
                continue

            term = clean_text(term_element)
            meaning = clean_text(meaning_element)

            # Menggabungkan istilah dengan definisinya.
            text = f"{term}: {meaning}"

            if not text.strip():
                continue

            # Menyimpan definisi sebagai record dengan
            # struktur metadata yang sama seperti record lain.
            record = {
                "text": text,
                "metadata": {
                    "book": "Psychology 2e",
                    "chapter_number": chapter_number,
                    "chapter_title": chapter_title,
                    "module_id": module_id,
                    "module_title": module_title,
                    "section_title": "Glossary",
                    "source_file": file_path.name
                }
            }

            records.append(record)

    return records


# ============================================================
# CHAPTER MAPPING
# ============================================================
# Mapping ini digunakan untuk menghubungkan nomor chapter
# dengan nama chapter dan daftar module yang termasuk di
# dalamnya.
#
# Informasi ini nantinya digunakan ketika setiap module
# diproses agar metadata chapter dapat diberikan dengan benar.

CHAPTER_MAPPING = {

    1: {
        "title": "Introduction to Psychology",
        "modules": [
            "m82161",
            "m82162",
            "m82163",
            "m82164",
            "m82165"
        ]
    },

    2: {
        "title": "Psychological Research",
        "modules": [
            "m82166",
            "m82167",
            "m82168",
            "m82169",
            "m82170"
        ]
    },

    3: {
        "title": "Biopsychology",
        "modules": [
            "m82171",
            "m82172",
            "m82173",
            "m82174",
            "m82175",
            "m82176"
        ]
    },

    4: {
        "title": "States of Consciousness",
        "modules": [
            "m82177",
            "m82180",
            "m82181",
            "m82182",
            "m82183",
            "m82184",
            "m82188"
        ]
    },

    5: {
        "title": "Sensation and Perception",
        "modules": [
            "m82189",
            "m82192",
            "m82193",
            "m82194",
            "m82195",
            "m82196",
            "m82197"
        ]
    },

    6: {
        "title": "Learning",
        "modules": [
            "m82198",
            "m82199",
            "m82200",
            "m82201",
            "m82202"
        ]
    },

    7: {
        "title": "Thinking and Intelligence",
        "modules": [
            "m82203",
            "m82205",
            "m82206",
            "m82207",
            "m82208",
            "m82209",
            "m82211"
        ]
    },

    8: {
        "title": "Memory",
        "modules": [
            "m82212",
            "m82213",
            "m82214",
            "m82215",
            "m82216"
        ]
    },

    9: {
        "title": "Lifespan Development",
        "modules": [
            "m82217",
            "m82218",
            "m82219",
            "m82220",
            "m82222"
        ]
    },

    10: {
        "title": "Emotion and Motivation",
        "modules": [
            "m82224",
            "m82225",
            "m82226",
            "m82228",
            "m82229"
        ]
    },

    11: {
        "title": "Personality",
        "modules": [
            "m82232",
            "m82233",
            "m82234",
            "m82235",
            "m82236",
            "m82237",
            "m82238",
            "m82239",
            "m82240",
            "m82241"
        ]
    },

    12: {
        "title": "Social Psychology",
        "modules": [
            "m82242",
            "m82273",
            "m82247",
            "m82248",
            "m82249",
            "m82250",
            "m82251",
            "m82252"
        ]
    },

    13: {
        "title": "Industrial-Organizational Psychology",
        "modules": [
            "m82253",
            "m82254",
            "m82255",
            "m82256",
            "m82257"
        ]
    },

    14: {
        "title": "Stress, Lifestyle, and Health",
        "modules": [
            "m82258",
            "m82259",
            "m82260",
            "m82261",
            "m82262",
            "m82263"
        ]
    },

    15: {
        "title": "Psychological Disorders",
        "modules": [
            "m82264",
            "m82265",
            "m82266",
            "m82267",
            "m82268",
            "m82271",
            "m82274",
            "m82275",
            "m82276",
            "m82277",
            "m82278",
            "m82279"
        ]
    },

    16: {
        "title": "Therapy and Treatment",
        "modules": [
            "m82280",
            "m82281",
            "m82282",
            "m82283",
            "m82284",
            "m82285"
        ]
    }
}


# ============================================================
# PARSE ALL MODULES
# ============================================================
# Fungsi ini menjalankan proses parsing untuk seluruh module
# dari Chapter 1 sampai Chapter 16.
#
# Setiap module diproses berdasarkan mapping chapter yang
# sudah dibuat sebelumnya.

def parse_all_modules():

    all_records = []

    total_modules = 0
    successful_modules = 0
    failed_modules = 0

    print("=" * 60)
    print("PARSING PSYCHOLOGY 2e")
    print("=" * 60)

    for chapter_number, chapter_info in CHAPTER_MAPPING.items():

        chapter_title = chapter_info["title"]
        modules = chapter_info["modules"]

        print()
        print(
            f"Chapter {chapter_number}: "
            f"{chapter_title}"
        )

        # Memproses setiap module yang termasuk dalam chapter.

        for module_id in modules:

            total_modules += 1

            file_path = (
                DATA_DIR / f"{module_id}.cnxml"
            )

            # Mengecek apakah file CNXML tersedia secara lokal.

            if not file_path.exists():

                print(
                    f"  [GAGAL] {module_id} "
                    f"- file tidak ditemukan"
                )

                failed_modules += 1

                continue

            try:

                # Melakukan parsing terhadap satu module
                # menggunakan nomor dan nama chapter yang sesuai.

                records = parse_module(
                    file_path=file_path,
                    chapter_number=chapter_number,
                    chapter_title=chapter_title
                )

                # Menambahkan hasil parsing module ke seluruh
                # record yang sudah dikumpulkan.

                all_records.extend(records)

                successful_modules += 1

                print(
                    f"  [OK] {module_id} "
                    f"- {len(records)} paragraph"
                )

            except Exception as e:

                # Jika terjadi error pada satu module,
                # proses module lain tetap dapat dilanjutkan.

                failed_modules += 1

                print(
                    f"  [GAGAL] {module_id} "
                    f"- {e}"
                )

    print()
    print("=" * 60)
    print("SELESAI")
    print("=" * 60)

    print(f"Total module      : {total_modules}")
    print(f"Berhasil          : {successful_modules}")
    print(f"Gagal             : {failed_modules}")
    print(f"Total record teks : {len(all_records)}")

    return all_records


# ============================================================
# MAIN
# ============================================================
# Bagian ini merupakan titik awal ketika file parser.py
# dijalankan.
#
# Setelah seluruh module selesai diproses, hasil parsing
# ditampilkan sebagai contoh dan kemudian disimpan dalam
# format JSON untuk digunakan pada tahap chunking.

if __name__ == "__main__":

    records = parse_all_modules()

    print()
    print("=" * 60)
    print("CONTOH HASIL")
    print("=" * 60)

    # Menampilkan tiga record pertama untuk mengecek apakah
    # hasil parsing sudah sesuai dengan isi sumber.

    for i, record in enumerate(records[:3], start=1):

        print()
        print(f"--- Record {i} ---")

        print("Text:")
        print(record["text"])

        print()
        print("Metadata:")

        for key, value in record["metadata"].items():
            print(f"{key}: {value}")

    print()
    print("=" * 60)
    print("CEK JUMLAH RECORD PER CHAPTER")
    print("=" * 60)

    # Menghitung jumlah record yang dihasilkan oleh masing-masing
    # chapter untuk membantu pengecekan hasil parsing.

    chapter_counts = {}

    for record in records:
        chapter_number = record["metadata"]["chapter_number"]

        if chapter_number not in chapter_counts:
            chapter_counts[chapter_number] = 0

        chapter_counts[chapter_number] += 1

    for chapter_number, count in chapter_counts.items():
        print(f"Chapter {chapter_number}: {count} record")

    # ========================================================
    # SAVE PARSED RECORDS
    # ========================================================
    # Menyimpan seluruh hasil parsing dalam satu file JSON.
    # File ini menjadi input untuk tahap chunking sehingga
    # proses parsing tidak perlu dilakukan kembali.

    OUTPUT_FILE = DATA_DIR / "parsed_records.json"

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=2
        )

    print()
    print(
        f"Hasil parsing disimpan ke: {OUTPUT_FILE}"
    )