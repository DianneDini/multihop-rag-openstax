import requests
import xml.etree.ElementTree as ET
from pathlib import Path
import time


# ============================================================
# KONFIGURASI
# ============================================================
# URL utama untuk mengambil file dari repository
# OpenStax Psychology 2e di GitHub.
#
# Repository digunakan sebagai sumber data karena file buku
# disimpan dalam format CNXML dan sudah memiliki struktur
# chapter serta module.

REPO_RAW_URL = (
    "https://raw.githubusercontent.com/"
    "openstax/osbooks-psychology/main"
)

# File collection digunakan untuk mengetahui struktur buku,
# termasuk daftar chapter dan module yang terdapat di
# Psychology 2e.

COLLECTION_URL = (
    f"{REPO_RAW_URL}/collections/psychology-2e.collection.xml"
)

# Folder penyimpanan file CNXML yang berhasil didownload.

OUTPUT_DIR = Path("data/psychology")

# Membuat folder output apabila folder belum tersedia.

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STEP 1 - MENGAMBIL COLLECTION.XML
# ============================================================
# collection.xml digunakan sebagai acuan struktur buku.
# Dari file ini akan diambil nama chapter dan ID module
# yang terdapat pada masing-masing chapter.

print("=" * 60)
print("STEP 1 - Mengambil collection.xml")
print("=" * 60)

try:
    response = requests.get(COLLECTION_URL, timeout=30)
    response.raise_for_status()
except requests.RequestException as e:
    print("GAGAL mengambil collection.xml")
    print("Error:", e)
    raise SystemExit

print("BERHASIL mengambil collection.xml")


# ============================================================
# STEP 2 - MEMBACA STRUKTUR CHAPTER DAN MODULE
# ============================================================
# collection.xml berbentuk XML sehingga perlu diparsing
# terlebih dahulu agar struktur chapter dan module dapat
# dibaca oleh program.
#
# Setiap chapter memiliki beberapa module. ID module tersebut
# nantinya digunakan untuk mengambil file index.cnxml dari
# repository OpenStax.

print()
print("=" * 60)
print("STEP 2 - Membaca struktur Chapter dan Module")
print("=" * 60)

try:
    root = ET.fromstring(response.text)
except ET.ParseError as e:
    print("GAGAL membaca XML")
    print("Error:", e)
    raise SystemExit


# Namespace diperlukan karena elemen-elemen pada XML
# menggunakan namespace dari CNX/CollXML.

MD_NS = "http://cnx.rice.edu/mdml"
COL_NS = "http://cnx.rice.edu/collxml"

# Menyimpan hasil pemetaan chapter dan module.
#
# Contoh:
# {
#     "chapter": "Introduction to Psychology",
#     "module_id": "m82161"
# }

modules = []


# Mengambil setiap subcollection.
# Dalam struktur collection.xml, subcollection digunakan
# untuk merepresentasikan chapter pada buku.

for subcollection in root.findall(
    f".//{{{COL_NS}}}subcollection"
):

    # Mengambil judul chapter.

    title_element = subcollection.find(
        f"{{{MD_NS}}}title"
    )

    if title_element is None:
        continue

    chapter_name = title_element.text.strip()

    # Setiap chapter memiliki elemen content yang berisi
    # daftar module.

    content = subcollection.find(
        f"{{{COL_NS}}}content"
    )

    if content is None:
        continue

    # Mengambil setiap module yang terdapat di dalam chapter.

    for module in content.findall(
        f"{{{COL_NS}}}module"
    ):

        # ID module digunakan untuk menentukan lokasi
        # file CNXML pada repository OpenStax.

        module_id = module.get("document")

        if module_id:
            modules.append(
                {
                    "chapter": chapter_name,
                    "module_id": module_id
                }
            )


# Menampilkan jumlah module yang berhasil ditemukan
# dari collection.xml.

print(f"Total module ditemukan: {len(modules)}")

print()
print("Contoh module:")

for item in modules[:5]:
    print(
        f"- {item['chapter']} → "
        f"{item['module_id']}"
    )


# ============================================================
# STEP 3 - DOWNLOAD FILE CNXML
# ============================================================
# Setelah daftar module diperoleh, program mengambil file
# index.cnxml dari masing-masing module.
#
# File CNXML merupakan sumber utama yang nantinya digunakan
# pada tahap parsing untuk mengambil teks dan metadata buku.

print()
print("=" * 60)
print("STEP 3 - Download CNXML")
print("=" * 60)

success = 0
skipped = 0
failed = 0


# Melakukan download untuk setiap module yang ditemukan.

for index, item in enumerate(modules, start=1):

    chapter_name = item["chapter"]
    module_id = item["module_id"]

    # Setiap module disimpan menggunakan module ID sebagai
    # nama file agar mudah dikenali dan diproses kembali.

    output_file = OUTPUT_DIR / f"{module_id}.cnxml"

    print(
        f"[{index}/{len(modules)}] "
        f"{chapter_name} - {module_id}"
    )

    # Jika file sudah pernah didownload sebelumnya,
    # file tidak perlu didownload ulang.

    if output_file.exists():

        print("  → Sudah ada, dilewati.")

        skipped += 1
        continue


    # Menentukan URL file CNXML berdasarkan module ID.

    url = (
        f"{REPO_RAW_URL}/modules/"
        f"{module_id}/index.cnxml"
    )


    try:

        # Mengambil file CNXML dari repository.

        response = requests.get(
            url,
            timeout=30
        )

        if response.status_code == 200:

            # Menyimpan isi file ke folder data/psychology.

            output_file.write_text(
                response.text,
                encoding="utf-8"
            )

            # Menghitung ukuran file untuk ditampilkan
            # sebagai informasi proses download.

            size_kb = (
                output_file.stat().st_size / 1024
            )

            print(
                f"  → BERHASIL "
                f"({size_kb:.2f} KB)"
            )

            success += 1

        else:

            # Jika server mengembalikan status selain 200,
            # module dianggap gagal didownload.

            print(
                f"  → GAGAL "
                f"(HTTP {response.status_code})"
            )

            failed += 1

    except requests.RequestException as e:

        # Menangani error koneksi atau request ke repository.

        print("  → GAGAL:", e)

        failed += 1


    # Memberikan jeda singkat antar request agar proses
    # download tidak dilakukan terlalu cepat.

    time.sleep(0.2)


# ============================================================
# RINGKASAN HASIL DOWNLOAD
# ============================================================
# Menampilkan hasil akhir proses pengambilan data:
# jumlah module, file yang berhasil, file yang dilewati,
# dan file yang gagal.

print()
print("=" * 60)
print("SELESAI")
print("=" * 60)

print(f"Total module : {len(modules)}")
print(f"Berhasil     : {success}")
print(f"Dilewati     : {skipped}")
print(f"Gagal        : {failed}")

print()
print("Lokasi data:")
print(OUTPUT_DIR.resolve())