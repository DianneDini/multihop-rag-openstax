
from pathlib import Path
from lxml import etree
import json

# Memeriksa isi elemen dalam file XML dan hasil parsing.
xml_path = Path("data/psychology/m82211.cnxml")
records_path = Path("data/psychology/parsed_records.json")

tree = etree.parse(str(xml_path))
root = tree.getroot()

NS = {
    "cnx": "http://cnx.rice.edu/cnxml"
}

with records_path.open("r", encoding="utf-8") as file:
    records = json.load(file)

# Menggabungkan teks hasil parsing modul yang sedang diperiksa.
parsed_text = " ".join(
    record["text"]
    for record in records
    if record.get("module_id") == "m82211"
)

for tag in ["item", "definition"]:
    elements = root.xpath(
        f".//cnx:{tag}",
        namespaces=NS
    )

    print(f"\n=== {tag.upper()} ===")

    for element in elements:
        # Mengambil teks dari elemen XML.
        element_text = " ".join(
            "".join(element.itertext()).split()
        )

        # Mengecek apakah teks elemen muncul di hasil parsing.
        if not element_text:
            continue

        found = element_text in parsed_text

        print(f"Teks        : {element_text[:150]}")
        print(f"Sudah masuk : {found}")
        print()
