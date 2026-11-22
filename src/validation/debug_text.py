
from lxml import etree
from pathlib import Path

file_path = Path("data/psychology/m82215.cnxml")

tree = etree.parse(str(file_path))
root = tree.getroot()

ns = {"cnx": "http://cnx.rice.edu/cnxml"}

for para in root.xpath(
    ".//cnx:para[contains(., 'controversy')]",
    namespaces=ns
):
    print("=== STRUKTUR XML PARAGRAF ===")
    print(etree.tostring(
        para,
        encoding="unicode",
        pretty_print=True
    ))
    print("\n=== TEKS HASIL itertext() ===")
    print(repr("".join(para.itertext())))
    print("\n=== TEKS DENGAN SPASI ANTARELEMEN ===")
    print(repr(" ".join(para.itertext())))
