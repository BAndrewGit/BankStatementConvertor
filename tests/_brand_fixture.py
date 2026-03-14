from __future__ import annotations

import os
from typing import List, Tuple

# Snapshot from manual Ctrl+F in DatasetsCAEN/od_firme.csv for AUCHAN/CARREFOUR mentions.
AUCHAN_AND_CARREFOUR_ROWS: List[Tuple[str, str]] = [
    ("A.F.C. GSM AUCHAN SRL", "31357160"),
    ("AUCHAN RENEWABLE ENERGY S.R.L.", "47654516"),
    ("AUCHAN COM SRL", "8857536"),
    ("AUCHAN IMPORT EXPORT ROUMANIE SRL", "21054151"),
    ("AUCHAN ROMANIA SA", "17233051"),
    ("FRANCE SELECTION SRL", "27505013"),
    ("PONNEY GROUP SRL", "21085468"),
    ("BANCA COMERCIALA ROMANA - SUCURSALA GHENCEA SA", "9558125"),
    ("COM-AL AUCHAN DUMITRESCU SRL", "1575090"),
    ("BEST PROFESSIONAL SERVICES SRL", "22857698"),
    ("BEAUCHANCE SRL", "16167618"),
    ("ARTIMA SA", "11735628"),
    ("AFC CARREFOUR MILITARI SRL", "32253729"),
    ("CHAMPION RO BUCURESTI FILIALA CHIAJNA SRL", "15199425"),
    ("COOPERATIVA AGRICOLA CARREFOUR VARASTI", "37311529"),
    ("CARREFOUR ROMANIA SA", "11588780"),
    ("CARREFOUR PROPERTY ROMANIA SRL", "24600134"),
    ("LE JADE AVENUE SRL", "22253268"),
    ("GAMA RIST SRL", "14132443"),
    ("TERRA ACHIZITII SRL", "18315613"),
    ("CARREFOUR PRODUCTIE SI DISTRIBUTIE SRL", "31765160"),
    ("NOU QUALITY SYSTEM CONTROL SRL", "16780778"),
    ("CARREFOUR LOGISTIC GROUP ROMANIA S.R.L.", "52148809"),
    ("CARREFOUR VOIAJ SRL", "21728849"),
    ("CARREFOUR VOIAJ SRL", "21728849-2"),
    ("NATURDIET SRL", "22369271"),
]


def write_brand_fixture_csv(folder: str, filename: str = "od_firme.csv") -> str:
    path = os.path.join(folder, filename)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("DENUMIRE^CUI\n")
        for name, cui in AUCHAN_AND_CARREFOUR_ROWS:
            handle.write(f"{name}^{cui}\n")
        handle.write("DM DROGERIE MARKT SRL^789\n")
        handle.write("MEGA IMAGE SRL^790\n")
    return path

