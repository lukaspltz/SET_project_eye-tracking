import pandas as pd
import re

input_file = "SET/Export/Data export - Emotionserkennung/Emotionserkennung Recording13.tsv"
output_stats = "SET/Auswertung_Aoi/Proband_41.tsv"


# 1 TSV einlesen und Zeilen auf Folien 3–72 (durch 3 teilbar) filtern
df = pd.read_csv(input_file, sep="\t")

col = "Presented Media name"

def get_slide_nr(val):
    # erwartet z.B. "Folie3.PNG"
    if isinstance(val, str) and val.startswith("Folie") and val.endswith(".PNG"):
        num_part = val[5:-4]  # zwischen "Folie" und ".PNG"
        if num_part.isdigit():
            return int(num_part)
    return None

slide_nrs = df[col].map(get_slide_nr)

row_mask = slide_nrs.notna() & (slide_nrs % 3 == 0) & (slide_nrs.between(3, 72))
df = df[row_mask]


# 2 AOI-Spalten identifizieren (nur Folien 3–72, durch 3 teilbar)
aoi_cols = [c for c in df.columns if c.startswith("AOI hit [Folie")]

pattern_info = re.compile(r"^AOI hit \[Folie(\d+) - \d+_(.+)\]")

folie_map = {}

for colname in aoi_cols:
    m = pattern_info.match(colname)
    if not m:
        continue
    folie = int(m.group(1))
    typ = m.group(2)

    # nur Folien 3–72, durch 3 teilbar
    if not ((folien_nr := folie) % 3 == 0 and 3 <= folien_nr <= 72):
        continue

    if folie not in folie_map:
        folie_map[folie] = {"Gesicht": [], "Auge": []}

    if typ == "Gesicht":
        folie_map[folie]["Gesicht"].append(colname)
    elif typ in ("AugeL", "AugeR"):
        folie_map[folie]["Auge"].append(colname)


# 3 Pro Folie: Gesicht = 100 %, Aufteilung Gesicht_ohne_Auge / Auge_im_Gesicht
results = []

for folie, groups in sorted(folie_map.items()):
    gesicht_cols = groups["Gesicht"]
    auge_cols = groups["Auge"]

    if not gesicht_cols and not auge_cols:
        continue

    folie_str = f"Folie{folie}.PNG"
    mask_rows_f = df["Presented Media name"] == folie_str
    df_f = df[mask_rows_f]

    if df_f.empty:
        continue

    has_eye = (df_f[auge_cols] == 1).any(axis=1) if auge_cols else pd.Series(False, index=df_f.index)
    has_face = (df_f[gesicht_cols] == 1).any(axis=1) if gesicht_cols else pd.Series(False, index=df_f.index)

    in_face_area = has_face | has_eye
    n_total_face_area = in_face_area.sum()
    if n_total_face_area == 0:
        continue

    n_eye = has_eye.sum()
    n_face_without_eye = ((~has_eye) & has_face).sum()

    p_eye = (n_eye / n_total_face_area) * 100
    p_face_wo_eye = (n_face_without_eye / n_total_face_area) * 100

    def fmt_percent(v):
        return f"{v:06.2f}"  # genau zwei Stellen vor und zwei nach dem Komma

    results.append({
        "Folie": folie,
        "Treffer_Gesamt_Gesichtsbereich": int(n_total_face_area),
        "Treffer_Gesicht_ohne_Auge": int(n_face_without_eye),
        "Treffer_Auge_im_Gesicht": int(n_eye),
        "Prozent_Gesicht_ohne_Auge_von_Gesicht": fmt_percent(p_face_wo_eye),
        "Prozent_Auge_von_Gesicht": fmt_percent(p_eye),
    })


# 4 Ergebnis als TSV speichern
result_df = pd.DataFrame(results).sort_values("Folie")
result_df.to_csv(output_stats, sep="\t", index=False)

print(f"Auswertung mit Gesicht = 100% pro Folie in {output_stats} gespeichert.")
