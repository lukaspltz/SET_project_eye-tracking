import pandas as pd
import re

input_file = "SET/Export/Data export - Emotionserkennung/Emotionserkennung Recording13.tsv"
output_file = "SET/Auswertung_erste_Fixation/Fixation_Proband_41.tsv"

df = pd.read_csv(input_file, sep="\t")

# AOI-Spalten finden
aoi_cols = [c for c in df.columns if c.startswith("AOI hit [Folie")]

# Folien- und Typ-Info aus Spaltennamen holen
pattern = re.compile(r"^AOI hit \[Folie(\d+) - \d+_(.+)\]")

# Map: Folie -> {"Gesicht": [cols], "Auge": [cols]}
folie_map = {}

for col in aoi_cols:
    m = pattern.match(col)
    if not m:
        continue
    folie = int(m.group(1))
    typ = m.group(2)

    if folie not in folie_map:
        folie_map[folie] = {"Gesicht": [], "Auge": []}

    if typ == "Gesicht":
        folie_map[folie]["Gesicht"].append(col)
    elif typ in ("AugeL", "AugeR"):
        folie_map[folie]["Auge"].append(col)

results = []

for folie, groups in sorted(folie_map.items()):
    gesicht_cols = groups["Gesicht"]
    auge_cols = groups["Auge"]

    # Alle relevanten AOI-Spalten dieser Folie
    cols_this_slide = gesicht_cols + auge_cols
    if not cols_this_slide:
        continue

    # Zeilen, in denen überhaupt ein Treffer (Gesicht oder Auge) auf dieser Folie vorkommt
    hits_any = (df[cols_this_slide] == 1).any(axis=1)

    if not hits_any.any():
        # keine Fixation auf diesen AOIs
        results.append({"Folie": folie, "ErsteFixation_Auge1_Gesicht0": None})
        continue

    # Index der ersten Zeile mit irgendeinem Treffer
    first_idx = hits_any.idxmax()  # erster True-Index
    row = df.loc[first_idx]

    # Prüfen, ob in dieser Zeile ein Augentreffer vorliegt
    is_eye = any((row[c] == 1) for c in auge_cols)
    # Gesicht ohne Auge: Gesicht = 1, aber kein Auge = 1
    is_face_without_eye = (not is_eye) and any((row[c] == 1) for c in gesicht_cols)

    if is_eye:
        code = 1  # erste Fixation auf Auge
    elif is_face_without_eye:
        code = 0  # erste Fixation im Gesicht (Augen ausgeschlossen)
    else:
        code = None  # erste Fixation auf anderer AOI / kein klarer Treffer

    results.append({
        "Folie": folie,
        "ErsteFixation_Auge1_Gesicht0": code
    })

# Ergebnis als TSV speichern
out_df = pd.DataFrame(results).sort_values("Folie")
out_df.to_csv(output_file, sep="\t", index=False)

print(f"Ergebnis zu ersten Fixationen in {output_file} gespeichert.")
