import os
import numpy as np
import pandas as pd

# --------------------------------------------------
# Grundeinstellungen
# --------------------------------------------------

ROOT = r"C:\Users\lukas\OneDrive\Dokumente\Studium\SET_Projekt"

# IDs der Aufzeichnungen
FEMALE_IDS = [5, 3, 2, 4, 12]
MALE_IDS   = [8, 11, 7, 13, 6, 10, 9]

# Ausgabeordner
OUT_DIR = os.path.join(ROOT, "Output_AOI")
os.makedirs(OUT_DIR, exist_ok=True)

# 24 Gesichter → Face-IDs "01" bis "24"
FACE_IDS = [f"{i:02d}" for i in range(1, 25)]

# Emotionen für jede Face-ID
EMOTION_MAP = {
    "01": "Trauer",
    "02": "Trauer",
    "03": "Ekel",
    "04": "Wut",
    "05": "Trauer",
    "06": "Ekel",
    "07": "Trauer",
    "08": "Freude",
    "09": "Neutral",
    "10": "Neutral",
    "11": "Wut",
    "12": "Neutral",
    "13": "Neutral",
    "14": "Angst",
    "15": "Ekel",
    "16": "Wut",
    "17": "Freude",
    "18": "Angst",
    "19": "Wut",
    "20": "Freude",
    "21": "Freude",
    "22": "Ekel",
    "23": "Angst",
    "24": "Angst"
}

# --------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------

def load_recording(recording_id: int) -> pd.DataFrame:
    path = os.path.join(ROOT, f"Emotionserkennung_Recording{recording_id}.tsv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"TSV fehlt: {path}")
    print(f"+ Lese Recording {recording_id}")
    return pd.read_csv(path, sep="\t", low_memory=False)


def build_aoi_mapping(df: pd.DataFrame):
    """
    Ordnet alle AOI-Spalten einer Face-ID ("01", "02", ...) zu.
    """
    mapping = {}

    for col in df.columns:
        if not col.startswith("AOI hit ["):
            continue

        inner = col[len("AOI hit ["):-1]  # "Folie3 - 01_AugeL"
        parts = inner.split(" - ")
        if len(parts) != 2:
            continue

        _, aoi_label = parts
        if "_" not in aoi_label:
            continue

        face_id, region = aoi_label.split("_", 1)

        if face_id not in FACE_IDS:
            continue

        region = region.lower()

        if face_id not in mapping:
            mapping[face_id] = {"eyes": [], "face": []}

        if "auge" in region:
            mapping[face_id]["eyes"].append(col)
        elif "gesicht" in region:
            mapping[face_id]["face"].append(col)

    return mapping


def any_hit(df: pd.DataFrame, cols):
    if not cols:
        return pd.Series(False, index=df.index)

    sub = df[cols].fillna(0)

    # numerische und textuelle Fälle
    try:
        return (sub.astype(float) != 0).any(axis=1)
    except:
        return (sub.astype(str) != "0").any(axis=1)


def compute_counts_and_switches(df_fix, face_id, eyes_cols, face_cols):
    # Hits pro Zeile
    eyes_hit = any_hit(df_fix, eyes_cols)
    face_hit = any_hit(df_fix, face_cols)

    mask = eyes_hit | face_hit
    df_face = df_fix[mask]
    if df_face.empty:
        return 0, 0, 0, 0

    df_face = df_face.sort_values("Eyetracker timestamp")

    # Zustand pro Zeile
    state = np.where(eyes_hit.loc[df_face.index], "Eyes", "Face")

    eyes_count = np.sum(state == "Eyes")
    face_count = np.sum(state == "Face")

    # Übergänge zählen
    if len(state) < 2:
        return eyes_count, face_count, 0, 0

    compressed = [state[0]]
    for s in state[1:]:
        if s != compressed[-1]:
            compressed.append(s)

    eyes_to_face = sum(a == "Eyes" and b == "Face"
                       for a, b in zip(compressed[:-1], compressed[1:]))
    face_to_eyes = sum(a == "Face" and b == "Eyes"
                       for a, b in zip(compressed[:-1], compressed[1:]))

    return eyes_count, face_count, eyes_to_face, face_to_eyes


def analyze_recording(recording_id, gender):
    df = load_recording(recording_id)

    df_fix = df[df["Eye movement type"] == "Fixation"].copy()
    if df_fix.empty:
        return []

    mapping = build_aoi_mapping(df_fix)

    results = []
    for face_id in sorted(mapping.keys()):
        eyes_cols = mapping[face_id]["eyes"]
        face_cols = mapping[face_id]["face"]

        eyes_hits, face_hits, e2f, f2e = compute_counts_and_switches(
            df_fix, face_id, eyes_cols, face_cols
        )

        results.append({
            "gender": gender,
            "recording_id": recording_id,
            "face_id": face_id,
            "emotion": EMOTION_MAP.get(face_id, "NA"),
            "eyes_hits": eyes_hits,
            "face_hits": face_hits,
            "switch_eyes_to_face": e2f,
            "switch_face_to_eyes": f2e,
        })

    return results


# --------------------------------------------------
# Hauptprogramm
# --------------------------------------------------

def main():
    all_results = []

    # weiblich
    for rid in FEMALE_IDS:
        all_results.extend(analyze_recording(rid, "female"))

    # männlich
    for rid in MALE_IDS:
        all_results.extend(analyze_recording(rid, "male"))

    df = pd.DataFrame(all_results)

    # 1) Detail-CSV – eine Zeile pro (gender, recording_id, face_id)
    out_path = os.path.join(OUT_DIR, "AOI_eyes_vs_face_aggregated_by_gender_and_face.csv")
    df.to_csv(out_path, sep=";", index=False, decimal=",")
    print("\n+ CSV (pro Person & Gesicht) gespeichert unter:", out_path)

    # 2) NEUE CSV: Mittelwerte der Switches nach Geschlecht & Emotion
    emotions = ["Trauer", "Ekel", "Wut", "Freude", "Neutral", "Angst"]
    rows = []

    for gender_long, gender_short in [("male", "m"), ("female", "w")]:
        row = {"gender": gender_short}
        df_g = df[df["gender"] == gender_long]

        for emo in emotions:
            df_ge = df_g[df_g["emotion"] == emo]

            if not df_ge.empty:
                mean_e2f = df_ge["switch_eyes_to_face"].mean()
                mean_f2e = df_ge["switch_face_to_eyes"].mean()
            else:
                mean_e2f = 0.0
                mean_f2e = 0.0

            col_e2f = f"{emo}_eyes_to_face"
            col_f2e = f"{emo}_face_to_eyes"

            row[col_e2f] = mean_e2f
            row[col_f2e] = mean_f2e

        rows.append(row)

    df_summary = pd.DataFrame(rows)

    # auf 2 Nachkommastellen runden und mit Komma als Dezimaltrennzeichen speichern
    df_summary = df_summary.round(2)

    summary_path = os.path.join(OUT_DIR, "AOI_switches_by_gender_and_emotion.csv")
    df_summary.to_csv(summary_path, sep=";", index=False, decimal=",")
    print("+ NEUE CSV (Switches nach Geschlecht & Emotion) gespeichert unter:", summary_path)

    print("\nFertig!")


if __name__ == "__main__":
    main()
