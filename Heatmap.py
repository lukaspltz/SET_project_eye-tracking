import os                       # Dateien und Ordner
import numpy as np              # Arrays, Matrizen
import pandas as pd             # Tabellen/Zeitreihen (tsv)
import matplotlib.pyplot as plt # Diagramm/Datenvisualisierung
import cv2                      # Bildverarbeitung für Blur + Colormap

# --------------------------------------------------
# Pfade je nach Proband anpassen!
# --------------------------------------------------

# Projektverzeichnis
ROOT = r"C:\Users\lukas\OneDrive\Dokumente\Studium\SET_Projekt"

# TSV-Rohdaten der einzelnen Probanden(Recordings)
TSV_PATH = os.path.join(ROOT, "Emotionserkennung_Recording6.tsv")

# Verzeichnis mit den Stimulus-Bildern (Folien)
STIMULI_DIR = os.path.join(ROOT, "stimuli")

# Verzeichnis für die Ausgaben (Fixation Plot + Heatmaps)
OUT_DIR = os.path.join(ROOT, "Output")
os.makedirs(OUT_DIR, exist_ok=True)

# Nur Folien mit Gesichtern (Folie3 (1), Folie6 (1), ..., Folie72 (1))
STIMULI = [f"Folie{i} (1)" for i in range(3, 73, 3)]

# --------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------

def to_float_series(series):
    """
    Wandelt eine Spalte mit evtl. Komma-Decimaltrennern in float um.
    Beispiel: "0,534" -> 0.534
    Ungültige Einträge werden zu NaN und später rausgefiltert.
    """
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )


def load_data(tsv_path):
    """
    Liest die TSV-Datei ein und prüft, ob alle benötigten Spalten vorhanden sind.
    Gibt ein pandas.DataFrame zurück.
    """
    print("+ Lese TSV-Datei ...")
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)

    # Relevante Spalten für die Erstellung der Heatmap
    needed = [
        "Presented Stimulus name",
        "Recording resolution width",
        "Recording resolution height",
        "Fixation point X (MCSnorm)",
        "Fixation point Y (MCSnorm)",
        "Eye movement type",
        "Validity left",
        "Validity right",
        "Eye movement event duration",
    ]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Spalte '{col}' fehlt im TSV!")

    return df


def get_fixations_for_stim(df, stim_name):
    """
    Filtert alle Fixationen für einen bestimmten Stimulus (Folie).

    Gibt zurück:
    - df_stim: DataFrame nur mit Fixationen dieser Folie
    - w:       Recording-Breite in Pixeln
    - h:       Recording-Höhe in Pixeln
    """
    df_stim = df[df["Presented Stimulus name"] == stim_name].copy()

    if df_stim.empty:
        return None, None, None

    # Nur echte Fixationen (keine Sakkaden)
    df_stim = df_stim[df_stim["Eye movement type"] == "Fixation"]

    # Nur gültige Augen verwenden
    valid_mask = (
        (df_stim["Validity left"] == "Valid") |
        (df_stim["Validity right"] == "Valid")
    )
    df_stim = df_stim[valid_mask]

    if df_stim.empty:
        return None, None, None

    # Normierte Fixationskoordinaten
    df_stim["fx"] = to_float_series(df_stim["Fixation point X (MCSnorm)"])
    df_stim["fy"] = to_float_series(df_stim["Fixation point Y (MCSnorm)"])

    # Fixationsdauer
    df_stim["dur"] = pd.to_numeric(
        df_stim["Eye movement event duration"],
        errors="coerce"
    )

    df_stim = df_stim.dropna(subset=["fx", "fy", "dur"])
    if df_stim.empty:
        return None, None, None

    w = int(df_stim["Recording resolution width"].iloc[0])
    h = int(df_stim["Recording resolution height"].iloc[0])

    return df_stim, w, h


def norm_to_pixels(df_fix, width, height):
    """
    Wandelt Tobii-Normalisierte Koordinaten (MCSnorm 0..1) in Pixelkoordinaten um.
    """
    x_px = df_fix["fx"].values * width
    y_px = df_fix["fy"].values * height

    # Nur Punkte innerhalb des Bildes behalten
    mask = (x_px >= 0) & (x_px < width) & (y_px >= 0) & (y_px < height)

    x_px = x_px[mask]
    y_px = y_px[mask]
    dur = df_fix["dur"].values[mask]

    return x_px, y_px, dur


def ensure_stimulus_matches_resolution(img, width, height):
    """
    Skaliert das Stimulusbild ggf. passend zur Recording-Auflösung.
    """
    h_img, w_img = img.shape[:2]

    if (w_img, h_img) == (width, height):
        return img.astype(np.float32) / 255.0

    if img.dtype != np.uint8:
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    else:
        img_uint8 = img

    resized = cv2.resize(img_uint8, (width, height), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def create_heatmap(width, height, xs, ys, weights,
                   sigma=45.0, alpha=0.7):
    """
    Erzeugt eine farbige Heatmap aus Fixationspunkten.
    """
    heat = np.zeros((height, width), dtype=np.float32)

    for x, y, w in zip(xs, ys, weights):
        ix = int(round(x))
        iy = int(round(y))
        if 0 <= ix < width and 0 <= iy < height:
            heat[iy, ix] += float(w)

    if heat.max() <= 0:
        return np.zeros((height, width, 3), dtype=np.float32), heat

    heat_blur = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma, sigmaY=sigma)

    if heat_blur.max() > 0:
        heat_norm = heat_blur / heat_blur.max()
    else:
        heat_norm = heat_blur

    gamma = 0.8
    heat_norm = heat_norm ** gamma

    heat_uint8 = (heat_norm * 255).astype(np.uint8)
    heat_color_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_color_rgb = cv2.cvtColor(heat_color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    mask = heat_norm[..., None]

    heat_final = heat_color_rgb * (alpha * mask)

    return heat_final, heat_norm


def overlay_heatmap_on_image(img_rgb, heat_rgb):
    """
    Legt die Heatmap halbtransparent über das Originalbild.
    """
    img = img_rgb[..., :3]
    heat = heat_rgb[..., :3]

    out = np.clip(img + heat, 0.0, 1.0)
    return out


# --------------------------------------------------
# Hauptlogik
# --------------------------------------------------

df = load_data(TSV_PATH)

print("Geplante Folien:", STIMULI)
print()

for stim_name in STIMULI:
    print(f"=== {stim_name} ===")

    # Fixationen laden
    fix_df, rec_w, rec_h = get_fixations_for_stim(df, stim_name)
    if fix_df is None:
        print(f"- Keine gültigen Fixationen für {stim_name} – überspringe.\n")
        continue

    # ----------------------------------------------
    # 1) Heatmap: alle Samples verwenden
    # ----------------------------------------------
    xs, ys, durs_raw = norm_to_pixels(fix_df, rec_w, rec_h)
    durs = np.log1p(durs_raw)

    if len(xs) == 0:
        print(f"- Keine gültigen Punkte im Bildbereich – überspringe.\n")
        continue

    # ----------------------------------------------
    # 2) FIXATION-PLOT: Fixations-Events gruppieren
    # ----------------------------------------------
    if "Eye movement type index" in fix_df.columns:
        fix_events = (
            fix_df
            .groupby("Eye movement type index", sort=True)
            .agg({"fx": "mean", "fy": "mean", "dur": "sum"})
            .reset_index(drop=True)
        )

        xs_fix, ys_fix, durs_fix_raw = norm_to_pixels(fix_events, rec_w, rec_h)
        durs_fix = np.log1p(durs_fix_raw)
    else:
        xs_fix, ys_fix, durs_fix = xs, ys, durs

    # Stimulus laden
    stim_file_png = os.path.join(STIMULI_DIR, f"{stim_name}.PNG")
    stim_file_png2 = os.path.join(STIMULI_DIR, f"{stim_name}.png")

    if os.path.isfile(stim_file_png):
        stim_path = stim_file_png
    elif os.path.isfile(stim_file_png2):
        stim_path = stim_file_png2
    else:
        print(f"- Kein Stimulusbild gefunden\n")
        continue

    img = plt.imread(stim_path)
    img = ensure_stimulus_matches_resolution(img, rec_w, rec_h)

    # -----------------------------
    # FIXATION PLOT (Scanpath)
    # -----------------------------

    fig_fix, ax_fix = plt.subplots(figsize=(rec_w / 200, rec_h / 200), dpi=200)
    ax_fix.imshow(img)

    MAX_FIX = 15
    n_total = len(xs_fix)
    if n_total == 0:
        print(f"- Keine Fixations-Events – kein Fixation Plot.\n")
        continue

    n_use = min(MAX_FIX, n_total)

    xs_plot = xs_fix[:n_use]
    ys_plot = ys_fix[:n_use]
    durs_plot = durs_fix[:n_use]

    # Fixationsnummern 1..n_use
    fix_numbers = np.arange(1, n_use + 1)

    # Kreisradien anhand (log-)Fixationsdauer skalieren
    dur_min = durs_plot.min()
    dur_max = durs_plot.max()
    r_min, r_max = 18, 45

    if dur_max > dur_min:
        radii = r_min + (durs_plot - dur_min) / (dur_max - dur_min) * (r_max - r_min)
    else:
        radii = np.full_like(durs_plot, (r_min + r_max) / 2.0, dtype=float)

    # --------------------------------
    # Scanpath-Linien zwischen Fixationen
    # --------------------------------
    if n_use > 1:
        ax_fix.plot(
            xs_plot,
            ys_plot,
            linestyle="-",
            linewidth=1.5,
            color="red",
            zorder=1  # unter den Kreisen
        )

    # Label-Sichtbarkeit: spätere Kreise dürfen frühere Zahlen überdecken
    label_visible = np.ones(n_use, dtype=bool)

    for j in range(n_use):
        for i in range(j):
            dx = xs_plot[i] - xs_plot[j]
            dy = ys_plot[i] - ys_plot[j]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < radii[j]:
                label_visible[i] = False

    # Kreise + (sichtbare) Fixationsnummern zeichnen
    for x, y, r, num, show in zip(xs_plot, ys_plot, radii, fix_numbers, label_visible):
        circle = plt.Circle(
            (x, y),
            radius=r,
            edgecolor="red",
            facecolor=(1, 0, 0, 0.3),
            linewidth=1.8,
            zorder=2
        )
        ax_fix.add_patch(circle)

        if show:
            ax_fix.text(
                x, y,
                str(num),
                color="black",
                fontsize=12,
                ha="center",
                va="center",
                weight="bold",
                zorder=3
            )

    ax_fix.set_axis_off()
    fig_fix.tight_layout(pad=0)

    fixation_out = os.path.join(OUT_DIR, f"fixation_plot_{stim_name.replace(' ', '_')}.png")
    fig_fix.savefig(fixation_out, bbox_inches="tight", pad_inches=0)
    plt.close(fig_fix)
    print(f"+ Fixation Plot für {stim_name} gespeichert unter: {fixation_out}")


    # -----------------------------
    # HEATMAP
    # -----------------------------
    
    heat_rgb, _ = create_heatmap(rec_w, rec_h, xs, ys, durs)

    overlay = overlay_heatmap_on_image(img, heat_rgb)

    fig_hm, ax_h = plt.subplots(figsize=(rec_w / 200, rec_h / 200), dpi=200)
    ax_h.imshow(overlay)
    ax_h.set_axis_off()
    fig_hm.tight_layout(pad=0)

    hm_out = os.path.join(OUT_DIR, f"heatmap_{stim_name.replace(' ', '_')}.png")
    fig_hm.savefig(hm_out, bbox_inches="tight", pad_inches=0)
    plt.close(fig_hm)
    print(f"+ Heatmap für {stim_name} gespeichert unter: {hm_out}\n")

print("Fertig! Alle Fixation Plots und Heatmaps des Probanden sind im Output-Ordner.")
