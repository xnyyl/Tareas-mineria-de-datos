import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "primero" / "fifa_world_cup_limpio.csv"
OUT_DIR = Path(__file__).resolve().parent / "graficas"
OUT_DIR.mkdir(exist_ok=True)


def guardar_fig(fig, nombre):
    fig.tight_layout()
    fig.savefig(OUT_DIR / nombre, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Guardado: {nombre}")


def main():
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    df["decada"] = (df["world_cup_year"] // 10 * 10).astype(int)

    variables = ["world_cup_year", "total_goals_team1", "total_goals_team2", "total_goals_match"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, col in zip(axes.flat, variables):
        ax.hist(df[col], bins=25, color="#2E86C1", edgecolor="white")
        ax.axvline(df[col].mean(), color="#C0392B", linestyle="--", linewidth=1.5)
        ax.set_title(col)
    guardar_fig(fig, "01_histogramas.png")

    stage_counts = df["stage"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.pie(stage_counts.values, labels=stage_counts.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("Distribución por etapa")
    guardar_fig(fig, "02_pastel.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    etapas = ["Group Stage", "Round of 16", "Quarter-final", "Semi-final", "Final"]
    datos = [df.loc[df["stage"] == etapa, "total_goals_match"].dropna() for etapa in etapas if etapa in df["stage"].unique()]
    labels = [etapa for etapa in etapas if etapa in df["stage"].unique()]
    ax.boxplot(datos, patch_artist=True, tick_labels=labels)
    ax.set_title("Goles por etapa")
    ax.set_ylabel("Goles")
    guardar_fig(fig, "03_caja.png")

    fig, ax = plt.subplots(figsize=(9, 7))
    x = df["world_cup_year"]
    y = df["total_goals_match"]
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.scatter(x, y, alpha=0.5, color="#1ABC9C")
    ax.plot(xs, m * xs + b, color="#C0392B", linewidth=2)
    ax.set_xlabel("Año del Mundial")
    ax.set_ylabel("Goles del partido")
    ax.set_title("Relación entre año y goles")
    guardar_fig(fig, "04_dispersion.png")

    por_decada = df.groupby("decada")["total_goals_match"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(por_decada.index.astype(str), por_decada.values, color="#27AE60")
    ax.set_title("Promedio de goles por década")
    ax.set_xlabel("Década")
    ax.set_ylabel("Promedio de goles")
    guardar_fig(fig, "05_barras.png")

    print(f"Se generaron {len(os.listdir(OUT_DIR))} gráficas en {OUT_DIR}")


if __name__ == "__main__":
    main()
