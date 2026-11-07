from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "primero" / "fifa_world_cup_limpio.csv"
OUT_DIR = Path(__file__).resolve().parent / "resultados"
PLOT_DIR = Path(__file__).resolve().parent / "graficas"
RANDOM_STATE = 42
VARIABLES = [
    "world_cup_year",
    "halftime_score_team1",
    "halftime_score_team2",
    "total_goals_match",
    "extra_time_score_team1",
    "extra_time_score_team2",
    "penalty_score_team1",
    "penalty_score_team2",
]


def guardar(fig: plt.Figure, nombre: str) -> None:
    fig.tight_layout()
    fig.savefig(PLOT_DIR / nombre, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"Gráfica guardada: {nombre}")


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el CSV limpio: {CSV_PATH}. "
            "Ejecuta primero primero/limpieza_datos.py."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    faltantes = sorted(set(VARIABLES + ["match_id", "stage", "team1", "team2"]) - set(df.columns))
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {faltantes}")

    df = df.dropna(subset=VARIABLES).reset_index(drop=True)
    if len(df) < 3:
        raise ValueError("Se necesitan al menos tres partidos para crear agrupamientos.")

    x = df[VARIABLES].to_numpy(dtype=float)
    escalador = StandardScaler()
    x_escalado = escalador.fit_transform(x)
    ks = range(2, min(8, len(df) - 1) + 1)
    indices_muestra = np.random.default_rng(RANDOM_STATE).choice(
        len(df), size=min(1000, len(df)), replace=False
    )

    filas = []
    modelos = {}
    for k in ks:
        modelo = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        grupos = modelo.fit_predict(x_escalado)
        silueta = silhouette_score(
            x_escalado[indices_muestra], grupos[indices_muestra]
        )
        db = davies_bouldin_score(x_escalado, grupos)
        modelos[k] = (modelo, grupos)
        filas.append({
            "k": k,
            "inercia": modelo.inertia_,
            "silueta": silueta,
            "davies_bouldin": db,
            "grupo_menor": int(np.bincount(grupos).min()),
        })
        print(f"k={k}: silueta={silueta:.4f}, Davies-Bouldin={db:.4f}")

    seleccion = pd.DataFrame(filas)
    seleccion.to_csv(OUT_DIR / "seleccion_k.csv", index=False)
    k_mejor = int(seleccion.loc[seleccion["silueta"].idxmax(), "k"])
    _, grupos = modelos[k_mejor]

    asignaciones = df[["match_id", "world_cup_year", "stage", "team1", "team2"]].copy()
    asignaciones["grupo"] = grupos
    asignaciones.to_csv(OUT_DIR / "partidos_agrupados.csv", index=False)

    df_perfil = df.assign(grupo=grupos)
    resumen = (
        df_perfil.groupby("grupo")
        .agg(
            partidos=("match_id", "size"),
            goles_totales_promedio=("total_goals_match", "mean"),
            anio_promedio=("world_cup_year", "mean"),
            etapa_mas_comun=("stage", lambda valores: valores.mode().iloc[0]),
        )
        .reindex(range(k_mejor))
    )
    resumen.to_csv(OUT_DIR / "resumen_clusters.csv")
    perfiles = df_perfil.groupby("grupo")[VARIABLES].mean()
    perfiles.to_csv(OUT_DIR / "perfil_numerico_clusters.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(seleccion["k"], seleccion["inercia"], "o-", color="#2980b9")
    axes[0].set(xlabel="Número de grupos (k)", ylabel="Inercia", title="Método del codo")
    axes[1].plot(seleccion["k"], seleccion["silueta"], "o-", color="#27ae60")
    axes[1].axvline(k_mejor, color="#c0392b", linestyle="--", label=f"k elegido = {k_mejor}")
    axes[1].set(xlabel="Número de grupos (k)", ylabel="Silueta", title="Calidad de agrupamiento")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    guardar(fig, "01_seleccion_k.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    colores = plt.cm.tab10(np.linspace(0, 1, k_mejor))
    for grupo in range(k_mejor):
        mascara = grupos == grupo
        ax.scatter(
            df.loc[mascara, "world_cup_year"],
            df.loc[mascara, "total_goals_match"],
            s=24,
            alpha=0.65,
            color=colores[grupo],
            label=f"Grupo {grupo}",
        )
    ax.set(
        xlabel="Año del Mundial",
        ylabel="Goles en el partido",
        title="Partidos agrupados por características",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    guardar(fig, "02_partidos_por_grupo.png")

    print(f"\nMejor k por silueta: {k_mejor}")
    print(resumen.round(2).to_string())


if __name__ == "__main__":
    main()
