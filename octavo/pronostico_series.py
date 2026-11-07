from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "primero" / "fifa_world_cup_limpio.csv"
OUT_DIR = Path(__file__).resolve().parent / "resultados"
PLOT_DIR = Path(__file__).resolve().parent / "graficas"
HORIZONTE = 3
EDICIONES_VALIDACION = 5


def pronostico_ingenuo(y: np.ndarray, horizonte: int) -> np.ndarray:
    return np.repeat(y[-1], horizonte)


def pronostico_media_movil(y: np.ndarray, horizonte: int) -> np.ndarray:
    return np.repeat(y[-min(3, len(y)):].mean(), horizonte)


def pronostico_deriva(y: np.ndarray, horizonte: int) -> np.ndarray:
    pendiente = (y[-1] - y[0]) / (len(y) - 1)
    return y[-1] + pendiente * np.arange(1, horizonte + 1)


def pronostico_tendencia(
    y: np.ndarray, horizonte: int, ventana: int | None = None
) -> np.ndarray:
    if ventana is not None:
        y = y[-min(ventana, len(y)):]
    t = np.arange(len(y), dtype=float)
    pendiente, intercepto = np.polyfit(t, y, 1)
    futuro = np.arange(len(y), len(y) + horizonte, dtype=float)
    return intercepto + pendiente * futuro


MODELOS = {
    "ingenuo": pronostico_ingenuo,
    "media_movil_3": pronostico_media_movil,
    "deriva": pronostico_deriva,
    "tendencia_completa": pronostico_tendencia,
    "tendencia_10_ediciones": lambda y, h: pronostico_tendencia(y, h, ventana=10),
    "tendencia_5_ediciones": lambda y, h: pronostico_tendencia(y, h, ventana=5),
}


def mae(observado: np.ndarray, predicho: np.ndarray) -> float:
    return float(np.mean(np.abs(observado - predicho)))


def rmse(observado: np.ndarray, predicho: np.ndarray) -> float:
    return float(np.sqrt(np.mean((observado - predicho) ** 2)))


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
    requeridas = {"world_cup_year", "total_goals_match"}
    faltantes = sorted(requeridas - set(df.columns))
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {faltantes}")

    serie = (
        df.groupby("world_cup_year", as_index=False)
        .agg(
            partidos=("total_goals_match", "size"),
            goles_totales=("total_goals_match", "sum"),
            goles_promedio=("total_goals_match", "mean"),
        )
        .sort_values("world_cup_year")
        .reset_index(drop=True)
    )
    if len(serie) < 8:
        raise ValueError("Se necesitan al menos ocho ediciones para validar los modelos.")
    serie["edicion"] = np.arange(len(serie))
    serie.to_csv(OUT_DIR / "serie_por_mundial.csv", index=False)

    objetivo = serie["goles_promedio"].to_numpy(dtype=float)
    anios = serie["world_cup_year"].to_numpy(dtype=int)
    inicio = max(2, len(serie) - EDICIONES_VALIDACION)
    errores = []
    for i in range(inicio, len(serie)):
        for nombre, funcion in MODELOS.items():
            predicho = float(funcion(objetivo[:i], 1)[0])
            errores.append({
                "anio": int(anios[i]),
                "modelo": nombre,
                "observado": objetivo[i],
                "predicho": predicho,
                "error": objetivo[i] - predicho,
            })

    detalle = pd.DataFrame(errores)
    detalle.to_csv(OUT_DIR / "errores_validacion.csv", index=False)
    evaluacion = (
        detalle.groupby("modelo")
        .apply(
            lambda grupo: pd.Series({
                "mae": mae(grupo["observado"].to_numpy(), grupo["predicho"].to_numpy()),
                "rmse": rmse(grupo["observado"].to_numpy(), grupo["predicho"].to_numpy()),
                "ediciones_evaluadas": len(grupo),
            }),
            include_groups=False,
        )
        .sort_values(["mae", "rmse"])
        .reset_index()
    )
    evaluacion.to_csv(OUT_DIR / "evaluacion_modelos.csv", index=False)
    elegido = str(evaluacion.iloc[0]["modelo"])
    print("Evaluación de pronósticos por edición:")
    print(evaluacion.round(4).to_string(index=False))
    print(f"\nModelo seleccionado según MAE: {elegido}")

    diferencias = np.diff(anios)
    intervalo = max(1, int(round(np.median(diferencias[-min(4, len(diferencias)):]))))
    anios_futuros = anios[-1] + intervalo * np.arange(1, HORIZONTE + 1)
    valores_futuros = MODELOS[elegido](objetivo, HORIZONTE)
    rmse_validacion = rmse(
        detalle.loc[detalle["modelo"] == elegido, "observado"].to_numpy(),
        detalle.loc[detalle["modelo"] == elegido, "predicho"].to_numpy(),
    )
    pronostico = pd.DataFrame({
        "anio_estimado": anios_futuros,
        "edicion_futura": np.arange(1, HORIZONTE + 1),
        "goles_promedio_pronosticados": valores_futuros,
        "limite_inferior_referencia": valores_futuros - 1.96 * rmse_validacion,
        "limite_superior_referencia": valores_futuros + 1.96 * rmse_validacion,
        "modelo": elegido,
        "rmse_validacion": rmse_validacion,
    })
    pronostico.to_csv(OUT_DIR / "pronostico.csv", index=False)
    print(
        f"\nSe proyectaron ediciones cada {intervalo} años a partir del intervalo "
        "mediano de las últimas ediciones."
    )
    print(pronostico.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(anios, objetivo, "o-", color="#2980b9", label="Promedio observado")
    ax.plot(
        anios_futuros,
        valores_futuros,
        "o--",
        color="#c0392b",
        label=f"Pronóstico: {elegido}",
    )
    ax.fill_between(
        anios_futuros,
        pronostico["limite_inferior_referencia"],
        pronostico["limite_superior_referencia"],
        color="#c0392b",
        alpha=0.15,
        label="±1.96 RMSE de validación",
    )
    ax.set(
        xlabel="Año del Mundial",
        ylabel="Goles promedio por partido",
        title="Serie histórica y pronóstico por edición",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    guardar(fig, "01_pronostico_goles_promedio.png")


if __name__ == "__main__":
    main()
