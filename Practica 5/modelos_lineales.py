from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "primero" / "fifa_world_cup_limpio.csv"
OUT_DIR = Path(__file__).resolve().parent / "resultados"
PLOT_DIR = Path(__file__).resolve().parent / "graficas"
ALFA = 0.05


def ajustar(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    matriz = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(matriz, y, rcond=None)[0]
    predicho = matriz @ beta
    residuos = y - predicho
    sse = float(residuos @ residuos)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - sse / sst
    n, p = len(y), matriz.shape[1] - 1
    gl_residuos = n - p - 1
    covarianza = np.linalg.pinv(matriz.T @ matriz) * (sse / gl_residuos)
    errores = np.sqrt(np.maximum(np.diag(covarianza), 0))
    return {
        "beta": beta,
        "predicho": predicho,
        "residuos": residuos,
        "r2": r2,
        "r2_ajustado": 1 - (1 - r2) * (n - 1) / gl_residuos,
        "errores": errores,
        "n": n,
        "p": p,
        "gl_residuos": gl_residuos,
    }


def r2_entre(x: np.ndarray, y: np.ndarray) -> float:
    predicho = np.column_stack([np.ones(len(x)), x]) @ np.linalg.lstsq(
        np.column_stack([np.ones(len(x)), x]), y, rcond=None
    )[0]
    sst = float(((y - y.mean()) ** 2).sum())
    return 1 - float(((y - predicho) ** 2).sum()) / sst


def guardar(fig: plt.Figure, nombre: str) -> None:
    fig.tight_layout()
    fig.savefig(PLOT_DIR / nombre, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"Gráfica guardada: {nombre}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    df["goles_medio_tiempo"] = (
        df["halftime_score_team1"] + df["halftime_score_team2"]
    )

    simples = [
        ("halftime_score_team1", "total_goals_match"),
        ("halftime_score_team2", "total_goals_match"),
        ("goles_medio_tiempo", "total_goals_match"),
        ("world_cup_year", "total_goals_match"),
    ]
    filas_simples = []
    for x_col, y_col in simples:
        resultado = ajustar(df[[x_col]].to_numpy(dtype=float), df[y_col].to_numpy(dtype=float))
        beta = resultado["beta"]
        errores = resultado["errores"]
        grados = resultado["gl_residuos"]
        t = beta[1] / errores[1] if errores[1] else np.nan
        filas_simples.append({
            "objetivo": y_col,
            "predictor": x_col,
            "intercepto": beta[0],
            "pendiente": beta[1],
            "error_estandar_pendiente": errores[1],
            "t": t,
            "p": 2 * stats.t.sf(abs(t), grados),
            "correlacion": df[x_col].corr(df[y_col]),
            "R2": resultado["r2"],
            "R2_ajustado": resultado["r2_ajustado"],
            "error_estandar_modelo": np.sqrt(
                float(resultado["residuos"] @ resultado["residuos"]) / grados
            ),
        })
    simple = pd.DataFrame(filas_simples).sort_values("R2", ascending=False)
    simple.to_csv(OUT_DIR / "modelos_simples.csv", index=False)
    print("Regresiones simples:")
    print(simple.to_string(index=False))

    numericas = ["world_cup_year", "halftime_score_team1", "halftime_score_team2"]
    categoricas = pd.get_dummies(df["stage"], prefix="etapa", drop_first=True, dtype=float)
    x_df = pd.concat([df[numericas].astype(float), categoricas], axis=1)
    x = x_df.to_numpy(dtype=float)
    y = df["total_goals_match"].to_numpy(dtype=float)
    multiple = ajustar(x, y)
    beta = multiple["beta"]
    errores = multiple["errores"]
    grados = multiple["gl_residuos"]
    t = beta / errores
    critico = stats.t.ppf(1 - ALFA / 2, grados)
    nombres = ["intercepto", *x_df.columns]
    coef = pd.DataFrame({
        "termino": nombres,
        "beta": beta,
        "error_estandar": errores,
        "t": t,
        "p": 2 * stats.t.sf(np.abs(t), grados),
        "ic_95_inf": beta - critico * errores,
        "ic_95_sup": beta + critico * errores,
    })
    desvios = x_df.std(ddof=1).to_numpy()
    coef["beta_estandarizado"] = [np.nan] + list(
        beta[1:] * desvios / df["total_goals_match"].std(ddof=1)
    )
    coef["significativo"] = coef["p"] < ALFA
    coef.to_csv(OUT_DIR / "coeficientes_multiple.csv", index=False)

    vif = []
    for i, nombre in enumerate(x_df.columns):
        otros = np.delete(x, i, axis=1)
        r2 = r2_entre(otros, x[:, i])
        vif.append({"predictor": nombre, "VIF": np.inf if r2 >= 1 else 1 / (1 - r2)})
    vif_df = pd.DataFrame(vif).sort_values("VIF", ascending=False)
    vif_df.to_csv(OUT_DIR / "vif.csv", index=False)

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(y))
    corte = int(len(y) * 0.8)
    entrenamiento, prueba = indices[:corte], indices[corte:]
    modelo_prueba = ajustar(x[entrenamiento], y[entrenamiento])
    x_prueba = np.column_stack([np.ones(len(prueba)), x[prueba]])
    pred_prueba = x_prueba @ modelo_prueba["beta"]
    sst_prueba = float(((y[prueba] - y[prueba].mean()) ** 2).sum())
    r2_prueba = 1 - float(((y[prueba] - pred_prueba) ** 2).sum()) / sst_prueba
    resumen = pd.DataFrame([{
        "partidos": len(y),
        "predictores": x.shape[1],
        "R2": multiple["r2"],
        "R2_ajustado": multiple["r2_ajustado"],
        "R2_prueba": r2_prueba,
        "RMSE_prueba": np.sqrt(np.mean((y[prueba] - pred_prueba) ** 2)),
        "MAE_prueba": np.mean(np.abs(y[prueba] - pred_prueba)),
        "F": (multiple["r2"] / x.shape[1])
        / ((1 - multiple["r2"]) / multiple["gl_residuos"]),
        "p_F": stats.f.sf(
            (multiple["r2"] / x.shape[1])
            / ((1 - multiple["r2"]) / multiple["gl_residuos"]),
            x.shape[1],
            multiple["gl_residuos"],
        ),
        "VIF_max": vif_df["VIF"].max(),
    }])
    resumen.to_csv(OUT_DIR / "resumen_multiple.csv", index=False)
    residuos = pd.DataFrame({
        "observado": y,
        "predicho": multiple["predicho"],
        "residuo": multiple["residuos"],
        "residuo_estandarizado": multiple["residuos"]
        / np.std(multiple["residuos"], ddof=x.shape[1] + 1),
    })
    residuos.to_csv(OUT_DIR / "residuos.csv", index=False)
    print("\nResumen de regresión múltiple:")
    print(resumen.to_string(index=False))
    print("\nCoeficientes:")
    print(coef.to_string(index=False))

    muestra = df.sample(min(1000, len(df)), random_state=42)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (x_col, y_col) in zip(axes.flat, simples):
        fila = simple.loc[simple["predictor"] == x_col].iloc[0]
        ax.scatter(muestra[x_col], muestra[y_col], s=14, alpha=0.35, color="#2980b9")
        rango = np.linspace(muestra[x_col].min(), muestra[x_col].max(), 80)
        ax.plot(rango, fila["intercepto"] + fila["pendiente"] * rango, color="#c0392b", lw=2)
        ax.set_xlabel(x_col)
        ax.set_ylabel("goles totales")
        ax.set_title(f"{x_col} vs goles totales (R2={fila['R2']:.3f})")
    guardar(fig, "01_regresiones_simples.png")

    indices_muestra = muestra.index.to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    predicho = multiple["predicho"]
    residuo = multiple["residuos"]
    axes[0, 0].scatter(predicho[indices_muestra], residuo[indices_muestra], s=14, alpha=0.4)
    axes[0, 0].axhline(0, color="#c0392b", lw=1.5)
    axes[0, 0].set(xlabel="predicción", ylabel="residuo", title="Residuos vs predicciones")
    stats.probplot(residuo[indices_muestra], dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Gráfico Q-Q de residuos")
    axes[1, 0].hist(residuo, bins=30, color="#2980b9", edgecolor="white")
    axes[1, 0].set_title("Distribución de residuos")
    axes[1, 1].scatter(y[indices_muestra], predicho[indices_muestra], s=14, alpha=0.4)
    limites = [min(y.min(), predicho.min()), max(y.max(), predicho.max())]
    axes[1, 1].plot(limites, limites, color="#c0392b", lw=1.5)
    axes[1, 1].set(xlabel="observado", ylabel="predicho", title="Observado vs predicho")
    guardar(fig, "02_diagnostico.png")

    orden = coef.iloc[1:].sort_values("beta_estandarizado")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colores = ["#27ae60" if valor >= 0 else "#c0392b" for valor in orden["beta_estandarizado"]]
    axes[0].barh(orden["termino"], orden["beta_estandarizado"], color=colores)
    axes[0].axvline(0, color="black", lw=1)
    axes[0].set_title("Coeficientes estandarizados")
    valores = list(simple["R2"]) + [multiple["r2"]]
    etiquetas = list(simple["predictor"]) + ["modelo múltiple"]
    axes[1].bar(etiquetas, valores, color=["#2980b9"] * len(simple) + ["#e67e22"])
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_ylabel("R2")
    axes[1].set_title("R2 de los modelos")
    guardar(fig, "03_coeficientes_r2.png")
    print(f"\nResultados: {OUT_DIR}")


if __name__ == "__main__":
    main()
