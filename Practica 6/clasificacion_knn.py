from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "primero" / "fifa_world_cup_limpio.csv"
OUT_DIR = Path(__file__).resolve().parent / "resultados"
PLOT_DIR = Path(__file__).resolve().parent / "graficas"
RANDOM_STATE = 42


def guardar(fig: plt.Figure, nombre: str) -> None:
    fig.tight_layout()
    fig.savefig(PLOT_DIR / nombre, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"Gráfica guardada: {nombre}")


def modelo(k: int, pesos: str) -> Pipeline:
    preprocesamiento = ColumnTransformer([
        ("numericas", StandardScaler(), ["world_cup_year", "halftime_score_team1", "halftime_score_team2"]),
        ("etapa", OneHotEncoder(handle_unknown="ignore"), ["stage"]),
    ])
    return Pipeline([
        ("preprocesamiento", preprocesamiento),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights=pesos)),
    ])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    ganador_equipo_1 = df["winner"].eq(df["team1"])
    ganador_equipo_2 = df["winner"].eq(df["team2"])
    empate = df["winner"].eq("Draw")
    valido = ganador_equipo_1 | ganador_equipo_2 | empate
    if not valido.all():
        raise ValueError("Hay partidos cuyo ganador no coincide con los equipos ni con 'Draw'.")

    df = df.copy()
    df["resultado"] = np.select(
        [empate, ganador_equipo_1, ganador_equipo_2],
        ["Empate", "Gana equipo 1", "Gana equipo 2"],
        default="",
    )
    predictores = [
        "world_cup_year",
        "halftime_score_team1",
        "halftime_score_team2",
        "stage",
    ]
    objetivo = "resultado"
    duplicados_exactos = int(df.duplicated().sum())
    x = df[predictores]
    y = df[objetivo].to_numpy()
    clases = ["Empate", "Gana equipo 1", "Gana equipo 2"]

    x_ent, x_pru, y_ent, y_pru = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    min_clase = int(pd.Series(y_ent).value_counts().min())
    pliegues = min(5, min_clase)
    if pliegues < 2:
        raise ValueError("No hay suficientes ejemplos por clase para validación cruzada estratificada.")
    cv = StratifiedKFold(n_splits=pliegues, shuffle=True, random_state=RANDOM_STATE)
    ks = [k for k in [1, 3, 5, 7, 9, 11, 15, 21, 31, 41, 51] if k < len(x_ent) * (pliegues - 1) / pliegues]
    filas = []
    for pesos in ["uniform", "distance"]:
        for k in ks:
            puntuaciones = cross_val_score(
                modelo(k, pesos),
                x_ent,
                y_ent,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1,
            )
            filas.append({
                "pesos": pesos,
                "k": k,
                "exactitud_cv": puntuaciones.mean(),
                "desv_cv": puntuaciones.std(),
            })
            print(f"{pesos}, k={k}: exactitud CV={puntuaciones.mean():.4f}")
    seleccion = pd.DataFrame(filas)
    seleccion.to_csv(OUT_DIR / "seleccion_k.csv", index=False)

    mejor = seleccion.loc[seleccion["exactitud_cv"].idxmax()]
    k_mejor = int(mejor["k"])
    pesos_mejores = str(mejor["pesos"])
    knn = modelo(k_mejor, pesos_mejores).fit(x_ent, y_ent)
    pred = knn.predict(x_pru)
    base = DummyClassifier(strategy="most_frequent").fit(x_ent, y_ent)
    pred_base = base.predict(x_pru)

    resumen = pd.DataFrame([{
        "partidos": len(y),
        "duplicados_exactos_detectados": duplicados_exactos,
        "entrenamiento": len(y_ent),
        "prueba": len(y_pru),
        "k": k_mejor,
        "pesos": pesos_mejores,
        "predictores": len(predictores),
        "clases": len(clases),
        "pliegues_cv": pliegues,
        "exactitud_cv": mejor["exactitud_cv"],
        "desv_cv": mejor["desv_cv"],
        "exactitud_entrenamiento": accuracy_score(y_ent, knn.predict(x_ent)),
        "exactitud_prueba": accuracy_score(y_pru, pred),
        "exactitud_balanceada": balanced_accuracy_score(y_pru, pred),
        "f1_macro": f1_score(y_pru, pred, average="macro", zero_division=0),
        "f1_ponderado": f1_score(y_pru, pred, average="weighted", zero_division=0),
        "exactitud_base": accuracy_score(y_pru, pred_base),
    }])
    resumen.to_csv(OUT_DIR / "resumen.csv", index=False)
    print("\nResumen del modelo:")
    print(resumen.to_string(index=False))

    reporte = pd.DataFrame(classification_report(
        y_pru, pred, labels=clases, output_dict=True, zero_division=0
    )).T
    reporte.index.name = "clase"
    reporte.to_csv(OUT_DIR / "metricas_por_clase.csv")
    mc = confusion_matrix(y_pru, pred, labels=clases)
    pd.DataFrame(mc, index=clases, columns=clases).to_csv(OUT_DIR / "matriz_confusion.csv")
    totales_fila = mc.sum(axis=1, keepdims=True)
    mc_norm = np.divide(mc, totales_fila, out=np.zeros_like(mc, dtype=float), where=totales_fila != 0)
    pd.DataFrame(mc_norm, index=clases, columns=clases).to_csv(
        OUT_DIR / "matriz_confusion_normalizada.csv"
    )
    pd.DataFrame({
        "real": y_pru,
        "predicho": pred,
        "correcto": y_pru == pred,
    }).to_csv(OUT_DIR / "predicciones.csv", index=False)

    confusiones = pd.DataFrame([
        {
            "real": clases[i],
            "predicho": clases[j],
            "casos": mc[i, j],
            "proporcion_clase_real": mc_norm[i, j],
        }
        for i in range(len(clases))
        for j in range(len(clases))
        if i != j and mc[i, j] > 0
    ], columns=["real", "predicho", "casos", "proporcion_clase_real"])
    confusiones = confusiones.sort_values("casos", ascending=False)
    confusiones.to_csv(OUT_DIR / "confusiones_frecuentes.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    for pesos, color in [("uniform", "#2980b9"), ("distance", "#27ae60")]:
        sub = seleccion.loc[seleccion["pesos"] == pesos]
        ax.plot(sub["k"], sub["exactitud_cv"], "o-", color=color, label=pesos)
        ax.fill_between(
            sub["k"],
            sub["exactitud_cv"] - sub["desv_cv"],
            sub["exactitud_cv"] + sub["desv_cv"],
            color=color,
            alpha=0.15,
        )
    ax.axhline(resumen["exactitud_base"].iloc[0], color="#7f8c8d", ls=":", label="modelo base")
    ax.axvline(k_mejor, color="#c0392b", ls="--", label=f"k elegido = {k_mejor}")
    ax.set(xlabel="k vecinos", ylabel="exactitud CV", title="Selección de k")
    ax.legend()
    ax.grid(alpha=0.3)
    guardar(fig, "01_seleccion_k.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, matriz, titulo, formato in [
        (axes[0], mc, "Conteos", "d"),
        (axes[1], mc_norm, "Proporción por clase real", ".2f"),
    ]:
        imagen = ax.imshow(matriz, cmap="Blues")
        ax.set_xticks(range(len(clases)), clases, rotation=25, ha="right")
        ax.set_yticks(range(len(clases)), clases)
        ax.set(xlabel="predicho", ylabel="real", title=titulo)
        for i in range(len(clases)):
            for j in range(len(clases)):
                ax.text(j, i, format(matriz[i, j], formato), ha="center", va="center")
        fig.colorbar(imagen, ax=ax, fraction=0.046)
    guardar(fig, "02_matriz_confusion.png")

    por_clase = reporte.loc[clases]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ancho = 0.24
    posiciones = np.arange(len(clases))
    for i, metrica in enumerate(["precision", "recall", "f1-score"]):
        axes[0].bar(
            posiciones + (i - 1) * ancho,
            por_clase[metrica],
            ancho,
            label=metrica,
        )
    axes[0].set_xticks(posiciones, clases, rotation=20, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Métricas por resultado")
    axes[0].legend()
    if not confusiones.empty:
        ejes = [
            f"{fila.real} → {fila.predicho}"
            for fila in confusiones.head(8).itertuples(index=False)
        ]
        axes[1].barh(ejes[::-1], confusiones["casos"].head(8).to_numpy()[::-1], color="#c0392b")
    axes[1].set_xlabel("partidos")
    axes[1].set_title("Confusiones más frecuentes")
    guardar(fig, "03_metricas_por_clase.png")

    variables_2d = ["halftime_score_team1", "halftime_score_team2"]
    modelo_2d = Pipeline([
        ("escala", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k_mejor, weights=pesos_mejores)),
    ])
    modelo_2d.fit(x_ent[variables_2d], y_ent)
    x1 = np.linspace(x[variables_2d[0]].min() - 0.5, x[variables_2d[0]].max() + 0.5, 100)
    x2 = np.linspace(x[variables_2d[1]].min() - 0.5, x[variables_2d[1]].max() + 0.5, 100)
    m1, m2 = np.meshgrid(x1, x2)
    zonas = modelo_2d.predict(pd.DataFrame({
        variables_2d[0]: m1.ravel(),
        variables_2d[1]: m2.ravel(),
    })).reshape(m1.shape)
    codigos = np.array([clases.index(valor) for valor in zonas.ravel()]).reshape(zonas.shape)
    colores = ["#f1c40f", "#3498db", "#e74c3c"]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.pcolormesh(
        m1,
        m2,
        codigos,
        cmap=ListedColormap(colores),
        alpha=0.22,
        shading="auto",
        vmin=-0.5,
        vmax=len(clases) - 0.5,
    )
    for clase, color in zip(clases, colores):
        seleccion_clase = y_ent == clase
        ax.scatter(
            x_ent.loc[seleccion_clase, variables_2d[0]],
            x_ent.loc[seleccion_clase, variables_2d[1]],
            s=22,
            alpha=0.8,
            color=color,
            edgecolors="white",
            linewidths=0.4,
            label=clase,
        )
    ax.set(
        xlabel="goles del equipo 1 al medio tiempo",
        ylabel="goles del equipo 2 al medio tiempo",
        title=f"Frontera de decisión KNN (k={k_mejor})",
    )
    ax.legend()
    guardar(fig, "04_frontera_decision.png")
    print(f"\nResultados: {OUT_DIR}")
    print(f"Gráficas: {PLOT_DIR}")


if __name__ == "__main__":
    main()
