from collections import Counter
from pathlib import Path
import re
import unicodedata

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "primero" / "fifa_world_cup_limpio.csv"
OUT_DIR = Path(__file__).resolve().parent / "resultados"
PLOT_DIR = Path(__file__).resolve().parent / "graficas"
STOPWORDS = set(
    """
    a al algo algun alguna algunas algunos ante antes aqui asi aun aunque bajo cada
    como con contra cual cuales cuando de del desde donde durante e el ella ellas
    ellos en entre era eran es esa esas ese eso esos esta estaba estaban estas este
    esto estos fue fueron ha hacia han hasta hay la las le les lo los mas me mi mis
    mucho muy nada ni no nos o os otra otras otro otros para pero por porque que
    quien se sean ser si sin sobre son su sus te tiene tienen todo todos tu tus un
    una unas uno unos y ya the of and to in on at for from by
    stadium estadio arena ground field park city cup world
    """.split()
)
COLUMNAS_TEXTO = ["team1", "team2", "host_country", "city", "stadium"]
COLUMNAS_DOCUMENTO = COLUMNAS_TEXTO + ["stage", "round_raw"]


def normalizar(texto: object) -> str:
    sin_acentos = unicodedata.normalize("NFKD", str(texto).lower())
    return "".join(c for c in sin_acentos if not unicodedata.combining(c))


def tokenizar(texto: object) -> list[str]:
    return [
        palabra
        for palabra in re.findall(r"[a-z0-9]+", normalizar(texto))
        if len(palabra) > 1 and palabra not in STOPWORDS
    ]


def contar_documentos(documentos: list[list[str]]) -> Counter[str]:
    frecuencias: Counter[str] = Counter()
    for documento in documentos:
        frecuencias.update(documento)
    return frecuencias


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
    faltantes = sorted(set(COLUMNAS_DOCUMENTO + ["match_id"]) - set(df.columns))
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {faltantes}")

    texto = df[COLUMNAS_TEXTO].fillna("").astype(str).agg(" ".join, axis=1)
    documentos = [tokenizar(valor) for valor in texto]
    frecuencias = contar_documentos(documentos)
    if not frecuencias:
        raise ValueError("No se encontraron palabras para analizar en el CSV.")

    tabla_frecuencias = pd.DataFrame(
        frecuencias.most_common(), columns=["palabra", "frecuencia"]
    )
    tabla_frecuencias["proporcion"] = (
        tabla_frecuencias["frecuencia"] / tabla_frecuencias["frecuencia"].sum()
    )
    tabla_frecuencias.to_csv(OUT_DIR / "frecuencias_palabras.csv", index=False)

    bigramas: Counter[tuple[str, str]] = Counter()
    campos = df[COLUMNAS_TEXTO].fillna("").astype(str)
    for fila in campos.itertuples(index=False, name=None):
        for campo in fila:
            tokens_campo = tokenizar(campo)
            bigramas.update(zip(tokens_campo, tokens_campo[1:]))
    tabla_bigramas = pd.DataFrame(
        [
            {"bigrama": f"{primera} {segunda}", "frecuencia": frecuencia}
            for (primera, segunda), frecuencia in bigramas.most_common(100)
        ]
    )
    tabla_bigramas.to_csv(OUT_DIR / "bigramas.csv", index=False)

    df["texto_documento"] = df[COLUMNAS_DOCUMENTO].fillna("").astype(str).agg(" ".join, axis=1)
    df["tokens"] = df["texto_documento"].map(tokenizar)
    etapas = sorted(df["stage"].dropna().astype(str).unique())
    frecuencias_etapa = {
        etapa: contar_documentos(df.loc[df["stage"].astype(str) == etapa, "tokens"].tolist())
        for etapa in etapas
    }
    presencia = Counter()
    for conteo in frecuencias_etapa.values():
        presencia.update(conteo.keys())

    filas_tfidf = []
    cantidad_etapas = len(etapas)
    for etapa, conteo in frecuencias_etapa.items():
        total = sum(conteo.values())
        for palabra, frecuencia in conteo.items():
            if frecuencia < 3:
                continue
            idf = np.log(cantidad_etapas / presencia[palabra])
            filas_tfidf.append({
                "etapa": etapa,
                "palabra": palabra,
                "frecuencia": frecuencia,
                "etapas_con_palabra": presencia[palabra],
                "tfidf": frecuencia / total * idf,
            })
    palabras_etapa = pd.DataFrame(
        filas_tfidf, columns=["etapa", "palabra", "frecuencia", "etapas_con_palabra", "tfidf"]
    )
    if not palabras_etapa.empty:
        palabras_etapa = palabras_etapa.sort_values(
            ["etapa", "tfidf"], ascending=[True, False]
        )
    palabras_etapa.to_csv(OUT_DIR / "palabras_distintivas_por_etapa.csv", index=False)

    resumen = pd.DataFrame([{
        "partidos": len(df),
        "palabras": sum(frecuencias.values()),
        "vocabulario": len(frecuencias),
        "palabras_unicas": sum(frecuencia == 1 for frecuencia in frecuencias.values()),
        "riqueza_lexica": len(frecuencias) / sum(frecuencias.values()),
        "palabras_por_partido": np.mean([len(documento) for documento in documentos]),
        "palabra_mas_frecuente": tabla_frecuencias.iloc[0]["palabra"],
        "frecuencia_mas_alta": int(tabla_frecuencias.iloc[0]["frecuencia"]),
        "etapas": cantidad_etapas,
    }])
    resumen.to_csv(OUT_DIR / "resumen_texto.csv", index=False)

    print("Palabras más frecuentes en equipos y sedes:")
    print(tabla_frecuencias.head(20).to_string(index=False))
    print("\nBigramas más frecuentes:")
    print(tabla_bigramas.head(15).to_string(index=False))
    print("\nResumen del análisis de texto:")
    print(resumen.to_string(index=False))

    top = tabla_frecuencias.head(15).sort_values("frecuencia")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["palabra"], top["frecuencia"], color="#2980b9")
    ax.set(xlabel="Frecuencia", ylabel="Palabra", title="Palabras más frecuentes en equipos y sedes")
    ax.grid(axis="x", alpha=0.25)
    guardar(fig, "01_palabras_frecuentes.png")

    if not palabras_etapa.empty:
        principales = (
            palabras_etapa.sort_values("tfidf", ascending=False)
            .groupby("etapa")
            .head(5)
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        for etapa, grupo in principales.groupby("etapa"):
            ax.scatter(grupo["tfidf"], grupo["palabra"], label=etapa, s=35)
        ax.set(xlabel="TF-IDF", ylabel="Palabra", title="Términos distintivos por etapa")
        ax.legend(title="Etapa", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(axis="x", alpha=0.25)
        guardar(fig, "02_palabras_por_etapa.png")


if __name__ == "__main__":
    main()
