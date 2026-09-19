import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "primero" / "fifa_world_cup_limpio.csv"
OUT_DIR = Path(__file__).resolve().parent / "dataframes"
OUT_DIR.mkdir(exist_ok=True)


def main():
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])

    num = [
        "world_cup_year",
        "total_goals_team1",
        "total_goals_team2",
        "total_goals_match",
    ]

    resumen = df[num].describe().T
    resumen["moda"] = [df[c].mode().iloc[0] if not df[c].mode().empty else None for c in num]
    resumen["varianza"] = df[num].var(ddof=1)
    resumen["desv_est"] = df[num].std(ddof=1)
    resumen["sesgo"] = df[num].skew()
    resumen["curtosis"] = df[num].kurt()
    resumen = resumen.reset_index().rename(columns={"index": "variable"})
    resumen.to_csv(OUT_DIR / "resumen_descriptivo.csv", index=False)

    cat = ["stage", "host_country", "winner"]
    categoricas = pd.DataFrame({
        "conteo": [df[c].count() for c in cat],
        "distintos": [df[c].nunique() for c in cat],
        "moda": [df[c].mode().iloc[0] if not df[c].mode().empty else None for c in cat],
    }, index=cat).reset_index().rename(columns={"index": "variable"})
    categoricas.to_csv(OUT_DIR / "resumen_categoricas.csv", index=False)

    por_anio = (
        df.groupby("world_cup_year", as_index=False)
        .agg(
            partidos=("match_id", "count"),
            goles_total=("total_goals_match", "sum"),
            goles_promedio=("total_goals_match", "mean"),
        )
        .sort_values("world_cup_year")
    )
    por_anio.to_csv(OUT_DIR / "por_anio.csv", index=False)

    equipos = pd.concat([
        df[["team1", "total_goals_team1"]].rename(columns={"team1": "equipo", "total_goals_team1": "goles"}),
        df[["team2", "total_goals_team2"]].rename(columns={"team2": "equipo", "total_goals_team2": "goles"}),
    ])
    top_equipos = (
        equipos.groupby("equipo", as_index=False)["goles"].sum()
        .sort_values("goles", ascending=False)
        .head(20)
        .rename(columns={"goles": "goles_totales"})
    )
    top_equipos.to_csv(OUT_DIR / "top_equipos.csv", index=False)

    paises = (
        df.groupby("host_country", as_index=False)
        .agg(partidos=("match_id", "count"), goles=("total_goals_match", "sum"), promedio_goles=("total_goals_match", "mean"))
        .sort_values("partidos", ascending=False)
    )
    paises.to_csv(OUT_DIR / "por_pais.csv", index=False)

    print("Archivos generados en:", OUT_DIR)
    print(resumen.to_string(index=False))


if __name__ == "__main__":
    main()
