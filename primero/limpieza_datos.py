from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "csv" / "fifa_world_cup_all_matches_1930_2026.csv"
OUT_PATH = ROOT / "primero" / "fifa_world_cup_limpio.csv"


def limpiar_datos(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["world_cup_year"] = pd.to_numeric(df["world_cup_year"], errors="coerce")

    for col in [
        "halftime_score_team1",
        "halftime_score_team2",
        "fulltime_score_team1",
        "fulltime_score_team2",
        "extra_time_score_team1",
        "extra_time_score_team2",
        "penalty_score_team1",
        "penalty_score_team2",
        "total_goals_team1",
        "total_goals_team2",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["total_goals_match"] = df["total_goals_team1"] + df["total_goals_team2"]
    df["winner"] = df["winner"].fillna("Sin ganador")

    df = df.drop_duplicates().reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = limpiar_datos(CSV_PATH)
    df.to_csv(OUT_PATH, index=False)

    print(f"Archivo original: {CSV_PATH.name}")
    print(f"Archivo limpio: {OUT_PATH.name}")
    print(f"Filas iniciales: {len(pd.read_csv(CSV_PATH))}")
    print(f"Filas finales: {len(df)}")
    print(f"Columnas: {list(df.columns)}")
