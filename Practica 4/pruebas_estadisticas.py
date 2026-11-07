import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "primero" / "fifa_world_cup_limpio.csv"
OUT_DIR = Path(__file__).resolve().parent / "resultados"
OUT_DIR.mkdir(exist_ok=True)
ALFA = 0.05


def grupos(var, etiqueta):
    return [
        df.loc[df[etiqueta] == g, var].dropna().values
        for g in sorted(df[etiqueta].unique())
    ]


def pooled(x, y):
    return np.sqrt(
        (((len(x) - 1) * x.var(ddof=1)) + ((len(y) - 1) * y.var(ddof=1)))
        / (len(x) + len(y) - 2)
    )


def main():
    global df
    df = pd.read_csv(CSV_PATH)
    df["es_local"] = df["team1"].eq(df["host_country"]) | df["team2"].eq(df["host_country"])
    df["total_goals_match"] = df["total_goals_team1"] + df["total_goals_team2"]

    variables = ["total_goals_match", "total_goals_team1", "total_goals_team2"]
    etapas = sorted(df["stage"].dropna().unique())

    sup = []
    for v in variables:
        gs = grupos(v, "stage")
        residuos = df[v] - df.groupby("stage")[v].transform("mean")
        W, p_norm = stats.shapiro(residuos.sample(min(500, len(residuos)), random_state=42))
        L, p_lev = stats.levene(*gs, center="median")
        sup.append({
            "variable": v,
            "shapiro_W": W,
            "shapiro_p": p_norm,
            "normal": p_norm > ALFA,
            "levene_F": L,
            "levene_p": p_lev,
            "varianzas_iguales": p_lev > ALFA,
        })
    pd.DataFrame(sup).to_csv(OUT_DIR / "supuestos.csv", index=False)
    print(pd.DataFrame(sup).to_string(index=False))

    anova = []
    for v in variables:
        gs = grupos(v, "stage")
        F, p = stats.f_oneway(*gs)
        todo = np.concatenate(gs)
        ss_ent = sum(len(g) * (g.mean() - todo.mean()) ** 2 for g in gs)
        ss_tot = ((todo - todo.mean()) ** 2).sum()
        anova.append({
            "variable": v,
            "gl_entre": len(gs) - 1,
            "gl_dentro": len(todo) - len(gs),
            "F": F,
            "p": p,
            "eta2": ss_ent / ss_tot,
            "significativo": p < ALFA,
        })
    anova_df = pd.DataFrame(anova).sort_values("F", ascending=False)
    anova_df.to_csv(OUT_DIR / "anova.csv", index=False)
    print("\nANOVA:")
    print(anova_df.to_string(index=False))

    kruskal = []
    for v in variables:
        gs = grupos(v, "stage")
        H, p = stats.kruskal(*gs)
        n = sum(len(g) for g in gs)
        kruskal.append({
            "variable": v,
            "gl": len(gs) - 1,
            "H": H,
            "p": p,
            "epsilon2": (H - len(gs) + 1) / (n - len(gs)),
            "significativo": p < ALFA,
        })
    kruskal_df = pd.DataFrame(kruskal).sort_values("H", ascending=False)
    kruskal_df.to_csv(OUT_DIR / "kruskal_wallis.csv", index=False)
    print("\nKruskal-Wallis:")
    print(kruskal_df.to_string(index=False))

    pares = list(combinations(etapas, 2))
    post = []
    for v in variables:
        for a, b in pares:
            x = df.loc[df["stage"] == a, v].dropna()
            y = df.loc[df["stage"] == b, v].dropna()
            U, p_u = stats.mannwhitneyu(x, y)
            post.append({
                "variable": v,
                "grupo_a": a,
                "grupo_b": b,
                "media_a": x.mean(),
                "media_b": y.mean(),
                "U": U,
                "p_u": p_u,
                "p_u_bonf": min(p_u * len(pares), 1),
                "significativo": p_u * len(pares) < ALFA,
            })
    post_df = pd.DataFrame(post)
    post_df.to_csv(OUT_DIR / "post_hoc.csv", index=False)
    print("\nPost hoc:")
    print(post_df.head().to_string(index=False))

    tt = []
    for v in variables:
        x = df.loc[df["es_local"], v].dropna()
        y = df.loc[~df["es_local"], v].dropna()
        t, p = stats.ttest_ind(x, y, equal_var=False)
        U, p_u = stats.mannwhitneyu(x, y)
        tt.append({
            "variable": v,
            "n_local": len(x),
            "n_visitante": len(y),
            "media_local": x.mean(),
            "media_visitante": y.mean(),
            "dif": x.mean() - y.mean(),
            "t": t,
            "p": p,
            "U": U,
            "p_u": p_u,
            "cohen_d": (x.mean() - y.mean()) / pooled(x, y),
            "significativo": p < ALFA,
        })
    tt_df = pd.DataFrame(tt)
    tt_df.to_csv(OUT_DIR / "prueba_t_mode.csv", index=False)
    print("\nT-test local vs visitante:")
    print(tt_df.to_string(index=False))

    print(f"\n{len(os.listdir(OUT_DIR))} archivos en {OUT_DIR}")


if __name__ == "__main__":
    main()
