"""
Compare les résultats des sweeps d'hyperparamètres.
  1) Par architecture : accuracy / F1 / precision / recall (train, val, test) pour chaque config
  2) Entre architectures : meilleur modèle de chacune (train, val, test)
Aucun entraînement : lit seulement les CSV/JSON déjà produits.
Sorties : tableaux affichés dans le terminal + PNG sauvegardés dans comparison_plots/ + affichage (plt.show).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----- À ADAPTER : nom de l'architecture -> dossier de résultats -----
ARCHS = {
    "CIFG": "results_cifg",
    "LSTM": "results_lstm",
    "LSTMP": "results_lstmp",
}
OUT = Path("comparison_plots")
OUT.mkdir(exist_ok=True)

METRICS = [("accuracy", "Accuracy"), ("f1", "Macro-F1"),
           ("precision", "Macro-Precision"), ("recall", "Macro-Recall")]
SPLITS = [("train", "Train"), ("val", "Validation"), ("test", "Test")]
SPLIT_COLORS = {"train": "tab:orange", "val": "tab:blue", "test": "tab:green"}

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 200)


def load(folder):
    folder = Path(folder)
    df = pd.read_csv(folder / "cifg_sentiment_comparison_final.csv")
    best = json.loads((folder / "final_test_metrics.json").read_text(encoding="utf-8"))["best_config_name"]
    return df, best


def show_and_save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")   # sauvegarde AVANT show (sinon image vide)
    print(f"[saved] {path}")
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------- 1) hyperparamètres, par architecture
def plot_hyperparams(arch, df, best):
    df = df.sort_values("val_f1", ascending=False).reset_index(drop=True)   # meilleure config en haut
    y = np.arange(len(df))
    h = 0.26
    offsets = [-h, 0, h]

    fig, axes = plt.subplots(1, 4, figsize=(26, max(8, len(df) * 0.5)), sharey=True)
    for ax, (m, title) in zip(axes, METRICS):
        for (s, sname), off in zip(SPLITS, offsets):
            ax.barh(y + off, df[f"{s}_{m}"], height=h, label=sname, color=SPLIT_COLORS[s])
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Score")
        ax.set_xlim(0, 1)
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["name"].astype(str), fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle(f"{arch}: Train vs Validation vs Test par configuration d'hyperparamètres "
                 f"(triées par Val Macro-F1 ; meilleur modèle = {best})", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    show_and_save(fig, OUT / f"{arch}_hyperparams_train_val_test.png")

    cols = ["name", "best_epoch"] + [f"{s}_{m}" for m, _ in METRICS for s, _ in SPLITS]
    table = df[cols].round(4)
    table.to_csv(OUT / f"{arch}_hyperparams_table.csv", index=False)
    print(f"\n===== {arch} : résultats par configuration (triés par val_f1) =====")
    print(table.to_string(index=False))
    print()


# ---------------------------------------------------------------- 2) meilleur modèle de chaque architecture
def compare_best(data):
    rows = []
    for arch, (df, best) in data.items():
        r = df[df["name"] == best].iloc[0]
        row = {"architecture": arch, "best_config": best, "params": int(r["params"]) if "params" in r else None}
        for m, _ in METRICS:
            for s, _ in SPLITS:
                row[f"{s}_{m}"] = r[f"{s}_{m}"]
        rows.append(row)
    best_df = pd.DataFrame(rows)
    best_df.round(4).to_csv(OUT / "best_models_comparison.csv", index=False)
    print("===== Meilleur modèle de chaque architecture =====")
    print(best_df.round(4).to_string(index=False))
    print()

    archs = best_df["architecture"].tolist()
    x = np.arange(len(METRICS))
    w = 0.8 / len(archs)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(1, 3, figsize=(22, 5.5), sharey=True)
    for ax, (s, sname) in zip(axes, SPLITS):
        for i, arch in enumerate(archs):
            vals = [best_df.loc[best_df["architecture"] == arch, f"{s}_{m}"].iloc[0] for m, _ in METRICS]
            label = f"{arch} ({best_df.loc[best_df['architecture'] == arch, 'best_config'].iloc[0]})"
            bars = ax.bar(x + i * w, vals, w, label=label, color=colors[i % len(colors)])
            ax.bar_label(bars, fmt="%.3f", fontsize=8)
        ax.set_xticks(x + w * (len(archs) - 1) / 2)
        ax.set_xticklabels([n for _, n in METRICS])
        ax.set_title(sname, fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, loc="lower right")
    axes[0].set_ylabel("Score")
    fig.suptitle("Comparaison des meilleurs modèles par architecture", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    show_and_save(fig, OUT / "best_models_comparison.png")


# ---------------------------------------------------------------- main
data = {}
for arch, folder in ARCHS.items():
    try:
        data[arch] = load(folder)
    except FileNotFoundError as e:
        print(f"[skip] {arch} : résultats introuvables dans '{folder}' ({e.filename})")

for arch, (df, best) in data.items():
    plot_hyperparams(arch, df, best)

if data:
    compare_best(data)
print(f"\nTout est dans ./{OUT}/")
