import argparse
import json
import time
import copy
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import re
import sys

import matplotlib
matplotlib.use("Agg")  # pas d'affichage : les graphiques sont sauvegardés en PNG
import matplotlib.pyplot as plt
import urllib.request
import zipfile

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)

RESULTS_DIR = Path("results_lstmp")  # dossier propre à l'architecture LSTMP
HISTORY_DIR = RESULTS_DIR / "histories"


class Tee:
    """Écrit tout ce qui est affiché dans le terminal ET dans un fichier log."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

DEFAULTS = {
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "optimizer": "adamw",
    "weight_decay": 0.1,
    "scheduler": "plateau",
    "gradient_clip": 1.0,
    "weight_init": "default",
    "embedding_size": 100,              # = dimension de GloVe (50/100/200/300)
    "hidden_size": 32,                  # taille de la cellule c_t (et de m_t)
    "projection_size": 16,              # taille de h_t = W_p m_t + b_p (LSTMP)
    "num_layers": 1,
    "dropout": 0.5,
    "embedding_dropout": 0.3,
    "label_smoothing": 0.1,
    "max_vocabulary": 8000,
    "min_freq": 2,
    "max_length": 128,
    "validation_fraction": 0.2,
    "early_stopping_patience": 2,
    "freeze_embeddings": True,          # GloVe gelé ; False = fine-tuning avec lr x 0.1
    "class_weights": True,
    "seed": 42,
}

SWEEP = {
    "learning_rate": [5e-4, 1e-3, 5e-3],
    "optimizer": ["adamw", "rmsprop", "adam"],
    "weight_init": ["default", "xavier"],
    "batch_size": [16, 32, 64],
    "dropout": [0.3, 0.5, 0.7],
    "label_smoothing": [0.0, 0.1, 0.2],
    "hidden_size": [32, 64],
    "projection_size": [8, 16, 32],
    "max_length": [128, 256],
    "weight_decay": [0.05, 0.1],
    "freeze_embeddings": [True, False],
    "class_weights": [True, False],
}

def build_experiments() -> list[dict]:
    experiments = [{"name": "baseline"}]
    for key, values in SWEEP.items():
        for value in values:
            if value == DEFAULTS[key]:
                continue
            experiments.append({"name": f"{key}-{value}", key: value})
    return experiments

def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).lower())

def load_sentiment_data(path: Path) -> tuple[list[str], list[int]]:
    df = pd.read_csv(path)
    texts = df["review_clean"].tolist()
    labels_map = {"negative": 0, "neutral": 1, "positive": 2}
    labels = [labels_map[s] for s in df["sentiment"]]
    return texts, labels

def make_vocabulary(texts: Sequence[str], max_vocabulary: int, min_freq: int) -> dict[str, int]:
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))
    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, count in counts.most_common(max_vocabulary - len(vocabulary)):
        if count < min_freq: break
        vocabulary[token] = len(vocabulary)
    return vocabulary

def encode_texts(texts: Sequence[str], vocabulary: dict[str, int], max_length: int):
    encoded, lengths = [], []
    for text in texts:
        token_ids = [vocabulary.get(token, vocabulary[UNK_TOKEN]) for token in tokenize(text)]
        if not token_ids: token_ids = [vocabulary[UNK_TOKEN]]
        token_ids = token_ids[:max_length]
        lengths.append(len(token_ids))
        token_ids.extend([vocabulary[PAD_TOKEN]] * (max_length - len(token_ids)))
        encoded.append(token_ids)
    return torch.tensor(encoded, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

_GLOVE_CACHE = {}

def load_glove_matrix(vocabulary: dict[str, int], dim: int) -> np.ndarray:
    """Matrice (vocab_size x dim) initialisée avec GloVe ; mots absents = petits vecteurs aléatoires."""
    key = (dim, tuple(vocabulary))
    if key in _GLOVE_CACHE:
        return _GLOVE_CACHE[key]

    path = Path(f"glove.6B.{dim}d.txt")
    if not path.exists():
        zip_path = Path("glove.6B.zip")
        if not zip_path.exists():
            print("Downloading GloVe (~820 MB, une seule fois)...")
            for url in ("https://nlp.stanford.edu/data/glove.6B.zip",
                        "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"):
                try:
                    urllib.request.urlretrieve(url, zip_path)
                    break
                except Exception as e:
                    print(f"  failed ({url}): {e}")
            else:
                raise SystemExit(f"Telecharge glove.6B.zip a la main et extrais {path.name} a cote du script.")
        with zipfile.ZipFile(zip_path) as z:
            z.extract(path.name)

    matrix = np.random.RandomState(0).normal(0, 0.1, (len(vocabulary), dim)).astype("float32")
    found = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            word, rest = line.rstrip().split(" ", 1)
            if word in vocabulary:
                matrix[vocabulary[word]] = np.asarray(rest.split(), dtype="float32")
                found += 1
    matrix[vocabulary[PAD_TOKEN]] = 0.0
    print(f"GloVe coverage: {found}/{len(vocabulary)} mots ({100 * found / len(vocabulary):.1f}%)")
    _GLOVE_CACHE[key] = matrix
    return matrix


# ---------------------------------------------------------------------------
# ARCHITECTURE : LSTM avec projection (LSTMP, Sak et al. 2014)
#   i_t = σ(W_xi x_t + W_hi h_{t-1} + b_i)
#   f_t = σ(W_xf x_t + W_hf h_{t-1} + b_f)
#   o_t = σ(W_xo x_t + W_ho h_{t-1} + b_o)
#   g_t = tanh(W_xg x_t + W_hg h_{t-1} + b_g)
#   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
#   m_t = o_t ⊙ tanh(c_t)
#   h_t = W_p m_t + b_p          <- projection : h_t est plus petit que m_t
# La récurrence utilise h_{t-1} (taille projection_size), donc W_h* est (4*hidden x proj).
# ---------------------------------------------------------------------------
class LSTMPCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, proj_size: int):
        super().__init__()
        self.cell_size = hidden_size                     # taille de c_t et m_t
        self.out_size = proj_size                        # taille de h_t
        self.input_gates = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_gates = nn.Linear(proj_size, 4 * hidden_size, bias=False)
        self.projection = nn.Linear(hidden_size, proj_size)   # W_p, b_p

    def forward(self, x_t, h_prev, c_prev):
        gates = self.input_gates(x_t) + self.hidden_gates(h_prev)
        i_gate, f_gate, o_gate, g_gate = gates.chunk(4, dim=-1)
        i_t, f_t, o_t = torch.sigmoid(i_gate), torch.sigmoid(f_gate), torch.sigmoid(o_gate)
        g_t = torch.tanh(g_gate)
        c_t = f_t * c_prev + i_t * g_t
        m_t = o_t * torch.tanh(c_t)
        h_t = self.projection(m_t)
        return h_t, c_t

class LSTMPClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, proj_size, num_classes, num_layers, dropout, emb_dropout,
                 emb_matrix=None, freeze_emb=False):
        super().__init__()
        if emb_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(emb_matrix), freeze=freeze_emb, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.embedding_dropout = nn.Dropout(emb_dropout)
        self.cells = nn.ModuleList(
            LSTMPCell(emb_size if i == 0 else proj_size, hidden_size, proj_size) for i in range(num_layers))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.cells[-1].out_size, num_classes)

    def forward(self, token_ids, lengths):
        embedded = self.embedding_dropout(self.embedding(token_ids))
        batch_size, seq_len, _ = embedded.shape
        hidden_states = [embedded.new_zeros((batch_size, c.out_size)) for c in self.cells]
        cell_states = [embedded.new_zeros((batch_size, c.cell_size)) for c in self.cells]

        active = torch.arange(seq_len, device=token_ids.device).unsqueeze(0) < lengths.unsqueeze(1)

        for pos in range(seq_len):
            layer_input = embedded[:, pos, :]
            active_pos = active[:, pos].unsqueeze(1)
            for layer, cell in enumerate(self.cells):
                next_h, next_c = cell(layer_input, hidden_states[layer], cell_states[layer])
                hidden_states[layer] = torch.where(active_pos, next_h, hidden_states[layer])
                cell_states[layer] = torch.where(active_pos, next_c, cell_states[layer])
                layer_input = hidden_states[layer]
                if layer < len(self.cells) - 1:
                    layer_input = self.dropout(layer_input)

        return self.classifier(self.dropout(hidden_states[-1]))


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------
def compute_metrics(targets, preds) -> dict:
    """Accuracy + precision / recall / F1 (macro)."""
    return {
        "accuracy": accuracy_score(targets, preds),
        "precision": precision_score(targets, preds, average="macro", zero_division=0),
        "recall": recall_score(targets, preds, average="macro", zero_division=0),
        "f1": f1_score(targets, preds, average="macro", zero_division=0),
    }


def balanced_ce(logits, targets, class_weights):
    """CE pure (sans label smoothing), pondérée par classe. Retourne (somme pondérée, somme des poids)."""
    w = class_weights[targets]
    return (F.cross_entropy(logits, targets, reduction="none") * w).sum().item(), w.sum().item()


@torch.no_grad()
def evaluate(model, loader, device, class_weights):
    """Loss (CE pondérée par classe) + métriques sur un loader (validation ou test)."""
    model.eval()
    loss_sum, n = 0.0, 0.0
    preds, targs = [], []
    for t_ids, lens, lbls in loader:
        lbls_d = lbls.to(device)
        logits = model(t_ids.to(device), lens.to(device))
        ls, ws = balanced_ce(logits, lbls_d, class_weights)
        loss_sum += ls
        n += ws
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        targs.extend(lbls.tolist())
    metrics = compute_metrics(targs, preds)
    metrics["loss"] = loss_sum / n
    return metrics, preds, targs


def run_experiment(config, train_texts, train_labels, test_texts, test_labels, device):
    print(f"Running: {config['name']}")
    torch.manual_seed(config['seed'])

    train_idx, val_idx = train_test_split(
        np.arange(len(train_texts)),
        test_size=config["validation_fraction"],
        stratify=train_labels,
        random_state=config["seed"]
    )

    fit_texts = [train_texts[i] for i in train_idx]
    fit_labels = [train_labels[i] for i in train_idx]
    val_texts = [train_texts[i] for i in val_idx]
    val_labels = [train_labels[i] for i in val_idx]

    vocab = make_vocabulary(fit_texts, config["max_vocabulary"], config["min_freq"])

    def make_loader(txt, lbl, shuffle):
        t_ids, lens = encode_texts(txt, vocab, config["max_length"])
        return DataLoader(TensorDataset(t_ids, lens, torch.tensor(lbl)), batch_size=config["batch_size"], shuffle=shuffle)

    train_loader = make_loader(fit_texts, fit_labels, True)
    val_loader = make_loader(val_texts, val_labels, False)
    test_loader = make_loader(test_texts, test_labels, False)

    # poids de classes = inverse de la fréquence dans le train
    counts = np.bincount(fit_labels, minlength=3)
    class_weights = torch.tensor(len(fit_labels) / (3.0 * counts), dtype=torch.float32, device=device)

    # embeddings GloVe pré-entraînés
    emb_matrix = load_glove_matrix(vocab, config["embedding_size"])
    model = LSTMPClassifier(len(vocab), config["embedding_size"], config["hidden_size"], config["projection_size"],
                           3, config["num_layers"],
                           config["dropout"], config["embedding_dropout"],
                           emb_matrix=emb_matrix, freeze_emb=config["freeze_embeddings"]).to(device)

    if config["weight_init"] == "xavier":
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    groups = [
        {"params": [p for n, p in model.named_parameters() if not n.startswith("embedding.")]},
        {"params": list(model.embedding.parameters()), "lr": config["learning_rate"] * 0.1},
    ]
    if config["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(groups, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(groups, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == "rmsprop":
        optimizer = torch.optim.RMSprop(groups, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    else:
        optimizer = torch.optim.SGD(groups, lr=config["learning_rate"])

    # scheduler + early stopping suivent la val loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=0)
    criterion_train = nn.CrossEntropyLoss(weight=class_weights if config["class_weights"] else None,
                                          label_smoothing=config["label_smoothing"])

    best_val_loss, best_epoch, patience_counter = float("inf"), 0, 0
    best_state, best_row = None, None
    history = []
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for epoch in range(1, config["epochs"] + 1):
        # ---------------- TRAIN ----------------
        model.train()
        lr_used = optimizer.param_groups[0]["lr"]
        for t_ids, lens, lbls in train_loader:
            lbls_d = lbls.to(device)
            optimizer.zero_grad()
            logits = model(t_ids.to(device), lens.to(device))
            loss = criterion_train(logits, lbls_d)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()

        # ---------------- EVALUATION (mêmes conditions pour train et val) ----------------
        train_m, _, _ = evaluate(model, train_loader, device, class_weights)
        val_m, _, _ = evaluate(model, val_loader, device, class_weights)
        scheduler.step(val_m["loss"])

        row = {
            "epoch": epoch,
            "lr": lr_used,
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}": v for k, v in val_m.items()},
        }
        history.append(row)
        print(
            f"  epoch {epoch:02d} | "
            f"train loss {train_m['loss']:.4f} acc {train_m['accuracy']:.4f} "
            f"P {train_m['precision']:.4f} R {train_m['recall']:.4f} F1 {train_m['f1']:.4f} | "
            f"val loss {val_m['loss']:.4f} acc {val_m['accuracy']:.4f} "
            f"P {val_m['precision']:.4f} R {val_m['recall']:.4f} F1 {val_m['f1']:.4f}"
        )

        if val_m["loss"] < best_val_loss - 1e-4:
            best_val_loss = val_m["loss"]
            best_epoch = epoch
            best_row = row
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                break

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(HISTORY_DIR / f"history_{config['name']}.csv", index=False)

    # ---------------- TEST (meilleur epoch) ----------------
    if best_state is not None:
        model.load_state_dict(best_state)
    test_m, _, _ = evaluate(model, test_loader, device, class_weights)

    result_dict = {
        "name": config["name"],
        "best_epoch": best_epoch,
        "train_loss": best_row["train_loss"],
        "val_loss": best_row["val_loss"],
        "train_accuracy": best_row["train_accuracy"],
        "val_accuracy": best_row["val_accuracy"],
        "train_precision": best_row["train_precision"],
        "val_precision": best_row["val_precision"],
        "train_recall": best_row["train_recall"],
        "val_recall": best_row["val_recall"],
        "train_f1": best_row["train_f1"],
        "val_f1": best_row["val_f1"],
        "test_loss": test_m["loss"],
        "test_accuracy": test_m["accuracy"],
        "test_precision": test_m["precision"],
        "test_recall": test_m["recall"],
        "test_f1": test_m["f1"],
        "f1_gap_train_minus_val": best_row["train_f1"] - best_row["val_f1"],
        "params": params,
    }
    return result_dict, best_state, model, test_loader, class_weights


# ---------------------------------------------------------------------------
# Graphiques (courbes train/val, comparaison avec d'autres architectures)
# Pour comparer : dans une cellule Colab, après avoir lancé les deux architectures :
#   make_plots(argparse.Namespace(label="LSTMP", compare=["LSTM=results_lstm"], plots_only=True))
# ---------------------------------------------------------------------------
CLASS_NAMES = ["Negative (0-4)", "Neutral (5-7)", "Positive (8-10)"]
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def slug(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def find_history(folder, name):
    for d in (folder / "histories", folder):
        for n in (name, name.replace(".", "_")):
            p = d / f"history_{n}.csv"
            if p.exists():
                return p
    return None


def load_run(label, folder):
    folder = Path(folder)
    metrics = json.loads((folder / "final_test_metrics.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(folder / "cifg_sentiment_comparison_final.csv")
    best = metrics["best_config_name"]
    row = summary[summary["name"] == best].iloc[0]
    hist_path = find_history(folder, best)
    if hist_path is None:
        raise SystemExit(f"[{label}] history file for '{best}' not found in {folder}")
    histories = {}
    for name in summary["name"]:
        p = find_history(folder, name)
        if p is not None:
            histories[name] = pd.read_csv(p)
    return dict(
        label=label, folder=folder, best=best, metrics=metrics, summary=summary,
        hist=pd.read_csv(hist_path), histories=histories,
        best_epoch=int(row["best_epoch"]), params=int(row["params"]) if "params" in row else None,
        report=pd.read_csv(folder / "final_test_report.csv", index_col=0),
        cm=pd.read_csv(folder / "final_test_confusion_matrix.csv", index_col=0),
    )


def legend_name(r):
    return f"{r['label']} ({r['params']:,} params)" if r["params"] else r["label"]


def plot_best_curves(r, out):
    h = r["hist"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in zip(axes, ["loss", "f1"], ["Loss (cross-entropy)", "Macro F1"]):
        ax.plot(h["epoch"], h[f"train_{key}"], "o--", label="train")
        ax.plot(h["epoch"], h[f"val_{key}"], "o-", label="validation")
        ax.axvline(r["best_epoch"], color="gray", ls=":", label=f"best epoch ({r['best_epoch']})")
        ax.set_xlabel("epoch")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"{r['label']} - best config: {r['best']}")
    fig.tight_layout()
    fig.savefig(out / f"curves_{slug(r['label'])}.png", dpi=150)
    plt.close(fig)


def plot_compare_curves(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title in zip(axes, ["loss", "f1"], ["Loss", "Macro F1"]):
        for i, r in enumerate(runs):
            c = COLORS[i % len(COLORS)]
            ax.plot(r["hist"]["epoch"], r["hist"][f"val_{key}"], "-", color=c, label=f"{legend_name(r)} - val")
            ax.plot(r["hist"]["epoch"], r["hist"][f"train_{key}"], "--", color=c, alpha=0.6,
                    label=f"{r['label']} - train")
        ax.set_xlabel("epoch")
        ax.set_title(title + " (solid = validation, dashed = train)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "compare_curves.png", dpi=150)
    plt.close(fig)


def plot_compare_test(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    width = 0.8 / len(runs)

    names = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(names))
    for i, r in enumerate(runs):
        vals = [r["metrics"][f"test_{n}"] for n in names]
        bars = axes[0].bar(x + i * width, vals, width, label=legend_name(r), color=COLORS[i % len(COLORS)])
        axes[0].bar_label(bars, fmt="%.3f", fontsize=7)
    axes[0].set_xticks(x + width * (len(runs) - 1) / 2, ["accuracy", "precision (macro)", "recall (macro)", "F1 (macro)"])
    axes[0].set_title("Test metrics")

    x = np.arange(len(CLASS_NAMES))
    for i, r in enumerate(runs):
        vals = [r["report"].loc[c, "f1-score"] for c in CLASS_NAMES]
        bars = axes[1].bar(x + i * width, vals, width, label=legend_name(r), color=COLORS[i % len(COLORS)])
        axes[1].bar_label(bars, fmt="%.3f", fontsize=7)
    axes[1].set_xticks(x + width * (len(runs) - 1) / 2, CLASS_NAMES)
    axes[1].set_title("Test F1 per class")

    for ax in axes:
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "compare_test.png", dpi=150)
    plt.close(fig)


def plot_compare_confusion(runs, out):
    fig, axes = plt.subplots(1, len(runs), figsize=(4.6 * len(runs), 4.2), squeeze=False)
    for ax, r in zip(axes[0], runs):
        cm = r["cm"].values.astype(float)
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{norm[i, j]:.0%}\n({int(cm[i, j])})", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xticks(range(3), ["Neg", "Neu", "Pos"])
        ax.set_yticks(range(3), ["Neg", "Neu", "Pos"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{r['label']}\nF1 macro {r['metrics']['test_f1']:.3f}")
    fig.tight_layout()
    fig.savefig(out / "compare_confusion.png", dpi=150)
    plt.close(fig)


def plot_sweep_overview(r, out):
    s = r["summary"].sort_values("val_f1")
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(s) + 1.5))
    ax.barh(y + 0.2, s["train_f1"], 0.4, label="train F1", color="tab:orange", alpha=0.8)
    ax.barh(y - 0.2, s["val_f1"], 0.4, label="val F1", color="tab:blue")
    ax.set_yticks(y, s["name"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("macro F1 at best epoch (the gap between the bars = overfitting)")
    ax.set_title(f"{r['label']} - sweep overview")
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"sweep_{slug(r['label'])}_overview.png", dpi=150)
    plt.close(fig)


def plot_sweep_params(r, out):
    folder = out / f"sweep_{slug(r['label'])}"
    folder.mkdir(exist_ok=True)
    groups = {}
    for name in r["histories"]:
        if name != "baseline":
            groups.setdefault(name.rsplit("-", 1)[0], []).append(name)
    for key, names in groups.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric, title in zip(axes, ["loss", "f1"], ["Validation loss", "Validation macro F1"]):
            if "baseline" in r["histories"]:
                h = r["histories"]["baseline"]
                ax.plot(h["epoch"], h[f"val_{metric}"], "k-", lw=2, label="baseline (default)")
            for i, name in enumerate(names):
                h = r["histories"][name]
                ax.plot(h["epoch"], h[f"val_{metric}"], "o-", ms=3, color=COLORS[i % len(COLORS)],
                        label=f"{key} = {name.rsplit('-', 1)[1]}")
            ax.set_xlabel("epoch")
            ax.set_title(title)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(f"{r['label']} - effect of {key}")
        fig.tight_layout()
        fig.savefig(folder / f"{slug(key)}.png", dpi=150)
        plt.close(fig)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="LSTMP", help="nom de cette architecture dans les graphiques")
    ap.add_argument("--compare", action="append", default=[], metavar="LABEL=FOLDER",
                    help="dossier de résultats d'une autre architecture (option répétable)")
    ap.add_argument("--plots-only", action="store_true",
                    help="ne pas réentraîner : refait seulement les graphiques")
    # Colab/Jupyter : on ignore les arguments du noyau
    args, unknown = ap.parse_known_args()
    return args


def make_plots(args):
    out = RESULTS_DIR / "plots"
    out.mkdir(parents=True, exist_ok=True)
    specs = [(args.label, RESULTS_DIR)]
    for spec in args.compare:
        label, _, folder = spec.partition("=")
        if not folder:
            raise SystemExit(f"--compare attend LABEL=FOLDER, recu '{spec}'")
        specs.append((label.strip(), folder.strip()))

    runs = [load_run(label, folder) for label, folder in specs]
    for r in runs:
        plot_best_curves(r, out)
        plot_sweep_overview(r, out)
        plot_sweep_params(r, out)
    plot_compare_curves(runs, out)
    plot_compare_test(runs, out)
    plot_compare_confusion(runs, out)
    print(f"Plots saved in {out}/")


if __name__ == "__main__":
    args = parse_args()
    if args.plots_only:
        make_plots(args)
    else:
        RESULTS_DIR.mkdir(exist_ok=True)
        log_file = open(RESULTS_DIR / "run_log.txt", "w", encoding="utf-8")
        sys.stdout = Tee(sys.__stdout__, log_file)

        print("Loading sentiment CSVs...")
        try:
            train_t, train_l = load_sentiment_data(Path("drugs_train_labeled.csv"))
            test_t, test_l = load_sentiment_data(Path("drugs_test_labeled.csv"))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"\n--> DEVICE IN USE: {device} <--\n")

            results = []
            global_best_loss = float("inf")
            global_best_f1 = 0.0
            global_best_cw = None
            global_best_state = None
            global_best_name = ""
            global_best_conf = None
            global_best_model = None
            global_best_loader = None

            print("\n--- 1. HYPERPARAMETER SWEEP ---")
            for conf in build_experiments():
                full_conf = {**DEFAULTS, **conf}
                res, state, model, t_loader, cw = run_experiment(full_conf, train_t, train_l, test_t, test_l, device)
                results.append(res)

                if res["val_loss"] < global_best_loss:
                    global_best_loss = res["val_loss"]
                    global_best_f1 = res["val_f1"]
                    global_best_cw = cw
                    global_best_state = state
                    global_best_name = res["name"]
                    global_best_conf = full_conf
                    global_best_model = model
                    global_best_loader = t_loader

            df = pd.DataFrame(results)
            df.to_csv(RESULTS_DIR / "cifg_sentiment_comparison_final.csv", index=False)  # nom conservé pour --compare
            print(f"\nSweep Complete. Comparison saved to {RESULTS_DIR / 'cifg_sentiment_comparison_final.csv'}")
            print(f"Per-epoch histories saved in {HISTORY_DIR}/")

            print(f"\n--- 2. FINAL GLOBAL TEST EVALUATION ---")
            print(f"Best configuration overall: {global_best_name} (Val loss: {global_best_loss:.4f} | Val F1: {global_best_f1:.4f})")

            global_best_model.load_state_dict(global_best_state)
            test_m, test_preds, test_targs = evaluate(global_best_model, global_best_loader, device, global_best_cw)

            print(f"Test loss: {test_m['loss']:.4f} | acc: {test_m['accuracy']:.4f} | "
                  f"P: {test_m['precision']:.4f} | R: {test_m['recall']:.4f} | F1: {test_m['f1']:.4f}")

            labels_names = CLASS_NAMES
            report = classification_report(test_targs, test_preds, target_names=labels_names, digits=4)
            print("\nFinal Test Classification Report:")
            print(report)

            (RESULTS_DIR / "final_test_report.txt").write_text(
                f"Best configuration: {global_best_name}\n\n{report}", encoding="utf-8"
            )
            report_dict = classification_report(test_targs, test_preds, target_names=labels_names, digits=4, output_dict=True)
            pd.DataFrame(report_dict).T.to_csv(RESULTS_DIR / "final_test_report.csv")

            cm = confusion_matrix(test_targs, test_preds)
            pd.DataFrame(cm, index=labels_names, columns=labels_names).to_csv(RESULTS_DIR / "final_test_confusion_matrix.csv")

            with open(RESULTS_DIR / "final_test_metrics.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "best_config_name": global_best_name,
                        "best_config": global_best_conf,
                        "best_val_loss": float(global_best_loss),
                        "best_val_f1": float(global_best_f1),
                        **{f"test_{k}": float(v) for k, v in test_m.items()},
                    },
                    f, indent=2,
                )

            torch.save(global_best_state, RESULTS_DIR / "global_best_lstmp_model.pt")
            print(f"\nAll results saved in ./{RESULTS_DIR}/")

            print("\n--- 3. PLOTS ---")
            make_plots(args)

        except FileNotFoundError as e:
            print(e)
        finally:
            sys.stdout = sys.__stdout__
            log_file.close()