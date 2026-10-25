
import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NEGATIVE_MAX = 4
POSITIVE_MIN = 8
SENTIMENTS = ("negative", "neutral", "positive")
COLORS = {"negative": "#d1495b", "neutral": "#edae49", "positive": "#00798c"}
TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)
TEXT_COLUMNS = ("benefitsReview", "sideEffectsReview", "commentsReview")

def rating_to_sentiment(rating: float) -> str:
    if rating <= NEGATIVE_MAX:
        return "negative"
    if rating >= POSITIVE_MIN:
        return "positive"
    return "neutral"

def clean_text(text: str) -> str:
    text = html.unescape(str(text)).strip(' "')
    return re.sub(r"\s+", " ", text)

def load_and_label(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if frame.columns[0] in ("", "Unnamed: 0"):
        frame = frame.rename(columns={frame.columns[0]: "id"})
        
    missing = set(TEXT_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    # FIX: Combine columns to create 'review'
    frame["review"] = frame[list(TEXT_COLUMNS)].agg(" ".join, axis=1)
    
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
    frame["review_clean"] = frame["review"].map(clean_text)
    unusable = frame["rating"].isna() | (frame["review_clean"] == "")
    if unusable.any():
        print(f"  {path.name}: dropping {int(unusable.sum())} rows without rating or text")
        frame = frame.loc[~unusable].copy()
    frame["rating"] = frame["rating"].astype(int)
    frame["sentiment"] = frame["rating"].map(rating_to_sentiment)
    frame["n_words"] = frame["review_clean"].map(lambda text: len(TOKEN_PATTERN.findall(text)))
    return frame

def save(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)

def plot_rating_and_sentiment(train: pd.DataFrame, test: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    counts = train["rating"].value_counts().sort_index()
    axes[0].bar(counts.index, counts.values, color=[COLORS[rating_to_sentiment(r)] for r in counts.index])
    axes[0].set_xticks(range(1, 11))
    axes[0].set_xlabel("rating")
    axes[0].set_ylabel("reviews (train)")
    axes[0].set_title("Rating distribution and the derived sentiment")
    axes[0].legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=COLORS[s]) for s in SENTIMENTS],
        labels=[f"{s}" for s in SENTIMENTS],
    )

    positions = np.arange(len(SENTIMENTS))
    for offset, (name, frame) in zip((-0.2, 0.2), (("train", train), ("test", test))):
        shares = frame["sentiment"].value_counts(normalize=True).reindex(SENTIMENTS).fillna(0)
        bars = axes[1].bar(positions + offset, shares.values, width=0.4, label=name)
        axes[1].bar_label(bars, labels=[f"{v:.0%}" for v in shares.values], fontsize=8)
    axes[1].set_xticks(positions, SENTIMENTS)
    axes[1].set_ylabel("share of reviews")
    axes[1].set_title("Class balance: train vs test")
    axes[1].legend()
    save(figure, path)

def plot_lengths(train: pd.DataFrame, path: Path) -> int:
    p95 = int(train["n_words"].quantile(0.95))
    figure, axis = plt.subplots(figsize=(7, 4.2))
    upper = int(train["n_words"].quantile(0.995))
    for sentiment in SENTIMENTS:
        axis.hist(train.loc[train["sentiment"] == sentiment, "n_words"].clip(upper=upper),
                  bins=60, alpha=0.5, label=sentiment, color=COLORS[sentiment], density=True)
    axis.axvline(p95, color="black", linestyle="--", label=f"95th percentile = {p95} words")
    axis.set_xlabel("words per review")
    axis.set_ylabel("density")
    axis.set_title("Review length (guides max_length)")
    axis.legend()
    save(figure, path)
    return p95

def plot_conditions(train: pd.DataFrame, path: Path) -> None:
    top = train["condition"].replace("", "unknown").value_counts().head(12)[::-1]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.barh(top.index, top.values, color="#4d6a8a")
    axis.set_xlabel("reviews (train)")
    axis.set_title("Most frequent conditions")
    save(figure, path)

def plot_rating_by_length(train: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.2))
    medians = train.groupby("sentiment")["n_words"].median().reindex(SENTIMENTS)
    axis.bar(SENTIMENTS, medians.values, color=[COLORS[s] for s in SENTIMENTS])
    axis.set_ylabel("median words per review")
    axis.set_title("Do negative reviews run longer?")
    save(figure, path)

def vocabulary_coverage(texts: pd.Series, sizes=(2000, 5000, 8000, 20000)) -> dict:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(TOKEN_PATTERN.findall(text.lower()))
    total = sum(counts.values())
    ranked = [count for _, count in counts.most_common()]
    return {
        "distinct_words": len(counts),
        "words_seen_at_least_twice": sum(1 for c in counts.values() if c >= 2),
        "token_coverage_by_vocab_size": {
            str(size): round(sum(ranked[:size]) / total, 4) for size in sizes
        },
    }

def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=project_dir)
    parser.add_argument("--train-file", default="drugLibTrain_raw.tsv")
    parser.add_argument("--test-file", default="drugLibTest_raw.tsv")
    parser.add_argument("--output-dir", type=Path, default=project_dir)
    args = parser.parse_args()

    train = load_and_label(args.data_dir / args.train_file)
    test = load_and_label(args.data_dir / args.test_file)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    eda_dir = args.output_dir / "eda"
    eda_dir.mkdir(exist_ok=True)
    train.to_csv(args.output_dir / "drugs_train_labeled.csv", index=False)
    test.to_csv(args.output_dir / "drugs_test_labeled.csv", index=False)

    duplicates_train = int(train["review_clean"].duplicated().sum())
    overlap = int(test["review_clean"].isin(set(train["review_clean"])).sum())
    p95 = plot_lengths(train, eda_dir / "review_length.png")
    plot_rating_and_sentiment(train, test, eda_dir / "rating_and_sentiment.png")
    plot_conditions(train, eda_dir / "top_conditions.png")
    plot_rating_by_length(train, eda_dir / "length_by_sentiment.png")

    stats = {
        "train_reviews": len(train),
        "test_reviews": len(test),
        "sentiment_rule": f"negative <= {NEGATIVE_MAX}, neutral {NEGATIVE_MAX + 1}-{POSITIVE_MIN - 1}, positive >= {POSITIVE_MIN}",
        "train_sentiment_counts": train["sentiment"].value_counts().reindex(SENTIMENTS).to_dict(),
        "test_sentiment_counts": test["sentiment"].value_counts().reindex(SENTIMENTS).to_dict(),
        "duplicate_reviews_in_train": duplicates_train,
        "test_reviews_also_in_train": overlap,
        "words_per_review": train["n_words"].describe().round(1).to_dict(),
        "words_per_review_p95": p95,
        "median_words_by_sentiment": train.groupby("sentiment")["n_words"].median().to_dict(),
        "vocabulary": vocabulary_coverage(train["review_clean"]),
    }
    (eda_dir / "eda_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"\nWrote {args.output_dir / 'drugs_train_labeled.csv'}, drugs_test_labeled.csv and {eda_dir}/")


if __name__ == "__main__":
    main()
