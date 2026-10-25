# Drug Review Sentiment Analysis Comparison

This repository compares different recurrent neural network architectures for 3-class sentiment classification on drug reviews. The workflow is:

1. prepare the raw DrugLib review data,
2. train separate models for each architecture,
3. compare the best run per architecture,
4. generate summary plots for the final comparison.

## Models compared

- LSTM
- CIFG
- LSTMP

## Project structure

- `DATA/prepare_data.py` — cleans the raw TSV data, fuses the review text fields, assigns sentiment labels from the numeric rating, and exports the processed CSVs and EDA plots.
- `DATA/drugLibTrain_raw.tsv` and `DATA/drugLibTest_raw.tsv` — original training and test datasets.
- `DATA/drugs_train_labeled.csv` and `DATA/drugs_test_labeled.csv` — labeled data used to train the models.
- `LSTM/LSTM_comparaison.py` — LSTM training and evaluation pipeline.
- `CIFG/CIFG_comparaison.py` — CIFG training and evaluation pipeline.
- `LSTMP/LSTMP_COMPARAISON.PY` — LSTMP training and evaluation pipeline.
- `plot_results.py` — reads the saved CSV/JSON result files and generates architecture comparison charts in `comparison_plots/`.
- `comparison_plots/` — final benchmark plots and exported tables.
- `requirements.txt` — project dependencies.

## Setup

Create a Python environment and install dependencies from the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Data preparation

Run the preprocessing script:

```bash
python DATA\prepare_data.py --data-dir DATA --output-dir DATA
```

This script:

- reads the raw DrugLib TSV files,
- merges `benefitsReview`, `sideEffectsReview`, and `commentsReview` into a single review field,
- cleans whitespace and HTML entities,
- converts numeric ratings into sentiment labels:
  - `negative` for ratings <= 4
  - `neutral` for ratings 5 to 7
  - `positive` for ratings >= 8
- saves the processed CSV files and EDA plots under `DATA/eda/`.

## Training

From the project root, run each architecture script separately:

```bash
python LSTM\LSTM_comparaison.py
python CIFG\CIFG_comparaison.py
python LSTMP\LSTMP_COMPARAISON.PY
```

Each script:

- loads the prepared data,
- tokenizes and encodes reviews,
- builds a vocabulary and optional GloVe embeddings,
- performs training/validation/testing for the chosen architecture,
- writes metrics and per-configuration summaries into its own `results_*` folder.

## Comparing results

After the training runs have produced their outputs, generate the final comparison plots:

```bash
python plot_results.py
```

This script reads the saved result files from each model folder, prints summary tables in the terminal, and saves figures such as:

- `comparison_plots/CIFG_hyperparams_train_val_test.png`
- `comparison_plots/LSTM_hyperparams_train_val_test.png`
- `comparison_plots/best_models_comparison.png`
- `comparison_plots/best_models_comparison.csv`

## Notes

- The model pipelines rely on tokenized review text and a vocabulary built from the training data.
- GloVe embeddings are downloaded automatically the first time they are required if they are not already present in the project folder.
- The generated comparison artifacts are useful for selecting the best-performing configuration across architectures.

## Requirements

The project depends on:

- Python 3
- PyTorch
- pandas
- NumPy
- scikit-learn
- matplotlib
