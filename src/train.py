"""
Renku Jobs: model training entrypoint

This script is written to run as a non-interactive Renku Job. It reads its
training data from an INPUT folder and writes every result to a run-<timestamp>
subfolder inside an OUTPUT folder, so that each run's files are kept separate
from previous runs. Both folders are configurable via environment variables so
that, in RenkuLab, you can point them at mounted data connectors instead of the
copies committed to the repository:

    INPUT_DATA_DIR   where training_data.csv is read from   (default: input-data)
    OUTPUT_DATA_DIR  where the model + reports are written  (default: results)

Everything the script prints goes to the job log, so you can follow progress
from the project page while the job runs in the background.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: a job has no display
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

# --- Configuration (folders + a couple of hyperparameters) -------------------
INPUT_DIR = Path(os.environ.get("INPUT_DATA_DIR", "input-data"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DATA_DIR", "results"))
N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", "300"))
MAX_DEPTH = os.environ.get("MAX_DEPTH")  # None => unlimited
MAX_DEPTH = int(MAX_DEPTH) if MAX_DEPTH else None
RANDOM_STATE = int(os.environ.get("RANDOM_STATE", "42"))


def log(message: str) -> None:
    """Timestamped, flushed logging so lines appear live in the job log."""
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def main() -> int:
    started = time.time()
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = OUTPUT_DIR / f"run-{run_id}"

    log("Renku Jobs demo - training run starting")
    log(f"Reading input from : {INPUT_DIR.resolve()}")
    log(f"Writing output to  : {run_dir.resolve()}")

    input_csv = INPUT_DIR / "training_data.csv"
    if not input_csv.exists():
        log(f"ERROR: expected training data at {input_csv} but it was not found.")
        log("If you are running in RenkuLab, check that the input data connector "
            "is mounted and that INPUT_DATA_DIR points at it.")
        return 1

    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ----------------------------------------------------------------
    df = pd.read_csv(input_csv)
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    X = df[feature_cols]
    y = df["target"]
    log(f"Loaded {len(df)} rows, {len(feature_cols)} features, "
        f"{y.nunique()} classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    log(f"Split into {len(X_train)} train / {len(X_test)} test samples")

    # --- Train ---------------------------------------------------------------
    log(f"Training RandomForest (n_estimators={N_ESTIMATORS}, "
        f"max_depth={MAX_DEPTH}) ...")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    log("Training complete")

    # --- Evaluate ------------------------------------------------------------
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    log(f"Test accuracy : {accuracy:.4f}")
    log(f"Macro F1      : {macro_f1:.4f}")

    # --- Persist results to the run folder ------------------------------------
    model_path = run_dir / "model.joblib"
    joblib.dump(model, model_path)
    log(f"Saved model -> {model_path.name}")

    metrics = {
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(df)),
        "n_features": len(feature_cols),
        "n_classes": int(y.nunique()),
        "hyperparameters": {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "random_state": RANDOM_STATE,
        },
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log("Saved metrics -> metrics.json")

    report = classification_report(y_test, preds, digits=3)
    (run_dir / "classification_report.txt").write_text(report)
    log("Saved report -> classification_report.txt")

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion matrix for Renku Jobs demo")
    fig.tight_layout()
    fig.savefig(run_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)
    log("Saved figure -> confusion_matrix.png")

    elapsed = time.time() - started
    log(f"All results written to {run_dir.resolve()}")
    log(f"Done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
