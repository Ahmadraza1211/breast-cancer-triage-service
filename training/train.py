"""Train candidate models, select by CV malignant recall, gate on test recall."""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

METRICS = ("recall", "accuracy", "roc_auc")


def build_candidates(hp: dict, seed: int) -> dict:
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=seed, **hp["logistic_regression"])),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=seed, **hp["random_forest"])),
        ]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="data/raw/breast_cancer.csv")
    p.add_argument("--output-dir", default="artifacts")
    p.add_argument("--model-version", default="1.0.0")
    p.add_argument("--params", default="params.yaml")
    args = p.parse_args()

    params = yaml.safe_load(Path(args.params).read_text())
    seed = params["seed"]

    df = pd.read_csv(args.data_path)
    feature_names = [c for c in df.columns if c != "target"]
    X = df[feature_names].to_numpy(dtype=float)
    # Dataset: target 0 = malignant. We make malignant the POSITIVE class (1).
    y = (df["target"] == 0).astype(int).to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=params["test_size"], stratify=y, random_state=seed
    )
    cv = StratifiedKFold(n_splits=params["cv_folds"], shuffle=True, random_state=seed)

    results = {}
    fitted = {}
    for name, pipe in build_candidates(params["models"], seed).items():
        scores = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=list(METRICS))
        cv_mean = {m: float(np.mean(scores[f"test_{m}"])) for m in METRICS}

        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]          # malignant probability
        pred = (proba >= 0.5).astype(int)
        test = {
            "recall": float(recall_score(y_te, pred)),
            "accuracy": float(accuracy_score(y_te, pred)),
            "roc_auc": float(roc_auc_score(y_te, proba)),
        }
        results[name] = {"cv_mean": cv_mean, "test": test}
        fitted[name] = pipe
        print(f"{name:22s} CV recall={cv_mean['recall']:.4f} "
              f"acc={cv_mean['accuracy']:.4f} auc={cv_mean['roc_auc']:.4f} | "
              f"TEST recall={test['recall']:.4f} acc={test['accuracy']:.4f} "
              f"auc={test['roc_auc']:.4f}")

    best = max(results, key=lambda n: (results[n]["cv_mean"]["recall"],
                                       results[n]["cv_mean"]["roc_auc"]))
    best_test = results[best]["test"]
    print(f"\nSELECTED: {best} (highest mean CV malignant recall)")

    if best_test["recall"] < params["min_test_recall"]:
        print(f"FAIL: test malignant recall {best_test['recall']:.4f} < "
              f"{params['min_test_recall']}", file=sys.stderr)
        sys.exit(1)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best], out / "model.joblib")

    card = {
        "model_version": args.model_version,
        "model_family": type(fitted[best].named_steps["clf"]).__name__,
        "positive_class": "malignant",
        "feature_names": feature_names,
        "metrics": {"cv_mean": results[best]["cv_mean"], "test": best_test},
        "seed": seed,
        "sklearn_version": sklearn.__version__,
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2))
    (out / "metrics.json").write_text(json.dumps(
        {"selected_model": best, "test": best_test, "candidates": results}, indent=2))
    print(f"Saved model.joblib, model_card.json, metrics.json -> {out}")


if __name__ == "__main__":
    main()
