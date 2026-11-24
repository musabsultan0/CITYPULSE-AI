from datetime import datetime, timezone
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "datasets" / "flood_data.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "rainfall_mm_hour",
    "water_level_m",
    "drainage_capacity_pct",
    "previous_24h_rainfall_mm",
    "soil_moisture_pct",
    "elevation_m",
]


def evaluate_candidate(name, estimator, X_train, y_train, X_test, y_test, cv):
    cv_auc = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)
    probs = estimator.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, preds).tolist()
    return estimator, {
        "model_name": name,
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
        "pr_auc": round(float(average_precision_score(y_test, probs)), 4),
        "brier_score": round(float(brier_score_loss(y_test, probs)), 4),
        "cv_roc_auc_mean": round(float(np.mean(cv_auc)), 4),
        "cv_roc_auc_std": round(float(np.std(cv_auc)), 4),
        "confusion_matrix": cm,
    }, preds


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run generate_dataset.py first.")

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df["flood_occurred"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    candidates = [
        (
            "logistic_regression_baseline",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
            ]),
        ),
        (
            "random_forest_classifier",
            RandomForestClassifier(
                n_estimators=250,
                max_depth=10,
                min_samples_split=6,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "gradient_boosting_classifier",
            GradientBoostingClassifier(
                n_estimators=180,
                learning_rate=0.06,
                max_depth=4,
                random_state=42,
            ),
        ),
    ]

    trained = []
    comparison = []
    for name, est in candidates:
        fitted, metrics, preds = evaluate_candidate(name, est, X_train, y_train, X_test, y_test, cv)
        trained.append((fitted, metrics, preds))
        comparison.append(metrics)

    # Select best model by ROC-AUC then F1 score
    best_model, best_metrics, best_preds = max(
        trained, key=lambda item: (item[1]["roc_auc"], item[1]["f1_score"])
    )

    importances = {}
    if hasattr(best_model, "feature_importances_"):
        importances = {
            feat: round(float(val), 4)
            for feat, val in sorted(
                zip(FEATURE_COLUMNS, best_model.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )
        }
    elif hasattr(best_model, "named_steps") and hasattr(best_model.named_steps.get("clf"), "coef_"):
        coefs = np.abs(best_model.named_steps["clf"].coef_[0])
        norm = coefs / max(float(np.sum(coefs)), 1e-9)
        importances = {
            feat: round(float(val), 4)
            for feat, val in sorted(zip(FEATURE_COLUMNS, norm), key=lambda item: item[1], reverse=True)
        }

    model_path = MODEL_DIR / "flood_model.joblib"
    metrics_path = MODEL_DIR / "flood_metrics.json"
    joblib.dump(best_model, model_path)

    metadata = {
        "selected_model": best_metrics["model_name"],
        "selected_metrics": best_metrics,
        "model_comparison": comparison,
        "features": FEATURE_COLUMNS,
        "feature_importances": importances,
        "dataset_rows": int(len(df)),
        "positive_rate": round(float(y.mean()), 4),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_provenance": {
            "is_synthetic_training_data": True,
            "dataset": "ml/flood/datasets/flood_data.csv",
            "note": "Synthetic demonstration dataset only. Do not use metrics as real-world flood forecasting accuracy.",
        },
    }
    metrics_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nFLOOD RISK MODEL COMPARISON (SYNTHETIC DATA)")
    print("--------------------------------------------")
    for row in comparison:
        print(
            f"{row['model_name']:30s} | Acc={row['accuracy']:.4f} | F1={row['f1_score']:.4f} "
            f"| ROC-AUC={row['roc_auc']:.4f} | CV-AUC={row['cv_roc_auc_mean']:.4f}"
        )

    print(f"\nSelected Best Model: {best_metrics['model_name']}")
    print(classification_report(y_test, best_preds, zero_division=0))
    print("Feature Importances:", importances)
    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
