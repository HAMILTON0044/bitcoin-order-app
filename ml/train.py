import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main():
    rng = np.random.default_rng(42)
    amount = rng.uniform(100, 10000, 2000)
    volatility = rng.uniform(0, 1, 2000)
    frequency = rng.integers(1, 50, 2000)
    features = np.column_stack([amount, volatility, frequency])
    risk_flags = np.column_stack([amount > 7000, volatility > 0.65, frequency > 35])
    labels = (risk_flags.sum(axis=1) >= 2).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=42, stratify=labels
    )

    model = RandomForestClassifier(
        n_estimators=150, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    model.fit(x_train, y_train)
    metrics = {
        "dataset": "synthetic",
        "accuracy": float(accuracy_score(y_test, model.predict(x_test))),
        "training_samples": len(x_train),
        "testing_samples": len(x_test),
        "model": "RandomForestClassifier",
        "random_state": 42,
    }
    artifacts = Path(__file__).resolve().parent / "artifacts"
    artifacts.mkdir(exist_ok=True)
    joblib.dump(model, artifacts / "risk_model.joblib")
    report = json.dumps(metrics, indent=2)
    (artifacts / "metrics.json").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
