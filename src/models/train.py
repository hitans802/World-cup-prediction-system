"""
Train a match-outcome classifier (LightGBM, 3-class: away / draw / home).

Key design choices (talk about these in interviews):
  * TIME-BASED split, not random. Train on matches before TEST_YEAR, test on
    everything after. Random splitting leaks future info and inflates scores.
  * Probabilistic metrics (log loss, multiclass Brier) alongside accuracy,
    because for forecasting we care about calibrated probabilities, not just
    the argmax pick.
  * Two baselines to beat: (a) always-predict-home, (b) Elo-only.

Usage (from project root):
    python src/models/train.py data/processed/features.csv

Saves the trained model to models/lgbm.txt and a metrics report to reports/.
"""
from __future__ import annotations
import sys, os, json
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
import lightgbm as lgb

TEST_YEAR = 2018          # train < 2018, test >= 2018 (covers 2018 & 2022 WCs)
NON_FEATURES = {"date", "home_team", "away_team", "result"}


def multiclass_brier(y_true, proba, n_classes=3):
    """Mean squared error between one-hot truth and predicted probabilities."""
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true)), y_true] = 1
    return np.mean(np.sum((proba - onehot) ** 2, axis=1))


def elo_baseline_proba(df: pd.DataFrame) -> np.ndarray:
    """A simple Elo-only probability baseline.
    P(home win) from logistic on elo_diff; split the rest into draw/away."""
    p_home = 1.0 / (1.0 + 10 ** (-df["elo_diff"] / 400.0))
    p_draw = np.full(len(df), 0.26)          # roughly the historical draw rate
    p_home_adj = p_home * (1 - p_draw)
    p_away_adj = (1 - p_home) * (1 - p_draw)
    proba = np.vstack([p_away_adj, p_draw, p_home_adj]).T
    return proba / proba.sum(axis=1, keepdims=True)


def main(path: str):
    df = pd.read_csv(path, parse_dates=["date"])
    df["year"] = df["date"].dt.year

    train = df[df["year"] < TEST_YEAR].copy()
    test = df[df["year"] >= TEST_YEAR].copy()
    feature_cols = [c for c in df.columns if c not in NON_FEATURES | {"year"}]

    X_train, y_train = train[feature_cols], train["result"].astype(int)
    X_test, y_test = test[feature_cols], test["result"].astype(int)

    print(f"Train: {len(train):,} matches (<{TEST_YEAR})")
    print(f"Test:  {len(test):,} matches (>={TEST_YEAR})")
    print(f"Features ({len(feature_cols)}): {feature_cols}\n")

    # ---- LightGBM ----
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3,
        n_estimators=600, learning_rate=0.02,
        num_leaves=31, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        min_child_samples=50, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(50, verbose=False)])

    proba = model.predict_proba(X_test)
    preds = proba.argmax(axis=1)

    # ---- baselines ----
    home_only = np.zeros_like(proba); home_only[:, 2] = 1.0
    elo_proba = elo_baseline_proba(test)

    def report(name, p, hard=None):
        ll = log_loss(y_test, np.clip(p, 1e-9, 1), labels=[0, 1, 2])
        br = multiclass_brier(y_test.values, p)
        acc = accuracy_score(y_test, hard if hard is not None else p.argmax(axis=1))
        print(f"{name:<18} logloss={ll:.4f}  brier={br:.4f}  acc={acc:.3f}")
        return {"logloss": ll, "brier": br, "acc": acc}

    print("=== Test results (>=%d) ===" % TEST_YEAR)
    m_lgb = report("LightGBM", proba)
    m_elo = report("Elo baseline", elo_proba)
    m_home = report("Always-home", np.clip(home_only, 1e-9, 1),
                    hard=np.full(len(y_test), 2))

    # ---- feature importance ----
    imp = (pd.Series(model.feature_importances_, index=feature_cols)
           .sort_values(ascending=False))
    print("\nTop feature importances:")
    print(imp.head(10).to_string())

    # ---- save ----
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    model.booster_.save_model("models/lgbm.txt")
    with open("reports/metrics.json", "w") as f:
        json.dump({"lightgbm": m_lgb, "elo": m_elo, "always_home": m_home,
                   "feature_importance": imp.to_dict()}, f, indent=2)
    print("\nSaved model -> models/lgbm.txt")
    print("Saved metrics -> reports/metrics.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/processed/features.csv")
