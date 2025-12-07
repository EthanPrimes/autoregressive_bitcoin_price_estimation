"""Decision tree and forest experiments driven by the provided grid.

This module loads ``Data.csv`` (date, price), engineers feature sets A/B/C,
builds decision tree and random forest models, runs time-series block
bootstraps, evaluates profit-centric metrics, and produces the plots listed in
the experiment grid.  Run directly to execute every tree/forest experiment:

    python forest_models.py --output-dir Pictures/forests --n-bootstrap 20

Dependencies: pandas, numpy, scikit-learn, matplotlib.  Install via
``pip install pandas numpy scikit-learn matplotlib`` before running.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# sklearn -- ensemble forests
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
plt.style.use("ggplot")

def load_price_data(csv_path: str = "Data.csv") -> pd.DataFrame:
    """Load Data.csv -> DataFrame indexed by date with float price."""
    df = pd.read_csv(csv_path)
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    price_col = next((c for c in df.columns if c.lower() == "price"), None)
    if price_col is None:
        raise ValueError("Expected a 'price' column in Data.csv")
    df[price_col] = (
        df[price_col]
        .astype(str)
        .str.replace(",", "")
        .astype(float)
    )
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col).set_index(date_col)
    df["return"] = df[price_col].pct_change()
    df["log_return"] = np.log(df[price_col]).diff()
    df["future_return"] = df["return"].shift(-1)
    df = df.dropna().copy()
    return df.rename(columns={price_col: "price"})

def build_features(
    df: pd.DataFrame,
    feature_set: str,
    n_lags: int = 5,
    roll_mean: int = 5,
    roll_vol: int = 20,
) -> pd.DataFrame:
    """Return feature matrix + targets for a given feature_set (A/B/C)."""
    feats = pd.DataFrame(index=df.index)
    for lag in range(1, n_lags + 1):
        feats[f"lag_ret_{lag}"] = df["return"].shift(lag)
    if feature_set in {"B", "C"}:
        feats["roll_mean"] = df["return"].rolling(roll_mean).mean()
        feats["roll_vol"] = df["return"].rolling(roll_vol).std()
    if feature_set == "C":
        dow = df.index.dayofweek
        month = df.index.month
        feats["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        feats["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        feats["month_sin"] = np.sin(2 * np.pi * month / 12)
        feats["month_cos"] = np.cos(2 * np.pi * month / 12)
    feats["target_reg"] = df["future_return"]
    feats["target_cls"] = (df["future_return"] > 0).astype(int)
    feats["price"] = df["price"]
    feats["future_return_raw"] = df["future_return"]
    feats = feats.dropna()
    return feats

def chronological_split(
    df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/val/test split."""
    n = len(df)
    t = int(train_frac * n)
    v = int(val_frac * n)
    train = df.iloc[:t]
    val = df.iloc[t : t + v]
    test = df.iloc[t + v :]
    return train, val, test

def moving_block_bootstrap(
    df: pd.DataFrame, block_size: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Time-series block bootstrap (moving blocks)."""
    if block_size <= 1:
        return df.copy()
    n = len(df)
    if n <= block_size:
        idx = rng.choice(n, size=n, replace=True)
        return df.iloc[idx].copy()
    idx: List[int] = []
    while len(idx) < n:
        start = rng.integers(0, n - block_size + 1)
        idx.extend(range(start, start + block_size))
    return df.iloc[idx[:n]].copy()

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())

@dataclass
class ProfitMetrics:
    total_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    equity: pd.Series
    pnl: pd.Series

def simulate_trading(
    returns: pd.Series,
    preds: np.ndarray,
    rule: str,
    threshold: float,
    transaction_cost: float,
    position_cap: float = 1.0,
) -> ProfitMetrics:
    """Apply trading rule to predictions and compute profit metrics."""
    preds = np.asarray(preds).reshape(-1)
    signals = np.zeros_like(preds, dtype=float)
    if rule == "binary":
        signals[preds > threshold] = 1.0
    elif rule == "prob":
        signals[preds >= threshold] = 1.0
    elif rule == "proportional":
        scale = np.tanh(preds)
        max_abs = np.max(np.abs(scale)) if len(scale) else 0
        if max_abs > 0:
            scale = scale / max_abs
        signals = np.clip(scale, -position_cap, position_cap)
    pnl = signals * returns.values
    changes = np.abs(np.diff(np.insert(signals, 0, 0)))
    pnl -= transaction_cost * changes
    equity = pd.Series(1 + pnl, index=returns.index).cumprod()
    ret = float(equity.iloc[-1] - 1)
    sharpe = 0.0
    if np.std(pnl) > 0:
        sharpe = float(np.mean(pnl) / np.std(pnl) * math.sqrt(252))
    win_rate = float(np.mean(pnl > 0))
    return ProfitMetrics(
        total_return=ret,
        sharpe=sharpe,
        max_drawdown=max_drawdown(equity),
        win_rate=win_rate,
        equity=equity,
        pnl=pd.Series(pnl, index=returns.index),
    )

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {"mae": float(mae), "rmse": float(rmse)}

def classification_metrics(
    y_true: np.ndarray, probs: np.ndarray, threshold: float
) -> Dict[str, Any]:
    pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": cm,
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics

def plot_equity_curves(curves: Dict[str, pd.Series], title: str, path: Path) -> None:
    plt.figure(figsize=(8, 4))
    for label, eq in curves.items():
        plt.plot(eq.index, eq.values, label=label)
    plt.legend()
    plt.title(title)
    plt.ylabel("Equity")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
 
# Is tomorrow up from today?
def plot_profit_box(profits: List[float], title: str, path: Path) -> None:
    plt.figure(figsize=(5, 4))
    plt.boxplot(profits, vert=True, tick_labels=["profit"])
    plt.title(title)
    plt.ylabel("Total return")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path) -> None:
    plt.figure(figsize=(5, 4))
    plt.scatter(y_true, y_pred, alpha=0.4, s=12)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def _draw_confusion(ax: plt.Axes, cm: np.ndarray, title: Optional[str] = None) -> None:
    """Render a 2x2 confusion matrix with rate labels and no colorbar."""
    cm = np.asarray(cm)
    if cm.shape != (2, 2):
        ax.imshow(cm, cmap="Blues")
        if title:
            ax.set_title(title)
        return
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0

    ax.imshow(cm, cmap="Blues", vmin=0)
    labels = [
        (0, 0, f"TN: {tn}\nTNR={tnr:.2f}"),
        (0, 1, f"FP: {fp}\nFPR={fpr:.2f}"),
        (1, 0, f"FN: {fn}\nFNR={fnr:.2f}"),
        (1, 1, f"TP: {tp}\nTPR={tpr:.2f}"),
    ]
    for i, j, text in labels:
        ax.text(j, i, text, ha="center", va="center", fontsize=9, fontweight="bold")

    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_aspect("equal")
    ax.grid(False)
    if title:
        ax.set_title(title)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)

def plot_confusion(cm: np.ndarray, title: str, path: Path, ax: Optional[plt.Axes] = None) -> None:
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    _draw_confusion(ax, cm, title=title)
    if fig is not None:
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

def plot_threshold_scan(thresholds: np.ndarray, profits: np.ndarray, title: str, path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, profits, marker="o")
    plt.xlabel("Threshold")
    plt.ylabel("Profit")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_feature_importance(
    importances: np.ndarray, feature_names: List[str], title: str, path: Path
) -> None:
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(importances)), importances[order])
    plt.xticks(range(len(importances)), np.array(feature_names)[order], rotation=60, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_roc_pr(
    y_true: np.ndarray, probs: np.ndarray, title: str, path_roc: Path, path_pr: Path
) -> None:
    fpr, tpr, _ = roc_curve(y_true, probs)
    prec, rec, _ = precision_recall_curve(y_true, probs)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC - {title}")
    plt.tight_layout()
    plt.savefig(path_roc)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR - {title}")
    plt.tight_layout()
    plt.savefig(path_pr)
    plt.close()

@dataclass
class ExperimentConfig:
    id: str
    task: str  # "R" or "C"
    feature_set: str  # "A", "B", "C"
    model_builder: Callable[[], Any]
    bootstrap_block: int
    threshold_strategy: str  # "zero", "cost", "fixed", "scan_profit", "f1", "proportional"
    profit_rule: str  # "binary", "prob", "proportional"
    description: str
    trees: Optional[int] = None
    threshold_grid: Optional[np.ndarray] = None
    max_features: Optional[Any] = None
    depth_label: Optional[str] = None
    transaction_cost: float = 0.001
    label_threshold: float = 0.0
    n_bootstrap: int = 20
    position_cap: float = 1.0
    use_oob: bool = False

def experiment_grid(cost: float, n_bootstrap: int) -> List[ExperimentConfig]:
    """Create the tree and forest experiment configs from the grid."""
    configs: List[ExperimentConfig] = [
        ExperimentConfig(
            id="T-R1",
            task="R",
            feature_set="A",
            bootstrap_block=30,
            threshold_strategy="zero",
            profit_rule="binary",
            description="Shallow DecisionTreeRegressor, buy if y_hat>0",
            model_builder=lambda: DecisionTreeRegressor(max_depth=3, random_state=42),
            depth_label="shallow",
            n_bootstrap=n_bootstrap,
        ),
        ExperimentConfig(
            id="T-R2",
            task="R",
            feature_set="B",
            bootstrap_block=30,
            threshold_strategy="cost",
            profit_rule="binary",
            description="Medium depth DecisionTreeRegressor, buy if y_hat>cost",
            model_builder=lambda: DecisionTreeRegressor(
                max_depth=5, min_samples_leaf=5, random_state=42
            ),
            depth_label="medium",
            transaction_cost=cost,
            n_bootstrap=n_bootstrap,
        ),
        ExperimentConfig(
            id="T-R3",
            task="R",
            feature_set="C",
            bootstrap_block=30,
            threshold_strategy="proportional",
            profit_rule="proportional",
            description="Deeper tree with position sizing proportional to y_hat",
            model_builder=lambda: DecisionTreeRegressor(
                max_depth=8, min_samples_leaf=10, random_state=42
            ),
            depth_label="deep",
            position_cap=1.0,
            n_bootstrap=n_bootstrap,
        ),
        ExperimentConfig(
            id="T-C1",
            task="C",
            feature_set="A",
            bootstrap_block=30,
            threshold_strategy="fixed",
            profit_rule="prob",
            description="Shallow DecisionTreeClassifier, tau=0.5",
            model_builder=lambda: DecisionTreeClassifier(max_depth=3, random_state=42),
            depth_label="shallow",
            n_bootstrap=n_bootstrap,
        ),
        ExperimentConfig(
            id="T-C2",
            task="C",
            feature_set="B",
            bootstrap_block=30,
            threshold_strategy="scan_profit",
            profit_rule="prob",
            description="Medium DecisionTreeClassifier, scan tau to maximize profit",
            model_builder=lambda: DecisionTreeClassifier(max_depth=6, random_state=42),
            depth_label="medium",
            threshold_grid=np.linspace(0.1, 0.9, 17),
            n_bootstrap=n_bootstrap,
        ),
        ExperimentConfig(
            id="T-C3",
            task="C",
            feature_set="C",
            bootstrap_block=30,
            threshold_strategy="f1",
            profit_rule="prob",
            description="Deeper DecisionTreeClassifier, tau from F1",
            model_builder=lambda: DecisionTreeClassifier(
                max_depth=10, min_samples_leaf=10, random_state=42
            ),
            depth_label="deep",
            n_bootstrap=n_bootstrap,
        ),
        ExperimentConfig(
            id="RF-R1",
            task="R",
            feature_set="B",
            bootstrap_block=45,
            threshold_strategy="zero",
            profit_rule="binary",
            description="RF regressor 50 trees, small max_features, OOB",
            model_builder=lambda: RandomForestRegressor(
                n_estimators=50,
                max_features=0.3,
                oob_score=True,
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
            ),
            trees=50,
            use_oob=True,
            n_bootstrap=n_bootstrap,
        ),
        ExperimentConfig(
            id="RF-R2",
            task="R",
            feature_set="B",
            bootstrap_block=45,
            threshold_strategy="cost",
            profit_rule="binary",
            description="RF regressor 200 trees, medium max_features, OOB",
            model_builder=lambda: RandomForestRegressor(
                n_estimators=200,
                max_features="sqrt",
                oob_score=True,
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
            ),
            trees=200,
            use_oob=True,
            transaction_cost=cost,
            n_bootstrap=n_bootstrap,
        ),
        ExperimentConfig(
            id="RF-R3",
            task="R",
            feature_set="C",
            bootstrap_block=45,
            threshold_strategy="proportional",
            profit_rule="proportional",
            description="RF regressor 500 trees, strong decorrelation, OOB",
            model_builder=lambda: RandomForestRegressor(
                n_estimators=500,
                max_features=0.2,
                oob_score=True,
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
            ),
            trees=500,
            use_oob=True,
            position_cap=1.0,
            n_bootstrap=n_bootstrap,
        ),
        ExperimentConfig(
            id="RF-C1",
            task="C",
            feature_set="A",
            bootstrap_block=45,
            threshold_strategy="fixed",
            profit_rule="prob",
            description="RF classifier 50 trees, tau=0.5, OOB",
            model_builder=lambda: RandomForestClassifier(
                n_estimators=50,
                max_features="sqrt",
                oob_score=True,
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
            ),
            trees=50,
            use_oob=True,
            n_bootstrap=n_bootstrap,
        ),
        ExperimentConfig(
            id="RF-C2",
            task="C",
            feature_set="B",
            bootstrap_block=45,
            threshold_strategy="scan_profit",
            profit_rule="prob",
            description="RF classifier 200 trees, scan tau on OOB profit",
            model_builder=lambda: RandomForestClassifier(
                n_estimators=200,
                max_features=0.3,
                oob_score=True,
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
            ),
            trees=200,
            use_oob=True,
            threshold_grid=np.linspace(0.1, 0.9, 17),
            n_bootstrap=n_bootstrap,
        ),
        ExperimentConfig(
            id="RF-C3",
            task="C",
            feature_set="C",
            bootstrap_block=45,
            threshold_strategy="f1",
            profit_rule="prob",
            description="RF classifier 500 trees, tau via F1, OOB importance",
            model_builder=lambda: RandomForestClassifier(
                n_estimators=500,
                max_features=0.2,
                oob_score=True,
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
            ),
            trees=500,
            use_oob=True,
            n_bootstrap=n_bootstrap,
        ),
    ]
    return configs

def build_targets_for_threshold(
    df: pd.DataFrame, label_threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    y_reg = df["target_reg"].values
    y_cls = (df["target_reg"].values > label_threshold).astype(int)
    return y_reg, y_cls

def choose_threshold_by_profit(
    returns: pd.Series,
    probs: np.ndarray,
    grid: np.ndarray,
    cost: float,
    profit_rule: str,
) -> float:
    best_tau = grid[0]
    best_profit = -np.inf
    for tau in grid:
        profit = simulate_trading(
            returns=returns,
            preds=probs,
            rule=profit_rule,
            threshold=tau,
            transaction_cost=cost,
        ).total_return
        if profit > best_profit:
            best_profit = profit
            best_tau = tau
    return float(best_tau)

class ExperimentRunner:
    def __init__(
        self,
        csv_path: str,
        output_dir: str,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        n_lags: int = 5,
        roll_mean: int = 5,
        roll_vol: int = 20,
        random_seed: int = 123,
    ):
        self.raw = load_price_data(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.n_lags = n_lags
        self.roll_mean = roll_mean
        self.roll_vol = roll_vol
        self.rng = np.random.default_rng(random_seed)
        self.feature_cache: Dict[str, pd.DataFrame] = {}
        self.group_results: List[Dict[str, Any]] = []

    def features(self, feature_set: str) -> pd.DataFrame:
        if feature_set not in self.feature_cache:
            feats = build_features(
                self.raw,
                feature_set=feature_set,
                n_lags=self.n_lags,
                roll_mean=self.roll_mean,
                roll_vol=self.roll_vol,
            )
            self.feature_cache[feature_set] = feats
        return self.feature_cache[feature_set].copy()

    def run_all(self, configs: Iterable[ExperimentConfig]) -> None:
        for cfg in configs:
            print(f"\n=== Running {cfg.id}: {cfg.description}")
            res = self.run_single(cfg)
            if res:
                self.group_results.append(res)
        self._plot_group_equity(task="R", filename="equity_regression.png")
        self._plot_group_equity(task="C", filename="equity_classification.png")
        self._plot_group_profit_box(task="R", filename="profit_box_regression.png")
        self._plot_group_profit_box(task="C", filename="profit_box_classification.png")
        self._plot_group_roc_pr()
        self._plot_group_confusion_matrices()

    def run_single(self, cfg: ExperimentConfig) -> None:
        feats = self.features(cfg.feature_set)
        train_full, val_full, test_full = chronological_split(
            feats, train_frac=self.train_frac, val_frac=self.val_frac
        )
        feature_cols = [c for c in feats.columns if c not in {"target_reg", "target_cls", "price", "future_return_raw"}]
        baseline_equity = (1 + test_full["target_reg"]).cumprod()

        agg_profits: List[float] = []
        agg_mae: List[float] = []
        agg_rmse: List[float] = []
        agg_f1: List[float] = []
        agg_sharpe: List[float] = []
        agg_drawdown: List[float] = []
        agg_win: List[float] = []
        oob_metrics: List[Dict[str, float]] = []

        for b in range(cfg.n_bootstrap):
            boot_rng = np.random.default_rng(self.rng.integers(0, 1_000_000))
            boot_df = moving_block_bootstrap(
                feats, cfg.bootstrap_block, boot_rng
            )
            train, val, test = chronological_split(
                boot_df, train_frac=self.train_frac, val_frac=self.val_frac
            )
            y_reg_train, y_cls_train = build_targets_for_threshold(train, cfg.label_threshold)
            y_reg_val, y_cls_val = build_targets_for_threshold(val, cfg.label_threshold)
            y_reg_test, y_cls_test = build_targets_for_threshold(test, cfg.label_threshold)

            model = cfg.model_builder()
            X_train = train[feature_cols].values
            X_val = val[feature_cols].values
            X_test = test[feature_cols].values

            if cfg.task == "R":
                model.fit(X_train, y_reg_train)
                preds_val = model.predict(X_val)
                preds_test = model.predict(X_test)
                thr = 0.0
                if cfg.threshold_strategy == "cost":
                    thr = cfg.transaction_cost
                elif cfg.threshold_strategy == "zero":
                    thr = 0.0
                elif cfg.threshold_strategy == "proportional":
                    thr = 0.0
                profit_metrics = simulate_trading(
                    returns=test["target_reg"],
                    preds=preds_test,
                    rule=cfg.profit_rule,
                    threshold=thr,
                    transaction_cost=cfg.transaction_cost,
                    position_cap=cfg.position_cap,
                )
                reg_m = regression_metrics(y_reg_test, preds_test)
                agg_mae.append(reg_m["mae"])
                agg_rmse.append(reg_m["rmse"])
                agg_profits.append(profit_metrics.total_return)
                agg_sharpe.append(profit_metrics.sharpe)
                agg_drawdown.append(profit_metrics.max_drawdown)
                agg_win.append(profit_metrics.win_rate)
                if cfg.use_oob and hasattr(model, "oob_prediction_"):
                    oob_pred = np.array(model.oob_prediction_)
                    oob_mask = ~np.isnan(oob_pred)
                    if oob_mask.sum() > 0:
                        oob_metrics.append(
                            regression_metrics(
                                y_reg_train[oob_mask], oob_pred[oob_mask]
                            )
                        )
                if b == 0:
                    plot_scatter(
                        y_reg_test,
                        preds_test,
                        title=f"{cfg.id} true vs pred",
                        path=self.output_dir / f"{cfg.id}_scatter.png",
                    )
            else:
                if len(np.unique(y_cls_train)) < 2:
                    print(f"Skipping bootstrap {b} for {cfg.id}: single-class labels")
                    continue
                model.fit(X_train, y_cls_train)
                prob_val = model.predict_proba(X_val)[:, 1]
                prob_test = model.predict_proba(X_test)[:, 1]
                if cfg.threshold_strategy == "fixed":
                    tau = 0.5
                elif cfg.threshold_strategy == "scan_profit":
                    grid = cfg.threshold_grid if cfg.threshold_grid is not None else np.linspace(0.1, 0.9, 17)
                    tau = choose_threshold_by_profit(
                        returns=val["target_reg"],
                        probs=prob_val,
                        grid=grid,
                        cost=cfg.transaction_cost,
                        profit_rule=cfg.profit_rule,
                    )
                    plot_threshold_scan(
                        grid,
                        np.array(
                            [
                                simulate_trading(
                                    returns=val["target_reg"],
                                    preds=prob_val,
                                    rule=cfg.profit_rule,
                                    threshold=t,
                                    transaction_cost=cfg.transaction_cost,
                                ).total_return
                                for t in grid
                            ]
                        ),
                        title=f"{cfg.id} profit vs tau",
                        path=self.output_dir / f"{cfg.id}_profit_vs_tau.png",
                    )
                elif cfg.threshold_strategy == "f1":
                    grid = cfg.threshold_grid if cfg.threshold_grid is not None else np.linspace(0.1, 0.9, 17)
                    f1s = []
                    for t in grid:
                        f1s.append(
                            classification_metrics(
                                y_true=y_cls_val, probs=prob_val, threshold=t
                            )["f1"]
                        )
                    best_idx = int(np.argmax(f1s))
                    tau = float(grid[best_idx])
                    plot_threshold_scan(
                        grid,
                        np.array(f1s),
                        title=f"{cfg.id} F1 vs tau",
                        path=self.output_dir / f"{cfg.id}_f1_vs_tau.png",
                    )
                else:
                    tau = 0.5
                cls_metrics = classification_metrics(y_cls_test, prob_test, tau)
                profit_metrics = simulate_trading(
                    returns=test["target_reg"],
                    preds=prob_test,
                    rule=cfg.profit_rule,
                    threshold=tau,
                    transaction_cost=cfg.transaction_cost,
                )
                agg_f1.append(cls_metrics["f1"])
                agg_profits.append(profit_metrics.total_return)
                agg_sharpe.append(profit_metrics.sharpe)
                agg_drawdown.append(profit_metrics.max_drawdown)
                agg_win.append(profit_metrics.win_rate)
                if b == 0:
                    plot_confusion(
                        cls_metrics["confusion_matrix"],
                        title=f"{cfg.id} confusion @ tau={tau:.2f}",
                        path=self.output_dir / f"{cfg.id}_cm.png",
                    )
                if cfg.use_oob and hasattr(model, "oob_decision_function_"):
                    oob_probs = model.oob_decision_function_[:, 1]
                    oob_metrics.append(
                        classification_metrics(
                            y_true=y_cls_train,
                            probs=oob_probs,
                            threshold=tau,
                        )
                    )
                if b == 0:
                    plot_roc_pr(
                        y_true=y_cls_test,
                        probs=prob_test,
                        title=cfg.id,
                        path_roc=self.output_dir / f"{cfg.id}_roc.png",
                        path_pr=self.output_dir / f"{cfg.id}_pr.png",
                    )

            if hasattr(model, "feature_importances_") and b == 0:
                plot_feature_importance(
                    model.feature_importances_,
                    feature_names=feature_cols,
                    title=f"{cfg.id} feature importance",
                    path=self.output_dir / f"{cfg.id}_importance.png",
                )

        if not agg_profits:
            print(f"No completed bootstraps for {cfg.id}; skipping summary.")
            return None
        summary: Dict[str, Any] = {
            "id": cfg.id,
            "description": cfg.description,
            "mean_profit": float(np.mean(agg_profits)),
            "median_profit": float(np.median(agg_profits)),
            "mean_sharpe": float(np.mean(agg_sharpe)) if agg_sharpe else float("nan"),
            "mean_drawdown": float(np.mean(agg_drawdown)) if agg_drawdown else float("nan"),
            "mean_win_rate": float(np.mean(agg_win)) if agg_win else float("nan"),
        }
        if cfg.task == "R":
            summary["mean_mae"] = float(np.mean(agg_mae))
            summary["mean_rmse"] = float(np.mean(agg_rmse))
            if oob_metrics:
                summary["oob_mae"] = float(
                    np.mean([m["mae"] for m in oob_metrics if "mae" in m])
                )
                summary["oob_rmse"] = float(
                    np.mean([m["rmse"] for m in oob_metrics if "rmse" in m])
                )
        else:
            summary["mean_f1"] = float(np.mean(agg_f1)) if agg_f1 else float("nan")
            if oob_metrics:
                summary["oob_accuracy"] = float(
                    np.mean([m.get("accuracy", np.nan) for m in oob_metrics])
                )
        # Chronological (non-bootstrap) run for visuals / combined plots
        chrono_result: Dict[str, Any] = {"id": cfg.id, "task": cfg.task}
        if cfg.task == "R":
            y_reg_train_full, _ = build_targets_for_threshold(train_full, cfg.label_threshold)
            y_reg_val_full, _ = build_targets_for_threshold(val_full, cfg.label_threshold)
            y_reg_test_full, _ = build_targets_for_threshold(test_full, cfg.label_threshold)
            model = cfg.model_builder()
            model.fit(train_full[feature_cols], y_reg_train_full)
            preds_test_full = model.predict(test_full[feature_cols])
            thr = 0.0 if cfg.threshold_strategy in {"zero", "proportional"} else cfg.transaction_cost
            profit_metrics_full = simulate_trading(
                returns=test_full["target_reg"],
                preds=preds_test_full,
                rule=cfg.profit_rule,
                threshold=thr,
                transaction_cost=cfg.transaction_cost,
                position_cap=cfg.position_cap,
            )
            plot_scatter(
                y_reg_test_full,
                preds_test_full,
                title=f"{cfg.id} true vs pred (chronological)",
                path=self.output_dir / f"{cfg.id}_scatter.png",
            )
            plot_equity_curves(
                {"baseline B2 (buy & hold)": baseline_equity, cfg.id: profit_metrics_full.equity},
                title=f"{cfg.id} equity vs baseline (chronological)",
                path=self.output_dir / f"{cfg.id}_equity.png",
            )
            chrono_result.update(
                equity=profit_metrics_full.equity,
                baseline=baseline_equity,
                profit=profit_metrics_full.total_return,
            )
        else:
            y_reg_train_full, y_cls_train_full = build_targets_for_threshold(train_full, cfg.label_threshold)
            y_reg_val_full, y_cls_val_full = build_targets_for_threshold(val_full, cfg.label_threshold)
            y_reg_test_full, y_cls_test_full = build_targets_for_threshold(test_full, cfg.label_threshold)
            if len(np.unique(y_cls_train_full)) >= 2:
                model = cfg.model_builder()
                model.fit(train_full[feature_cols], y_cls_train_full)
                prob_val_full = model.predict_proba(val_full[feature_cols])[:, 1]
                prob_test_full = model.predict_proba(test_full[feature_cols])[:, 1]
                if cfg.threshold_strategy == "fixed":
                    tau = 0.5
                elif cfg.threshold_strategy == "scan_profit":
                    grid = cfg.threshold_grid if cfg.threshold_grid is not None else np.linspace(0.1, 0.9, 17)
                    tau = choose_threshold_by_profit(
                        returns=val_full["target_reg"],
                        probs=prob_val_full,
                        grid=grid,
                        cost=cfg.transaction_cost,
                        profit_rule=cfg.profit_rule,
                    )
                elif cfg.threshold_strategy == "f1":
                    grid = cfg.threshold_grid if cfg.threshold_grid is not None else np.linspace(0.1, 0.9, 17)
                    f1s = [
                        classification_metrics(
                            y_true=y_cls_val_full, probs=prob_val_full, threshold=t
                        )["f1"]
                        for t in grid
                    ]
                    tau = float(grid[int(np.argmax(f1s))])
                else:
                    tau = 0.5
                cls_metrics_full = classification_metrics(y_cls_test_full, prob_test_full, tau)
                profit_metrics_full = simulate_trading(
                    returns=test_full["target_reg"],
                    preds=prob_test_full,
                    rule=cfg.profit_rule,
                    threshold=tau,
                    transaction_cost=cfg.transaction_cost,
                )
                plot_confusion(
                    cls_metrics_full["confusion_matrix"],
                    title=f"{cfg.id} confusion @ tau={tau:.2f} (chronological)",
                    path=self.output_dir / f"{cfg.id}_cm.png",
                )
                plot_roc_pr(
                    y_true=y_cls_test_full,
                    probs=prob_test_full,
                    title=f"{cfg.id} (chronological)",
                    path_roc=self.output_dir / f"{cfg.id}_roc.png",
                    path_pr=self.output_dir / f"{cfg.id}_pr.png",
                )
                plot_equity_curves(
                    {"baseline B2 (buy & hold)": baseline_equity, cfg.id: profit_metrics_full.equity},
                    title=f"{cfg.id} equity vs baseline (chronological)",
                    path=self.output_dir / f"{cfg.id}_equity.png",
                )
                chrono_result.update(
                    equity=profit_metrics_full.equity,
                    baseline=baseline_equity,
                    profit=profit_metrics_full.total_return,
                    roc=roc_curve(y_cls_test_full, prob_test_full),
                    pr=precision_recall_curve(y_cls_test_full, prob_test_full),
                    threshold=tau,
                    confusion=cls_metrics_full["confusion_matrix"],
                )
        with open(self.output_dir / f"{cfg.id}_metrics.txt", "w") as f:
            f.write("\n".join(f"{k}: {v}" for k, v in summary.items()))
        plot_profit_box(
            agg_profits,
            title=f"{cfg.id} profit distribution",
            path=self.output_dir / f"{cfg.id}_profit_box.png",
        )
        print(f"Saved metrics for {cfg.id}: {summary}")
        # Return condensed info for group plots
        chrono_result["profit_boot"] = agg_profits
        return chrono_result

    def _plot_group_equity(self, task: str, filename: str) -> None:
        subset = [r for r in self.group_results if r.get("task") == task and "equity" in r]
        if not subset:
            return
        plt.figure(figsize=(9, 5))
        baseline = subset[0].get("baseline")
        if baseline is not None:
            plt.plot(baseline.index, baseline.values, label="baseline B2 (buy & hold)", color="tab:red")
        for res in subset:
            plt.plot(res["equity"].index, res["equity"].values, label=res["id"])
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.title(f"{'Regression' if task == 'R' else 'Classification'} equity curves")
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()

    def _plot_group_profit_box(self, task: str, filename: str) -> None:
        subset = [r for r in self.group_results if r.get("task") == task and r.get("profit_boot")]
        if not subset:
            return
        data = [r["profit_boot"] for r in subset]
        labels = [r["id"] for r in subset]
        plt.figure(figsize=(9, 5))
        plt.boxplot(data, tick_labels=labels)
        plt.ylabel("Total return")
        plt.title(f"{'Regression' if task == 'R' else 'Classification'} profit distribution (bootstrap)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()

    def _plot_group_roc_pr(self) -> None:
        subset = [
            r for r in self.group_results if r.get("task") == "C" and "roc" in r and "pr" in r
        ]
        if not subset:
            return
        plt.figure(figsize=(6, 5))
        for r in subset:
            fpr, tpr, _ = r["roc"]
            plt.plot(fpr, tpr, label=r["id"])
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("Classification ROC (chronological)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "roc_all_classifiers.png")
        plt.close()

        plt.figure(figsize=(6, 5))
        for r in subset:
            precision, recall, _ = r["pr"]
            plt.plot(recall, precision, label=r["id"])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Classification PR (chronological)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "pr_all_classifiers.png")
        plt.close()

    def _plot_group_confusion_matrices(self) -> None:
        subset = [
            r
            for r in self.group_results
            if r.get("task") == "C" and "confusion" in r
        ]
        if not subset:
            return
        subset = sorted(subset, key=lambda r: r.get("id", ""))
        fig, axes = plt.subplots(2, 3, figsize=(9, 6))
        axes = axes.ravel()
        for ax, res in zip(axes, subset):
            plot_confusion(
                res["confusion"],
                title=res.get("id", ""),
                path=self.output_dir / "tmp.png",  # path ignored because ax is provided
                ax=ax,
            )
        # Hide any unused axes (if fewer than 6 classifiers)
        for ax in axes[len(subset) :]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(self.output_dir / "confusion_all_classifiers.png")
        plt.close(fig)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run decision tree and forest experiments.")
    parser.add_argument("--csv", default="Data.csv", help="Path to Data.csv")
    parser.add_argument("--output-dir", default="Pictures/forests", help="Where to save plots/metrics")
    parser.add_argument("--n-bootstrap", type=int, default=10, help="Bootstrap replicates per experiment")
    parser.add_argument("--transaction-cost", type=float, default=0.001, help="Per-trade transaction cost")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Train fraction")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Validation fraction")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    runner = ExperimentRunner(
        csv_path=args.csv,
        output_dir=args.output_dir,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )
    configs = experiment_grid(cost=args.transaction_cost, n_bootstrap=args.n_bootstrap)
    runner.run_all(configs)

if __name__ == "__main__":
    main()
