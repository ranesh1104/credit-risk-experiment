import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    brier_score_loss,
)


def ks_stat(y_true, y_prob) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def compute_metrics(y_true, y_prob, label: str = 'model') -> dict:
    return {
        'label': label,
        'n': int(len(y_true)),
        'default_rate': float(y_true.mean()),
        'auc_roc': roc_auc_score(y_true, y_prob),
        'auc_pr': average_precision_score(y_true, y_prob),
        'ks': ks_stat(y_true, y_prob),
        'brier': brier_score_loss(y_true, y_prob),
    }


def compare_models(
    y_true,
    predictions: dict[str, np.ndarray],
    y_train=None,
    train_predictions: dict[str, np.ndarray] = None,
) -> pd.DataFrame:
    rows = []
    for name, te_p in predictions.items():
        row = compute_metrics(y_true, te_p, label=name)
        if y_train is not None and train_predictions and name in train_predictions:
            tr_auc = roc_auc_score(y_train, train_predictions[name])
            row['train_auc_roc'] = tr_auc
            row['overfit_gap'] = tr_auc - row['auc_roc']
        rows.append(row)
    return pd.DataFrame(rows).set_index('label')


def threshold_analysis(
    y_true,
    y_prob,
    fp_cost: float = 1.0,
    fn_cost: float = 10.0,
    thresholds: np.ndarray = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.50, 50)

    rows = []
    n_actual_defaults = y_true.sum()

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        total_cost = fp * fp_cost + fn * fn_cost

        rows.append({
            'threshold': round(float(t), 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'total_cost': int(total_cost),
            'predicted_defaults': int(y_pred.sum()),
            'captured_defaults': tp,
            'capture_rate': round(tp / n_actual_defaults, 4) if n_actual_defaults > 0 else 0.0,
            'approval_rate': round(1 - y_pred.mean(), 4),
        })

    return pd.DataFrame(rows)


def optimal_threshold(
    y_true,
    y_prob,
    method: str = 'f1',
    fp_cost: float = 1.0,
    fn_cost: float = 10.0,
    beta: float = 1.0,
) -> tuple[float, pd.DataFrame]:
    df = threshold_analysis(y_true, y_prob, fp_cost=fp_cost, fn_cost=fn_cost)

    if method == 'f1':
        best_t = df.loc[df['f1'].idxmax(), 'threshold']
    elif method == 'fbeta':
        precision = df['precision'].values
        recall = df['recall'].values
        fbeta = (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall + 1e-9)
        best_t = float(df['threshold'].iloc[np.argmax(fbeta)])
    elif method == 'cost':
        best_t = df.loc[df['total_cost'].idxmin(), 'threshold']
    elif method == 'ks':
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        best_t = float(thresholds[np.argmax(tpr - fpr)])
    else:
        raise ValueError("method must be one of: 'f1', 'fbeta', 'cost', 'ks'")

    return best_t, df


def vintage_stability(
    y_true,
    y_prob,
    vintages,
    min_defaults: int = 3,
) -> pd.DataFrame:
    rows = []

    for vintage in sorted(np.unique(vintages)):
        mask = vintages == vintage
        n_defaults = y_true[mask].sum()

        if n_defaults < min_defaults:
            continue

        try:
            rows.append({
                'vintage': int(vintage),
                'n': int(mask.sum()),
                'defaults': int(n_defaults),
                'default_rate': float(y_true[mask].mean()),
                'auc_roc': roc_auc_score(y_true[mask], y_prob[mask]),
                'ks': ks_stat(y_true[mask], y_prob[mask]),
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


def validate_monotone(
    model,
    X_ref: pd.DataFrame,
    monotone: dict[str, int],
    quantile_lo: float = 0.05,
    quantile_hi: float = 0.95,
) -> pd.DataFrame:
    X_med = X_ref.median().to_frame().T
    rows = []

    for feat, direction in monotone.items():
        if direction == 0:
            continue

        lo = X_ref[feat].quantile(quantile_lo)
        hi = X_ref[feat].quantile(quantile_hi)

        if lo == hi:
            continue

        X_lo = X_med.copy()
        X_hi = X_med.copy()
        X_lo[feat] = lo
        X_hi[feat] = hi

        p_lo = model.predict_proba(X_lo)[0, 1]
        p_hi = model.predict_proba(X_hi)[0, 1]

        passes = (p_hi >= p_lo - 1e-6) if direction == 1 else (p_lo >= p_hi - 1e-6)

        rows.append({
            'feature': feat,
            'constraint': '+1' if direction == 1 else '-1',
            'p_low': round(p_lo, 4),
            'p_high': round(p_hi, 4),
            'passes': passes,
        })

    return pd.DataFrame(rows)