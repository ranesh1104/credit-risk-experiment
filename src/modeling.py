import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from preprocessing import FEATURE_COLS


MONOTONE: dict[str, int] = {
    'fico_score': -1,
    'income': -1,
    'debt_to_income': 1,
    'num_open_trades': 0,
    'utilization_rate': 1,
    'inquiries_last_6m': 1,
    'loan_amount': 0,
    'term': 1,
    'apr': 1,
    'employment_length': -1,
    'age': 0,
    'fico_score_missing': 1,
    'income_missing': 1,
    'inquiries_last_6m_missing': 1,
    'loan_to_income': 1,
    'monthly_payment_burden': 1,
    'fico_tier': -1,
    'high_utilization': 1,
    'high_inquiries': 1,
    'high_apr': 1,
    'channel_enc': 0,
    'product_type_enc': 0,
    'state_enc': 0,
}

MONOTONE_LIST = [MONOTONE[f] for f in FEATURE_COLS]
MONOTONE_DICT = {f: v for f, v in MONOTONE.items() if v != 0}


class VintageTimeSeriesSplit(BaseCrossValidator):
    def __init__(self, n_splits: int = 3):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        unique_vintages = np.sort(np.unique(groups))
        n_vintages = len(unique_vintages)
        fold_size = n_vintages // (self.n_splits + 1)

        for i in range(1, self.n_splits + 1):
            train_vintages = unique_vintages[:fold_size * i]
            val_vintages = unique_vintages[fold_size * i: fold_size * (i + 1)]

            if len(val_vintages) == 0:
                continue

            train_idx = np.where(np.isin(groups, train_vintages))[0]
            val_idx = np.where(np.isin(groups, val_vintages))[0]
            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class CalibratedModel:
    def __init__(self, base_model, X_cal, y_cal):
        raw_pred = base_model.predict_proba(X_cal)[:, 1]
        self.base = base_model
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(raw_pred, y_cal)

    def predict_proba(self, X):
        raw_pred = self.base.predict_proba(X)[:, 1]
        calibrated = self.calibrator.predict(raw_pred)
        return np.column_stack([1 - calibrated, calibrated])


def train_logistic(
    X_train,
    y_train,
    X_cal,
    y_cal,
    sample_weight=None,
    C: float = 0.1,
    random_state: int = 42,
) -> tuple:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_cal_scaled = scaler.transform(X_cal)

    raw_model = LogisticRegression(
        C=C,
        class_weight='balanced',
        max_iter=1000,
        solver='lbfgs',
        random_state=random_state,
    )
    raw_model.fit(X_train_scaled, y_train, sample_weight=sample_weight)

    calibrated_model = CalibratedModel(raw_model, X_cal_scaled, y_cal)
    return raw_model, calibrated_model, scaler


def tune_lightgbm(
    X_train,
    y_train,
    X_cal,
    y_cal,
    vintage_groups,
    sample_weight=None,
    n_trials: int = 50,
    random_state: int = 42,
) -> tuple:
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    cv = VintageTimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'scale_pos_weight': scale_pos_weight,
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': random_state,
            'monotone_constraints': MONOTONE_LIST,
            'monotone_constraints_method': 'advanced',
            'num_leaves': trial.suggest_int('num_leaves', 16, 64),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        }

        aucs = []
        for train_idx, val_idx in cv.split(X_train, y_train, groups=vintage_groups):
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train.iloc[train_idx],
                y_train[train_idx],
                sample_weight=sample_weight[train_idx] if sample_weight is not None else None,
                eval_set=[(X_train.iloc[val_idx], y_train[val_idx])],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
            )
            val_pred = model.predict_proba(X_train.iloc[val_idx])[:, 1]
            aucs.append(roc_auc_score(y_train[val_idx], val_pred))

        return float(np.mean(aucs))

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials)

    raw_model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        scale_pos_weight=scale_pos_weight,
        verbosity=-1,
        n_jobs=-1,
        random_state=random_state,
        monotone_constraints=MONOTONE_LIST,
        monotone_constraints_method='advanced',
        **study.best_params,
    )
    raw_model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_cal, y_cal)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    calibrated_model = CalibratedModel(raw_model, X_cal, y_cal)
    return raw_model, calibrated_model, study


def tune_xgboost(
    X_train,
    y_train,
    X_cal,
    y_cal,
    vintage_groups,
    sample_weight=None,
    n_trials: int = 50,
    random_state: int = 42,
) -> tuple:
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    cv = VintageTimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'scale_pos_weight': scale_pos_weight,
            'tree_method': 'hist',
            'random_state': random_state,
            'verbosity': 0,
            'n_jobs': -1,
            'monotone_constraints': MONOTONE_DICT,
            'early_stopping_rounds': 30,
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        }

        aucs = []
        for train_idx, val_idx in cv.split(X_train, y_train, groups=vintage_groups):
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train.iloc[train_idx],
                y_train[train_idx],
                sample_weight=sample_weight[train_idx] if sample_weight is not None else None,
                eval_set=[(X_train.iloc[val_idx], y_train[val_idx])],
                verbose=False,
            )
            val_pred = model.predict_proba(X_train.iloc[val_idx])[:, 1]
            aucs.append(roc_auc_score(y_train[val_idx], val_pred))

        return float(np.mean(aucs))

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials)

    raw_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',
        random_state=random_state,
        verbosity=0,
        n_jobs=-1,
        monotone_constraints=MONOTONE_DICT,
        early_stopping_rounds=50,
        **study.best_params,
    )
    raw_model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_cal, y_cal)],
        verbose=False,
    )

    calibrated_model = CalibratedModel(raw_model, X_cal, y_cal)
    return raw_model, calibrated_model, study