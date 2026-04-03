import sys
import os
import hashlib
import platform
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(__file__))

from metaflow import FlowSpec, step, Parameter, card, retry
import pandas as pd
import numpy as np
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from preprocessing import (
    remove_leakage, build_feature_matrix, vintage_split, FEATURE_COLS,
)
from modeling import (
    train_logistic, tune_lightgbm, tune_xgboost,
    MONOTONE,
)
from evaluation import (
    compare_models,
    optimal_threshold,
    vintage_stability, validate_monotone,
)


class CreditRiskFlow(FlowSpec):
    data_path = Parameter(
        'data_path',
        default='../data/credit_risk_data_enhanced.csv',
        help='Path to input CSV',
    )
    train_vintage_end = Parameter(
        'train_vintage_end', default=202209, type=int,
        help='Last vintage (YYYYMM) in base training set',
    )
    cal_vintage_start = Parameter(
        'cal_vintage_start', default=202210, type=int,
        help='First vintage (YYYYMM) in calibration holdout',
    )
    cal_vintage_end = Parameter(
        'cal_vintage_end', default=202212, type=int,
        help='Last vintage (YYYYMM) in calibration holdout',
    )
    test_vintage_start = Parameter(
        'test_vintage_start', default=202401, type=int,
        help='First vintage (YYYYMM) in OOT test set',
    )
    use_sample_weights = Parameter(
        'use_sample_weights', default=True, type=bool,
        help='Apply sample_weight column during training',
    )
    hpo_trials = Parameter(
        'hpo_trials', default=50, type=int,
        help='Optuna HPO trials per gradient boosting model',
    )
    save_models = Parameter(
        'save_models', default=True, type=bool,
        help='Persist model artifacts to outputs/ directory',
    )

    @card
    @retry(times=2)
    @step
    def start(self):
        print('\nLoading data')

        self.df = pd.read_csv(self.data_path)
        n, ncols = self.df.shape
        print(f'{n:,} rows, {ncols} columns')
        print(f'vintage: {self.df["vintage"].min()} to {self.df["vintage"].max()}')
        print(f'default rate: {self.df["default_12m"].mean():.2%}')

        required = [
            'loan_id', 'vintage', 'months_on_book', 'fico_score', 'income',
            'debt_to_income', 'num_open_trades', 'utilization_rate',
            'inquiries_last_6m', 'loan_amount', 'term', 'apr', 'channel',
            'product_type', 'employment_length', 'state', 'age',
            'days_past_due_current', 'total_payments_to_date',
            'sample_weight', 'default_12m',
        ]
        missing_cols = [c for c in required if c not in self.df.columns]
        assert not missing_cols, f'Missing required columns: {missing_cols}'

        dupes = self.df['loan_id'].duplicated().sum()
        assert dupes == 0, f'{dupes} duplicate loan_ids found'

        print('\nQuality checks')
        from preprocessing import SENTINEL_VALUES
        for col, val in SENTINEL_VALUES.items():
            n_sent = (self.df[col] == val).sum()
            print(f'{col} == {val}: {n_sent:,} ({n_sent / len(self.df):.1%})')

        insuf = (self.df['months_on_book'] < 12).sum()
        print(f'months_on_book < 12: {insuf:,} ({insuf / len(self.df):.1%})')

        self.next(self.preprocess)

    @retry(times=2)
    @step
    def preprocess(self):
        print('\nPreprocessing')

        df = remove_leakage(self.df)
        print('removed post-origination fields')

        self.df_train, self.df_cal, self.df_test = vintage_split(
            df,
            train_end=self.train_vintage_end,
            cal_start=self.cal_vintage_start,
            cal_end=self.cal_vintage_end,
            test_start=self.test_vintage_start,
        )

        print(
            f'train: {len(self.df_train):,} rows '
            f'({self.df_train["vintage"].min()}-{self.df_train["vintage"].max()}), '
            f'default_rate={self.df_train["default_12m"].mean():.2%}'
        )
        print(
            f'cal: {len(self.df_cal):,} rows '
            f'({self.df_cal["vintage"].min()}-{self.df_cal["vintage"].max()}), '
            f'default_rate={self.df_cal["default_12m"].mean():.2%}'
        )
        print(
            f'test: {len(self.df_test):,} rows '
            f'({self.df_test["vintage"].min()}-{self.df_test["vintage"].max()}), '
            f'default_rate={self.df_test["default_12m"].mean():.2%}'
        )

        gap = len(df) - len(self.df_train) - len(self.df_cal) - len(self.df_test)
        print(f'excluded gap: {gap:,} rows')

        self.next(self.feature_engineering)

    @retry(times=2)
    @step
    def feature_engineering(self):
        print('\nBuilding feature matrices')

        self.X_train, self.train_medians, self.train_encoders = build_feature_matrix(self.df_train)
        self.X_cal, _, _ = build_feature_matrix(self.df_cal, self.train_medians, self.train_encoders)
        self.X_test, _, _ = build_feature_matrix(self.df_test, self.train_medians, self.train_encoders)

        self.y_train = self.df_train['default_12m'].values
        self.y_cal = self.df_cal['default_12m'].values
        self.y_test = self.df_test['default_12m'].values
        self.w_train = self.df_train['sample_weight'].values if self.use_sample_weights else None

        print(f'X_train: {self.X_train.shape}, nan={self.X_train.isna().sum().sum()}')
        print(f'X_cal: {self.X_cal.shape}, nan={self.X_cal.isna().sum().sum()}')
        print(f'X_test: {self.X_test.shape}, nan={self.X_test.isna().sum().sum()}')
        print(f'features ({len(FEATURE_COLS)}): {FEATURE_COLS}')

        self.next(self.train)

    @step
    def train(self):
        from sklearn.metrics import roc_auc_score

        vintage_groups = self.df_train['vintage'].values

        print('\nTraining logistic regression')
        self.lr_raw, self.lr_cal, self.scaler = train_logistic(
            self.X_train, self.y_train, self.X_cal, self.y_cal,
            sample_weight=self.w_train,
        )
        lr_te = self.lr_cal.predict_proba(self.scaler.transform(self.X_test))[:, 1]
        print(f'logistic regression test_auc={roc_auc_score(self.y_test, lr_te):.4f}')

        print(f'\nTraining LightGBM ({self.hpo_trials} Optuna trials)')
        self.lgb_raw, self.lgb_cal, lgb_study = tune_lightgbm(
            self.X_train, self.y_train, self.X_cal, self.y_cal,
            vintage_groups=vintage_groups,
            sample_weight=self.w_train,
            n_trials=self.hpo_trials,
        )
        lgb_te = self.lgb_cal.predict_proba(self.X_test)[:, 1]
        print(
            f'lightgbm cv_auc={lgb_study.best_value:.4f} '
            f'test_auc={roc_auc_score(self.y_test, lgb_te):.4f}'
        )

        print(f'\nTraining XGBoost ({self.hpo_trials} Optuna trials)')
        self.xgb_raw, self.xgb_cal, xgb_study = tune_xgboost(
            self.X_train, self.y_train, self.X_cal, self.y_cal,
            vintage_groups=vintage_groups,
            sample_weight=self.w_train,
            n_trials=self.hpo_trials,
        )
        xgb_te = self.xgb_cal.predict_proba(self.X_test)[:, 1]
        print(
            f'xgboost cv_auc={xgb_study.best_value:.4f} '
            f'test_auc={roc_auc_score(self.y_test, xgb_te):.4f}'
        )

        self.next(self.evaluate)

    @card
    @retry(times=1)
    @step
    def evaluate(self):
        X_te_sc = self.scaler.transform(self.X_test)
        lr_te = self.lr_cal.predict_proba(X_te_sc)[:, 1]
        lgb_te = self.lgb_cal.predict_proba(self.X_test)[:, 1]
        xgb_te = self.xgb_cal.predict_proba(self.X_test)[:, 1]

        predictions = {
            'logistic_regression': lr_te,
            'lightgbm': lgb_te,
            'xgboost': xgb_te,
        }

        lr_tr = self.lr_cal.predict_proba(self.scaler.transform(self.X_train))[:, 1]
        lgb_tr = self.lgb_cal.predict_proba(self.X_train)[:, 1]
        xgb_tr = self.xgb_cal.predict_proba(self.X_train)[:, 1]
        train_preds = {
            'logistic_regression': lr_tr,
            'lightgbm': lgb_tr,
            'xgboost': xgb_tr,
        }

        metrics_table = compare_models(
            self.y_test, predictions,
            y_train=self.y_train, train_predictions=train_preds,
        )
        self.metrics = metrics_table.to_dict(orient='index')

        print('\nModel comparison')
        print(metrics_table.round(4).to_string())

        self.champion = metrics_table['auc_roc'].idxmax()
        champ_p = predictions[self.champion]
        print(
            f'\nchampion: {self.champion} '
            f'(auc={self.metrics[self.champion]["auc_roc"]:.4f}, '
            f'brier={self.metrics[self.champion]["brier"]:.5f})'
        )

        best_t, thresh_df = optimal_threshold(
            self.y_test, champ_p, method='cost', fp_cost=1.0, fn_cost=10.0
        )
        self.optimal_threshold = float(best_t)
        print(f'optimal threshold: {best_t:.4f} (fp=1, fn=10)')

        t_row = thresh_df[thresh_df['threshold'] == min(
            thresh_df['threshold'], key=lambda t: abs(t - best_t)
        )].iloc[0]
        print(
            f'precision={t_row["precision"]:.3f}, '
            f'recall={t_row["recall"]:.3f}, '
            f'capture_rate={t_row["capture_rate"]:.3f}'
        )

        test_vintages = self.df_test['vintage'].values
        stab = vintage_stability(self.y_test, champ_p, test_vintages)
        self.vintage_stability = stab.to_dict(orient='records')
        if not stab.empty:
            print(f'\nvintage stability: {self.champion}')
            print(stab.round(4).to_string(index=False))

        champ_raw = (
            self.xgb_raw if self.champion == 'xgboost'
            else self.lgb_raw if self.champion == 'lightgbm'
            else self.lr_raw
        )
        mono_df = validate_monotone(champ_raw, self.X_test, MONOTONE)
        self.monotone_report = mono_df.to_dict(orient='records')
        n_fail = (~mono_df['passes']).sum()
        if n_fail == 0:
            print('\nmonotone constraints: ok')
        else:
            print(f'\nmonotone constraints: {n_fail} violations')

        if self.champion == 'xgboost':
            self.feature_importance = dict(
                self.xgb_raw.get_booster().get_score(importance_type='gain')
            )
        elif self.champion == 'lightgbm':
            self.feature_importance = dict(
                zip(FEATURE_COLS, self.lgb_raw.feature_importances_.tolist())
            )
        else:
            self.feature_importance = dict(
                zip(FEATURE_COLS, np.abs(self.lr_raw.coef_[0]).tolist())
            )

        self.next(self.end)

    @step
    def end(self):
        out_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
        os.makedirs(out_dir, exist_ok=True)

        data_hash = hashlib.md5(
            open(os.path.join(os.path.dirname(__file__), self.data_path), 'rb').read()
        ).hexdigest()[:12]

        payload = {
            'model_version': {
                'run_id': os.environ.get('MF_RUN_ID', 'local'),
                'trained_at_utc': datetime.now(timezone.utc).isoformat(),
                'python_version': platform.python_version(),
                'data_hash_md5': data_hash,
                'data_path': self.data_path,
                'feature_version': f'v{len(FEATURE_COLS)}f',
                'hpo_trials': self.hpo_trials,
            },
            'champion': self.champion,
            'optimal_threshold': self.optimal_threshold,
            'models': self.metrics,
            'vintage_stability': self.vintage_stability,
            'monotone_report': self.monotone_report,
            'feature_importance_top10': dict(
                sorted(self.feature_importance.items(), key=lambda x: -x[1])[:10]
            ),
            'split_config': {
                'train_vintage_end': self.train_vintage_end,
                'cal_vintage_start': self.cal_vintage_start,
                'cal_vintage_end': self.cal_vintage_end,
                'test_vintage_start': self.test_vintage_start,
            },
        }

        metrics_path = os.path.join(out_dir, 'run_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f'\nmetrics saved to {metrics_path}')

        if self.save_models:
            model_path = os.path.join(out_dir, 'models.pkl')
            joblib.dump({
                'champion': self.champion,
                'lr_raw': self.lr_raw,
                'lr_cal': self.lr_cal,
                'lgb_raw': self.lgb_raw,
                'lgb_cal': self.lgb_cal,
                'xgb_raw': self.xgb_raw,
                'xgb_cal': self.xgb_cal,
                'scaler': self.scaler,
                'feature_cols': FEATURE_COLS,
                'train_medians': self.train_medians,
                'train_encoders': self.train_encoders,
                'monotone': MONOTONE,
                'threshold': self.optimal_threshold,
                'model_version': payload['model_version'],
            }, model_path)
            print(f'models saved to {model_path}')

        print('\nResults')
        for name, m in self.metrics.items():
            print(
                f'{name}: auc={m["auc_roc"]:.4f}, '
                f'ap={m["auc_pr"]:.4f}, '
                f'ks={m["ks"]:.4f}, '
                f'brier={m["brier"]:.5f}'
            )
        print(f'\nchampion: {self.champion}')
        print(f'threshold: {self.optimal_threshold:.4f}')


if __name__ == '__main__':
    CreditRiskFlow()