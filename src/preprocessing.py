import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


SENTINEL_VALUES = {
    'fico_score': 99999,
    'income': -1,
    'inquiries_last_6m': 99,
}

LEAKAGE_COLS = ['days_past_due_current', 'total_payments_to_date', 'months_on_book']

FEATURE_COLS = [
    'fico_score', 'income', 'debt_to_income', 'num_open_trades',
    'utilization_rate', 'inquiries_last_6m', 'loan_amount', 'term',
    'apr', 'employment_length', 'age',
    'fico_score_missing', 'income_missing', 'inquiries_last_6m_missing',
    'loan_to_income', 'monthly_payment_burden', 'fico_tier',
    'high_utilization', 'high_inquiries', 'high_apr',
    'channel_enc', 'product_type_enc', 'state_enc',
]

NUMERIC_FILL_COLS = [
    'fico_score', 'income', 'debt_to_income', 'num_open_trades',
    'utilization_rate', 'inquiries_last_6m', 'loan_amount', 'term',
    'apr', 'employment_length', 'age',
    'loan_to_income', 'monthly_payment_burden', 'fico_tier',
]


def remove_leakage(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])


def handle_special_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, sentinel in SENTINEL_VALUES.items():
        if col not in df.columns:
            continue
        df[f'{col}_missing'] = (df[col] == sentinel).astype(int)
        df[col] = df[col].replace(sentinel, np.nan)
    return df


def engineer_features(df: pd.DataFrame, income_fill: float = None) -> pd.DataFrame:
    df = df.copy()

    if income_fill is None:
        income_fill = df['income'].median()

    income_base = df['income'].fillna(income_fill)

    df['loan_to_income'] = df['loan_amount'] / (income_base + 1)
    df['monthly_payment_burden'] = (
        (df['loan_amount'] / df['term']) /
        (income_base / 12 + 1)
    )
    df['fico_tier'] = pd.cut(
        df['fico_score'].fillna(650),
        bins=[0, 580, 620, 670, 720, 780, 900],
        labels=[0, 1, 2, 3, 4, 5],
    ).astype(float)
    df['high_utilization'] = (df['utilization_rate'] > 0.8).astype(int)
    df['high_inquiries'] = (df['inquiries_last_6m'].fillna(0) >= 5).astype(int)
    df['high_apr'] = (df['apr'] > 25).astype(int)

    return df


def encode_categoricals(
    df: pd.DataFrame,
    encoders: dict = None,
) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    encoders = {} if encoders is None else encoders
    fitted = {}

    for col in ['channel', 'product_type', 'state']:
        if col not in df.columns:
            continue

        values = df[col].fillna('Unknown').astype(str)

        if col in encoders:
            le = encoders[col]
            known = set(le.classes_)
            values = values.map(lambda x: x if x in known else 'Unknown')
            df[f'{col}_enc'] = le.transform(values)
        else:
            le = LabelEncoder()
            df[f'{col}_enc'] = le.fit_transform(values)

        fitted[col] = le

    return df, fitted


def build_feature_matrix(
    df: pd.DataFrame,
    train_medians: dict = None,
    train_encoders: dict = None,
) -> tuple[pd.DataFrame, dict, dict]:
    df = handle_special_values(df)

    income_fill = (
        df['income'].median()
        if train_medians is None
        else train_medians.get('income', df['income'].median())
    )

    df = engineer_features(df, income_fill=income_fill)
    df, encoders = encode_categoricals(df, encoders=train_encoders)

    medians = (
        {col: float(df[col].median()) for col in NUMERIC_FILL_COLS if col in df.columns}
        if train_medians is None
        else train_medians
    )

    for col in NUMERIC_FILL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(medians.get(col, 0.0))

    X = df[FEATURE_COLS].copy()
    return X, medians, encoders


def vintage_split(
    df: pd.DataFrame,
    train_end: int,
    cal_start: int,
    cal_end: int,
    test_start: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_train = df[df['vintage'] <= train_end].copy()
    df_cal = df[(df['vintage'] >= cal_start) & (df['vintage'] <= cal_end)].copy()
    df_test = df[df['vintage'] >= test_start].copy()
    return df_train, df_cal, df_test