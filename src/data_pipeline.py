import pandas as pd
import numpy as np


def generate_mock_metrics(days: int = 7, freq_minutes: int = 1) -> pd.DataFrame:
    dates = pd.date_range(start="2026-03-01", periods=(days * 24 * 60) // freq_minutes, freq=f"{freq_minutes}min")
    df = pd.DataFrame({'timestamp': dates})

    time_numeric = np.arange(len(df))
    df['cpu_usage'] = 30 + 10 * np.sin(time_numeric / (24 * 60 / freq_minutes) * 2 * np.pi) + np.random.normal(0, 2,
                                                                                                               len(df))
    df['is_incident'] = 0

    incident_indices = np.random.choice(df.index, size=int(days / 1.5), replace=False)
    for idx in incident_indices:
        start_idx = max(0, idx - 10)
        df.loc[start_idx:idx, 'cpu_usage'] += np.linspace(5, 50, idx - start_idx + 1)
        df.loc[idx:idx + 5, 'is_incident'] = 1

    df['cpu_usage'] = df['cpu_usage'].clip(0, 100)
    return df


def create_features_and_target(df: pd.DataFrame, lead_time_minutes: int = 5) -> pd.DataFrame:
    df = df.sort_values('timestamp').copy()
    df['cpu_rolling_mean_15m'] = df['cpu_usage'].rolling(window=15, min_periods=1).mean()
    df['cpu_rolling_std_15m'] = df['cpu_usage'].rolling(window=15, min_periods=1).std().fillna(0)
    df['cpu_spike'] = df['cpu_usage'] - df['cpu_rolling_mean_15m']
    df['target_predict_incident'] = df['is_incident'].shift(-lead_time_minutes)
    return df.dropna()