import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


def train_and_evaluate_model(df: pd.DataFrame, model_save_path: str = "models/latest_model.pkl"):
    """Trains the predictive model and evaluates it based on Recall and FPR."""
    # Chronological split to prevent data leakage in time-series
    split_index = int(len(df) * 0.8)
    train, test = df.iloc[:split_index], df.iloc[split_index:]

    features = ['cpu_usage', 'cpu_rolling_mean_15m', 'cpu_rolling_std_15m', 'cpu_spike']
    target = 'target_predict_incident'

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    # Initialize model with balanced class weights due to severe target imbalance
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluation focusing on JetBrains requirements (High Recall, low FPR)
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"[Evaluation] Recall: {recall:.2%} | False Positive Rate: {fpr:.2%}")

    # Persist model artifacts (simulating AWS S3 storage)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"[Storage] Model successfully saved to {model_save_path}")

    return model