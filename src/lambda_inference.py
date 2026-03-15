import os
import joblib
import pandas as pd


def load_model(model_path: str = "models/latest_model.pkl"):
    """Loads the model artifact from simulated cloud storage."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Run retraining first.")
    return joblib.load(model_path)


def lambda_handler(event, context):
    """
    AWS Lambda handler for streaming inference.
    Triggered frequently (e.g., every minute) by CloudWatch metric streams.
    """
    model = load_model()
    features = ['cpu_usage', 'cpu_rolling_mean_15m', 'cpu_rolling_std_15m', 'cpu_spike']

    # Fallback to a mock high-risk event if payload is empty during local testing
    if not event:
        event = {
            'cpu_usage': [92.0],
            'cpu_rolling_mean_15m': [40.0],
            'cpu_rolling_std_15m': [3.5],
            'cpu_spike': [52.0]
        }

    input_data = pd.DataFrame(event, columns=features)
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        print("[ALERT] High risk of incident detected within the next 5 minutes!")
        return {"statusCode": 200, "status": "ALERT_TRIGGERED"}
    else:
        print("[OK] System behavior nominal.")
        return {"statusCode": 200, "status": "SYSTEM_STABLE"}


if __name__ == "__main__":
    # Local testing execution
    lambda_handler(event={}, context={})