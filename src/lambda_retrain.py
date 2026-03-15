from src.data_pipeline import generate_mock_metrics, create_features_and_target
from src.train import train_and_evaluate_model


def lambda_handler(event, context):
    """
    AWS Lambda handler for periodic model retraining.
    Triggered via Amazon EventBridge (e.g., daily schedule).
    """
    print("[Lambda: Retrain] Initializing scheduled retraining job...")

    # 1. Fetch latest historical data
    raw_data = generate_mock_metrics(days=30)

    # 2. Process time-series features
    ml_data = create_features_and_target(raw_data, lead_time_minutes=5)

    # 3. Train and persist model artifacts
    train_and_evaluate_model(ml_data, model_save_path="models/latest_model.pkl")

    return {
        "statusCode": 200,
        "body": "Periodic retraining completed successfully."
    }


if __name__ == "__main__":
    # Local testing execution
    lambda_handler(event={}, context={})