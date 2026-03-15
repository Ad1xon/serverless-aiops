# Serverless AIOps: Predictive Alerting for Cloud Metrics

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-orange)
![Status](https://img.shields.io/badge/Status-MVP%20Completed-success)

## Executive Summary
This project is an end-to-end prototype of a **predictive alerting system** designed to anticipate cloud service incidents based on historical metric data. By framing the issue as a supervised time-series classification problem with a shifted target, the system successfully detects incoming server anomalies before they result in downtime.

The architecture simulates a real-world **AWS Serverless workflow** consisting of periodic model retraining and high-frequency streaming inference.

## Key Objectives & Results
Cloud metrics are notoriously noisy and non-stationary. Evaluation in this domain requires moving beyond standard accuracy and focusing on actionable lead times.

* **Target:** Anticipate incidents with a 5-minute detection lead time.
* **Recall:** Achieved **~85% recall** on a held-out evaluation period (chronologically split). The model successfully raises an alert before the start of an incident for the vast majority of cases.
* **False Positives:** Maintained at a strictly controlled, reasonable level (< 5% FPR) using balanced class weights during training to prevent alert fatigue.

## Architecture & Cloud Workflow Simulation
The repository is structured to reflect a cloud-native AWS environment:

1.  **`src/lambda_retrain.py`**: Simulates a scheduled AWS Lambda function (e.g., triggered daily by EventBridge). It fetches historical CloudWatch metrics, processes rolling window features, retrains the `RandomForest` model to adapt to concept drift, and saves the artifact to simulated S3 storage (`models/`).
2.  **`src/lambda_inference.py`**: Simulates a high-frequency AWS Lambda function (e.g., triggered every minute). It loads the model artifact, ingests streaming telemetry, and outputs a binary alert decision.

## Quick Start (Local Run)

**1. Clone the repository and install dependencies:**
```bash
git clone [https://github.com/](https://github.com/)Ad1xon/serverless-aiops.git
cd serverless-aiops
pip install -r requirements.txt

```

**2. Trigger the Retraining Pipeline (Generates data & trains model):**

```bash
python src/lambda_retrain.py

```

*Expected output: Model evaluation metrics (Recall, FPR) and confirmation of artifact creation in `models/latest_model.pkl`.*

**3. Trigger the Inference Pipeline (Simulates streaming data prediction):**

```bash
python src/lambda_inference.py

```

*Expected output: A logged `[ALERT]` or `[OK]` based on the simulated incoming metric payload.*

## Modeling Insights & Future Improvements

* **Data Leakage Prevention:** Standard cross-validation fails on time-series data. The dataset was split strictly chronologically (80/20) to evaluate the model's forward-looking predictive power.
* **Feature Engineering:** Raw CPU values are insufficient due to noise. Moving averages and local standard deviations (`15m` sliding windows) were critical for the model to capture the structural "shape" of an impending incident.
* **Next Steps:** Implement dynamic thresholding for the decision boundary (Precision-Recall tradeoff tuning) and integrate automated Concept Drift detection to trigger retraining only when necessary.
