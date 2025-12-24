Project Overview

This project implements an end-to-end machine learning pipeline to predict customer churn for a telecommunications company. The solution addresses a real-world business problem by transforming raw customer data into actionable churn predictions that support proactive retention strategies.

The pipeline covers data ingestion, cleansing, feature engineering, model training, validation, deployment readiness, and governance artifacts, demonstrating a production-minded ML workflow.

Business Problem

Customer churn significantly impacts revenue in subscription-based businesses. Retaining existing customers is more cost-effective than acquiring new ones. This project predicts the likelihood of customer churn so marketing and retention teams can target at-risk customers with personalized offers.

Dataset

Source: Telco Customer Churn dataset (Excel format)

File: Telco-customer-churn.xlsx

Target Variable: Churn Value

1 = Customer churned

0 = Customer retained

Pipeline Architecture

The pipeline follows a modular, cloud-ready architecture:

Data Ingestion – Load Excel data using pandas.read_excel()

Data Storage – Local or cloud-compatible storage

Data Preparation – Cleaning, imputation, and feature engineering

Feature Store – Processed features for reuse

Model Training – RandomForestClassifier in a scikit-learn Pipeline

Model Validation – Cross-validation and performance metrics

Model Registry – Versioned model artifacts (local / MLflow-ready)

Deployment Readiness – Batch inference or API-based serving

Monitoring & Retraining – Drift detection and alerts

An architecture diagram is automatically generated and saved as:

artifacts/architecture_diagram.png

Data Cleansing & Feature Engineering

Key preprocessing steps include:

Standardizing column names

Converting Total Charges to numeric values

Imputing missing Total Charges using tenure × monthly charges

Removing data leakage and irrelevant columns (e.g., customer identifiers, churn labels, geographic data)

Scaling numerical features

One-hot encoding categorical variables

These steps ensure data quality, prevent leakage, and improve model generalization.

Model Details

Algorithm: RandomForestClassifier

Trees: 300

Class Handling: Balanced class weights

Framework: scikit-learn Pipeline

Preprocessing: ColumnTransformer with scaling and encoding

Model Evaluation

The model is evaluated on a stratified test set using:

AUC-ROC

Precision, Recall, F1-score

Confusion Matrix

ROC Curve

Precision–Recall Curve

Feature Importance analysis

All evaluation visualizations are saved to:

artifacts/metrics_plots.png

Performance Summary (Approximate)

AUC-ROC: ~0.93

Accuracy: ~93%

Precision (Churn): ~0.92

Recall (Churn): ~0.80

These results indicate strong discriminative power and business usability for churn prevention.

Model Card & Governance

A formal Model Card is generated to document:

Intended use

Performance metrics

Key features

Limitations and risks

Fairness considerations

Monitoring and retraining strategy

Ethical considerations

Saved as:

artifacts/model_card.pdf

Artifacts Generated

After running the script, the following artifacts are created automatically:

artifacts/
│── architecture_diagram.png
│── metrics_plots.png
│── model_card.pdf

How to Run

Ensure Python 3.8+ is installed

Install required dependencies:

pip install pandas scikit-learn matplotlib seaborn openpyxl


Place Telco-customer-churn.xlsx in the project root

Run the script:

python churn_pipeline.py

Deployment Readiness

Designed for batch inference (monthly churn prediction)

Easily extendable to:

REST API inference

Cloud platforms (Azure ML, Databricks, BigQuery)

Supports monitoring and retraining triggers based on performance drift

Ethical & Fairness Considerations

No personally identifiable information (PII) is used

Geographic and sensitive attributes are excluded

Predictions are intended solely for customer retention, not discrimination
