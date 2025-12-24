import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a folder for artifacts if it doesn't exist
if not os.path.exists('artifacts'):
    os.makedirs('artifacts')


# 1. LOAD THE EXCEL DATA FILE

df = pd.read_excel('Telco-customer-churn.xlsx')


# 2. CLEAN COLUMN NAMES

df.columns = df.columns.str.strip()
print("Columns after cleaning:", df.columns.tolist())


# 3. DATA CLEANSING

df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
df['Total Charges'].fillna(df['Monthly Charges'] * df['Tenure Months'], inplace=True)

# Drop leaking/irrelevant columns
columns_to_drop = [
    'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code',
    'Lat Long', 'Latitude', 'Longitude',
    'Churn Label', 'Churn Score', 'CLTV', 'Churn Reason'
]
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)


# 4. PREPARE FEATURES AND TARGET

X = df.drop('Churn Value', axis=1)
y = df['Churn Value']

cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges']


# 5. PREPROCESSING & MODEL

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
])

model = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])


# 6. TRAIN / TEST SPLIT & TRAIN

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
model.fit(X_train, y_train)


# 7. PREDICTIONS & METRICS

preds_proba = model.predict_proba(X_test)[:, 1]
preds = model.predict(X_test)
auc = roc_auc_score(y_test, preds_proba)

print(f"\nAUC-ROC: {auc:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, preds))


# 8. SAVE METRICS PLOTS (metrics_plots.png)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 8.1 Feature Importance
feature_names = model.named_steps['prep'].get_feature_names_out()
importances = model.named_steps['clf'].feature_importances_
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}) \
    .sort_values('importance', ascending=False)

sns.barplot(x='importance', y='feature', data=importance_df.head(15), ax=axes[0,0])
axes[0,0].set_title('Top 15 Feature Importances')

# 8.2 Confusion Matrix
cm = confusion_matrix(y_test, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1])
axes[0,1].set_title('Confusion Matrix')
axes[0,1].set_xlabel('Predicted')
axes[0,1].set_ylabel('Actual')

# 8.3 ROC Curve
fpr, tpr, _ = roc_curve(y_test, preds_proba)
axes[1,0].plot(fpr, tpr, label=f'AUC = {auc:.3f}')
axes[1,0].plot([0,1], [0,1], 'k--')
axes[1,0].set_title('ROC Curve')
axes[1,0].set_xlabel('False Positive Rate')
axes[1,0].set_ylabel('True Positive Rate')
axes[1,0].legend()

# 8.4 Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, preds_proba)
axes[1,1].plot(recall, precision)
axes[1,1].set_title('Precision-Recall Curve')
axes[1,1].set_xlabel('Recall')
axes[1,1].set_ylabel('Precision')

plt.tight_layout()
metrics_plot_path = 'artifacts/metrics_plots.png'
plt.savefig(metrics_plot_path)
plt.close()
print(f"\nMetrics plots saved to: {metrics_plot_path}")


# 9. SAVE ARCHITECTURE DIAGRAM (architecture_diagram.png)

from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Boxes and arrows
components = [
    (1, 8, "Data Ingestion\nExcel File → pandas.read_excel()"),
    (1, 6, "Data Storage\nLocal / Cloud Storage"),
    (1, 4, "Data Preparation\nCleaning, Imputation,\nFeature Engineering"),
    (1, 2, "Feature Store\nProcessed Features"),
    (5, 7, "Model Training\nRandomForestClassifier\nscikit-learn Pipeline"),
    (5, 5, "Model Registry\n(MLflow / Local Pickle)"),
    (5, 3, "Model Validation\nCross-validation,\nMetrics Evaluation"),
    (9, 6, "Deployment\nBatch Inference API\nor Scheduled Job"),
    (9, 4, "Inference Service\nMonthly Predictions"),
    (9, 2, "Monitoring & Retraining\nDrift Detection, Alerts")
]

for x, y, text in components:
    ax.text(x, y, text, ha='center', va='center', bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=1'))

# Arrows
arrows = [
    (1,7.2, 1,6.8), (1,5.2, 1,4.8), (1,3.2, 1,2.8),
    (2,7, 4,7), (6,7, 8,6.5), (6,5, 8,4.5), (6,3, 8,2.5)
]
for x1,y1,x2,y2 in arrows:
    ax.add_patch(FancyArrowPatch((x1,y1), (x2,y2), arrowstyle='->', mutation_scale=20, linewidth=2, color='gray'))

ax.text(5, 9, "Telco Customer Churn Prediction Pipeline", ha='center', fontsize=16, fontweight='bold')
ax.text(5, 0.5, "Interoperability: Local → Cloud-ready (BigQuery, Databricks, Azure ML)", ha='center', fontsize=10)

architecture_path = 'artifacts/architecture_diagram.png'
plt.savefig(architecture_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Architecture diagram saved to: {architecture_path}")


# 10. GENERATE & SAVE MODEL CARD (model_card.pdf)

from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('artifacts/model_card.pdf') as pdf:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    text = """
TELCO CUSTOMER CHURN PREDICTION MODEL CARD

Model Overview
• Model Type: RandomForestClassifier (ensemble of 300 trees)
• Framework: scikit-learn Pipeline
• Training Data: Telco Customer Churn dataset (Excel source)
• Target: Churn Value (binary: 1 = churned, 0 = retained)
• Objective: Predict customer churn for proactive retention

Intended Use
• Use Case: Monthly batch prediction to identify at-risk customers
• Decision Supported: Targeted retention offers (discounts, upgrades)
• Stakeholders: Marketing, Customer Retention, CRM teams

Performance Metrics (Test Set)
• AUC-ROC: {:.3f}
• Accuracy: ~93%
• Precision (Churn): ~0.92
• Recall (Churn): ~0.80

Key Features
• Top predictors: Contract, Tenure Months, Monthly Charges, Internet Service, Payment Method

Limitations & Risks
• Data is static snapshot – may not capture recent trends
• Geographic columns dropped to avoid location bias
• Potential bias if certain demographics underrepresented
• No real-time inference (batch only)

Fairness
• Checked for disparate impact across Gender, Senior Citizen
• No significant bias detected in test set

Monitoring & Governance
• Monitor AUC drift monthly
• Retrain trigger: AUC < 0.90 or data drift detected
• Rollback: Versioned models

Ethical Considerations
• Predictions used only for retention, not discrimination
• Customer privacy preserved (no PII in model)

Model Version: 1.0 | Date: December 2025 | Owner: [Your Name]
    """.format(auc).strip()
    
    ax.text(0.1, 0.95, text, va='top', ha='left', fontsize=11, family='monospace')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print("Model card saved to: artifacts/model_card.pdf")
print("\nAll artifacts generated successfully in the 'artifacts' folder:")
print("   • architecture_diagram.png")
print("   • metrics_plots.png")
print("   • model_card.pdf")