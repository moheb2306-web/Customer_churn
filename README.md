# Customer Churn Prediction

Predict whether a bank customer will leave (`Exited = 1`) using exploratory data analysis and three classification models: Decision Tree, Random Forest and XGBoost.

## Project structure

```
.
├── churn_prediction.ipynb   # EDA, preprocessing, training, evaluation
├── Churn_Modelling.csv      # Dataset (add this yourself)
├── requirements.txt         # Python dependencies
├── README.md
└── best_churn_model.pkl     # Created when the notebook is run
```

## Dataset

`Churn_Modelling.csv` has 10,000 customers and 14 columns.

| Column | Description |
|---|---|
| RowNumber, CustomerId, Surname | Identifiers (dropped before modelling) |
| CreditScore | Customer credit score |
| Geography | Country: France, Germany or Spain |
| Gender | Female or Male |
| Age | Customer age |
| Tenure | Years as a customer |
| Balance | Account balance |
| NumOfProducts | Number of bank products held |
| HasCrCard | 1 if the customer has a credit card |
| IsActiveMember | 1 if the customer is an active member |
| EstimatedSalary | Estimated salary |
| **Exited** | **Target: 1 = churned, 0 = stayed** |

About 20% of customers churned, so the classes are imbalanced.

## Setup

1. (Optional) create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Put `Churn_Modelling.csv` in the same folder as the notebook.

4. Start Jupyter and open the notebook:

   ```bash
   jupyter notebook churn_prediction.ipynb
   ```

5. Run all cells (Kernel > Restart & Run All).

## What the notebook does

1. **Imports**: pandas, numpy, matplotlib, seaborn, LabelEncoder, SMOTE, cross_val_score, the three models and the evaluation metrics.
2. **EDA**: data types, missing values, duplicates, class balance, numeric distributions by churn, box plots, churn rate by category, correlation heatmap.
3. **Preprocessing**:
   - Drop identifier columns.
   - Label encode `Geography` and `Gender`.
   - Stratified split into train, validation and test sets.
   - SMOTE on the training portion only.
4. **Cross-validation**: 5-fold stratified `cross_val_score` (ROC-AUC and F1). SMOTE runs inside each fold through an `imblearn` pipeline to avoid data leakage.
5. **Training**: Decision Tree, Random Forest and XGBoost. XGBoost uses early stopping on the validation set (stops after 50 rounds without AUC improvement).
6. **Evaluation** on the held-out test set: accuracy, F1, ROC-AUC, classification report, confusion matrices and ROC curves.
7. **Feature importance** for Random Forest and XGBoost.
8. **Saving**: the best model (by test ROC-AUC) is saved with its label encoders to `best_churn_model.pkl`.

## Using the saved model

```python
import joblib
import pandas as pd

artifact = joblib.load("best_churn_model.pkl")
model = artifact["model"]
encoders = artifact["label_encoders"]
features = artifact["features"]

# new_customers: a DataFrame with the same columns as the training features
new_customers["Geography"] = encoders["Geography"].transform(new_customers["Geography"])
new_customers["Gender"] = encoders["Gender"].transform(new_customers["Gender"])

churn_probability = model.predict_proba(new_customers[features])[:, 1]
```

## Notes

- Accuracy alone is misleading with imbalanced data, so focus on F1 and ROC-AUC for the churn class.
- The default 0.5 decision threshold may not suit your business case. Lower it to catch more churners at the cost of more false alarms.
- Tree-based models do not need feature scaling, so none is applied.
- The best model is chosen using the test set for simplicity. For a stricter setup, choose with cross-validation and use the test set once at the end.

## Possible improvements

- Hyperparameter search with `RandomizedSearchCV`
- One-hot encoding for `Geography`
- Engineered features such as balance-to-salary ratio or age groups
- SHAP for model explanations
