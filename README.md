# Credit Card Fraud Detection Model

## Overview
Credit card fraud is a significant challenge for both consumers and financial institutions. With millions of transactions occurring daily, identifying fraudulent activity accurately and efficiently is critical for protecting users and minimizing financial risk.

This project explores the application of machine learning techniques to detect fraudulent credit card transactions using historical transaction and identity data. By leveraging supervised and unsupervised learning approaches, the goal was to build a robust model capable of distinguishing fraudulent transactions from legitimate ones while navigating challenges such as high dimensionality, class imbalance, and computational constraints.

## Research Question

Can we build a machine learning model that accurately predicts whether a credit card transaction is fraudulent or not?

## Dataset

This project uses the IEEE-CIS Fraud Detection dataset from Kaggle, provided by Vesta Corporation. The dataset reflects real-world e-commerce transactions and is widely used for fraud detection research.

### Dataset Characteristics:

- Two tables:

  - **Transaction table** (transaction amounts, product types, card identifiers, email domains, transaction flags)

  - **Identity table** (device type, device information, and behavioral indicators)

Joined using a unique **Transaction ID**

Final dataset size:

- 590,540 rows

- 434 features

Binary target variable: **Is Fraud**

## Machine Learning Models

### Supervised Learning

- Random Forest Classifier

- XGBoost Classifier

Both models were trained using train, validation, and test splits across different months to preserve temporal integrity. Hyperparameters were optimized using GridSearchCV, with a limited search space due to computational constraints.

### Unsupervised Learning

- Isolation Forest

- Used to evaluate how anomaly detection compares to supervised approaches in identifying fraud

### Model Evaluation

Models were evaluated using:

- Accuracy

- ROC AUC

- Precision, Recall, and F1-score (with emphasis on the minority fraud class)

Multiple datasets were compared to assess performance trade-offs between feature reduction and predictive power.

## Final Results & Key Findings

### Best Performing Model

The XGBoost classifier trained on the Original Dataset was identified as the best-performing model.

**Optimized Hyperparameters:**

- n_estimators: 100

- max_depth: 10

- learning_rate: 0.1

**Performance Highlights:**

- Accuracy: High (≈ 0.98)

- ROC AUC: Strong discrimination capability (≈ 0.92)

- Precision (Fraud Class): 0.94

- Recall (Fraud Class): 0.52

- F1-score (Fraud Class): 0.67

While the model demonstrated excellent precision—meaning most flagged fraud cases were correct—the recall was lower, indicating that a portion of fraudulent transactions were still missed. This reflects a common challenge in highly imbalanced datasets, where improving recall without sacrificing precision remains difficult.

Despite limited computational resources restricting broader hyperparameter tuning, the results reinforce the effectiveness of tree-based machine learning models for fraud detection tasks.

## Key Challenges

- Severe class imbalance between fraudulent and non-fraudulent transactions

- High dimensionality and memory constraints

- Limited computational resources restricting extensive hyperparameter searches

- These challenges were addressed through careful feature reduction, dataset comparisons, and pragmatic modeling decisions.

## Future Improvements

- Potential next steps to improve performance include:

- Feature engineering to better capture individual cardholder behavior

- Creating user-level aggregates (e.g., average transaction amount, spending patterns)

- Applying resampling techniques such as SMOTE or class weighting

- Threshold tuning based on ROC curves

- Expanding hyperparameter searches with increased computational resources
