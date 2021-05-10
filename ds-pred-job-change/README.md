# Predict which Data Scientists are likely to change jobs

## Overview
- Dataset from [HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists)
- Engineered features by filling in NaN values, encoding categorical data and oversampling of minority class.
- Selectied most important features using chi2 test.
- Used Logistic Regression, RandomForestClassification and XGBoost for machine learning.

## Code and resources used
**Python Version:** 3.9
**Packages:** python, scikit-learn, xgboost, pandas, missingno, imbalanced-learn, jupyter

## Feature Engineering
1. Filled in missing values using KNNImputer (k-Nearest Neighbors).
2. Renamed certain values (removed special characters from the value).
3. One hot encode categorical features.
4. Dataset given has a minority class, used SMOTE to oversample the undersampled class.

## Feature Selection
Used SelectBest() from sklearn to select the top features based on chi2 test.

## Machine Learning
Dataset was split into train and test set with a test size of 0.2.

Three different models was tests and evaluated using sklearn accuracy score.

The three models are:
- **Logistic Regression:** Used as a baseline model
- **Random Forest Classification:** Chosen as Random forest tend to work well with a mixture of numerical and categorical features and works well with large datasets.
- **XGBoost:** Chosen as it is one of the most pouplar ML algorithm, and its is able to combien the advantages of random forest and gradient boosting.
