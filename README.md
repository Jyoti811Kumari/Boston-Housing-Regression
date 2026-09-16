# Boston Housing Regression – Machine Learning Model Comparison

## Project Overview

This project compares multiple supervised machine learning regression models for predicting median house prices using the Boston Housing dataset.

The project covers exploratory data analysis, data preprocessing, model training, model evaluation, model comparison, and feature importance analysis.

## Objectives

* Perform exploratory data analysis
* Analyze relationships between features and house prices
* Preprocess and standardize the data where required
* Train and compare multiple regression models
* Evaluate model performance using RMSE, MAE, and R²
* Analyze Random Forest feature importance
* Compare models based on prediction performance

## Models Used

The project compares the following regression models:

1. Ordinary Least Squares (OLS)
2. Lasso Regression
3. Post-Lasso Regression
4. Ridge Regression
5. Elastic Net Regression
6. Decision Tree Regressor
7. Random Forest Regressor
8. Gradient Boosting Regressor
9. Neural Network Regressor

## Evaluation Metrics

Model performance is evaluated using:

* **RMSE (Root Mean Squared Error)** – measures the magnitude of prediction errors, with larger errors receiving greater weight.
* **MAE (Mean Absolute Error)** – measures the average absolute prediction error.
* **R² (R-squared)** – measures the proportion of variation in the target variable explained by the model.

Lower RMSE and MAE indicate lower prediction error, while a higher R² indicates better performance.

## Project Workflow

Data Loading
↓
Exploratory Data Analysis
↓
Train-Test Split
↓
Feature Scaling
↓
Model Training
↓
Model Evaluation
↓
Model Comparison
↓
Random Forest Feature Importance

## Results

The project generates:

- A model comparison table containing RMSE, MAE, and R² for all models
- MEDV distribution visualization
- Correlation matrix visualization
- Random Forest feature importance visualization
- Random Forest actual-vs-predicted visualization
- CSV and Excel files containing model performance results

## Visualizations

### 1. MEDV Distribution

![MEDV Distribution](medv_distribution.png)

### 2. Correlation Matrix

![Correlation Matrix](correlation_matrix.png)

### 3. Random Forest Feature Importance

![Feature Importance](feature_importance.png)

### 4. Random Forest Actual vs Predicted

![Actual vs Predicted](rf_actual_vs_predicted.png)

## Project Files

- `ML- BOSTON HOUSING PRICE.py` – Main Python script containing the complete analysis and machine learning workflow
- `boston_housing_model_results.csv` – Model comparison results
- `boston_housing_model_results.xlsx` – Model comparison results in Excel format
- `medv_distribution.png` – Distribution of median house value (MEDV)
- `correlation_matrix.png` – Correlation matrix of the dataset variables
- `feature_importance.png` – Random Forest feature importance visualization
- `rf_actual_vs_predicted.png` – Random Forest actual-vs-predicted visualization
- `requirements.txt` – Python libraries required to run the project
- `.gitignore` – Files excluded from Git version control

## Dataset

The Boston Housing dataset is retrieved programmatically from OpenML using scikit-learn's `fetch_openml()` function. The raw dataset is therefore not included in this repository.

## Dataset Note

The Boston Housing dataset is a historical benchmark dataset with known limitations. It is used in this project for educational purposes and for demonstrating regression model comparison.

## Key Takeaway

This project demonstrates a complete machine learning regression workflow, from exploratory analysis and preprocessing to model training, evaluation, comparison, and interpretation of Random Forest feature importance.
