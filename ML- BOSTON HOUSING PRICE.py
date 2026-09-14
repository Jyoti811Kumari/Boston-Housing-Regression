# Numerical computation and data handling
import numpy as np
import pandas as pd
import os

# Scikit-learn ML tools
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Fetch Boston dataset (modern replacement)
boston = fetch_openml(name="boston", version=1, as_frame=True)
data = boston.frame

# Check shape and columns
print("Dataset shape:", data.shape)
print("Columns:", data.columns)

# First 5 rows
print("\nFirst 5 rows:")
print(data.head())

# Dataset information
print("\nDataset information:")
print(data.info())

# Statistical summary
print("\nStatistical summary:")
print(data.describe())

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Check for duplicate rows
print("\nNumber of duplicate rows:")
print(data.duplicated().sum())

# Distribution of target variable
plt.figure(figsize=(8, 5))
sns.histplot(data["MEDV"], kde=True)
plt.title("Distribution of Median House Value (MEDV)")
plt.xlabel("MEDV")
plt.ylabel("Frequency")
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(numeric_only=True), annot=True, fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Correlation of features with target
target_corr = data.corr(numeric_only=True)["MEDV"].sort_values(ascending=False)

print("\nCorrelation of features with MEDV:")
print(target_corr)

# Features (X) and target (y)
X = data.drop(columns=["MEDV"])  # Drop target column
y = data["MEDV"]                 # Target variable

# Split dataset: 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Standardize features for models that are scale-sensitive
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Step 2 complete: Features and target defined, data split and scaled")
print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape, "y_test shape:", y_test.shape)

# Ordinary Least Square(OLS)
ols = LinearRegression()
ols.fit(X_train_scaled, y_train)
ols_pred = ols.predict(X_test_scaled)
print("OLS trained ")

# Lasso Regression
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train_scaled, y_train)
lasso_pred = lasso.predict(X_test_scaled)
print("Lasso trained ")

# Post-Lasso
selected = np.abs(lasso.coef_) > 1e-6  # Non-zero features
ols_postlasso = LinearRegression()
ols_postlasso.fit(X_train_scaled[:, selected], y_train)
postlasso_pred = ols_postlasso.predict(X_test_scaled[:, selected])
print("Post-Lasso trained ")

# Ridge Regression
ridge = RidgeCV(cv=5)
ridge.fit(X_train_scaled, y_train)
ridge_pred = ridge.predict(X_test_scaled)
print("Ridge trained ")

# Elastic Net
elastic = ElasticNetCV(cv=5, random_state=42)
elastic.fit(X_train_scaled, y_train)
elastic_pred = elastic.predict(X_test_scaled)
print("Elastic Net trained ")

# Train Nonlinear Regression Models

# Decision Tree
tree = DecisionTreeRegressor(random_state=42)
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)
print("Decision Tree trained ")

# Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Random Forest trained ")

# Feature Importance - Random Forest

feature_importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("Feature Importance:")
print(feature_importance)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
feature_importance.plot(kind='bar')

plt.title("Feature Importance - Random Forest")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Model Interpretation

print("Model Interpretation:")
print("The feature importance values show the relative contribution of each")
print("feature to the Random Forest model's predictions.")
print("Features with higher importance have a greater influence on the predictions.")
print("Feature importance indicates predictive contribution, not causality.")

# Gradient Boosted Trees
gb = GradientBoostingRegressor(n_estimators=200, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
print("Boosted Trees trained ")

# Neural Network
nn = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=2000, random_state=42)
nn.fit(X_train_scaled, y_train)
nn_pred = nn.predict(X_test_scaled)
print("Neural Network trained ")

# Evaluate All Models

def evaluate_model(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return [name, rmse, mae, r2]

results = []
results.append(evaluate_model("OLS", y_test, ols_pred))
results.append(evaluate_model("Lasso", y_test, lasso_pred))
results.append(evaluate_model("Post-Lasso", y_test, postlasso_pred))
results.append(evaluate_model("Ridge", y_test, ridge_pred))
results.append(evaluate_model("Elastic Net", y_test, elastic_pred))
results.append(evaluate_model("Decision Tree", y_test, tree_pred))
results.append(evaluate_model("Random Forest", y_test, rf_pred))
results.append(evaluate_model("Boosted Trees", y_test, gb_pred))
results.append(evaluate_model("Neural Network", y_test, nn_pred))

df_results = pd.DataFrame(results, columns=["Model","RMSE","MAE","R2"])
df_results = df_results.sort_values(by="RMSE", ascending=True)

print(df_results)


# Plot Example (Random Forest Predicted vs Actual)

plt.figure(figsize=(6,6))
plt.scatter(y_test, rf_pred, alpha=0.6)
plt.xlabel("Actual MEDV")
plt.ylabel("Predicted MEDV")
plt.title("Random Forest: Actual vs Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()
print("Plot generated")
plt.savefig("rf_actual_vs_predicted.png")
plt.close()

# Export Results to CSV/Excel 

# Get folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define file paths in the same folder
csv_path = os.path.join(script_dir, "boston_housing_model_results.csv")
excel_path = os.path.join(script_dir, "boston_housing_model_results.xlsx")

# Save evaluation results
df_results.to_csv(csv_path, index=False)
df_results.to_excel(excel_path, index=False)

print("CSV path:", csv_path)
print("Excel path:", excel_path)







