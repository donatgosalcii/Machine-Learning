# -----------------------------------------------------------------------------
# Project: Machine Learning Sem 10 (Regression Example)
# Dataset: California Housing
# -----------------------------------------------------------------------------

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib
# Attempt to use a non-interactive backend that can save files.
# 'Agg' is a good default for saving files if no display is available.
try:
    matplotlib.use('Agg') # Ensure it can run without a display server for saving
except ImportError:
    print("Warning: Could not set Matplotlib backend to Agg. Plots might not save correctly if no display is available.")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set some display options for Pandas and Matplotlib/Seaborn
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6) # Default figure size

print("--------------------------------------------------")
print("Project: Machine Learning Sem 10 - Regression")
print("--------------------------------------------------\n")

# -----------------------------------------------------------------------------
# I. Data Wrangling (Preparing the Data)
# -----------------------------------------------------------------------------
print("I. Data Wrangling (Preparing the Data)")
print("======================================\n")

# --- Data Acquisition: ---
print("1. Data Acquisition:")
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

print("   Dataset loaded successfully.")
print(f"   Shape of the dataset: {df.shape}")
print("   First 5 rows of the dataset:\n", df.head())
print("\n")

# --- Data Inspection and Exploration (Initial): ---
print("2. Data Inspection and Exploration (Initial):")
print("   Basic information about the dataset (df.info()):")
df.info()
print("\n")

print("   Descriptive statistics (df.describe()):")
print(df.describe())
print("\n")

# --- Data Cleaning: ---
print("3. Data Cleaning:")
print("   Checking for missing values (df.isnull().sum()):")
missing_values = df.isnull().sum()
print(missing_values)

if missing_values.sum() == 0:
    print("   No missing values found.")
else:
    print("   Missing values found. Handling them (e.g., imputation or removal):")
    # Example handling (not needed for this dataset)
    # for col in df.columns:
    #     if df[col].isnull().any():
    #         if pd.api.types.is_numeric_dtype(df[col]):
    #             df[col].fillna(df[col].median(), inplace=True)
    #         else:
    #             df[col].fillna(df[col].mode()[0], inplace=True)
    # print("   Missing values handled.")
print("\n")

duplicate_rows = df.duplicated().sum()
print(f"   Number of duplicate rows: {duplicate_rows}")
if duplicate_rows > 0:
    df.drop_duplicates(inplace=True)
    print(f"   Dropped {duplicate_rows} duplicate rows. New shape: {df.shape}")
print("\n")


# --- Data Transformation: ---
print("4. Data Transformation (Feature Engineering & Scaling - initial setup):")
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

print(f"   Features (X) shape: {X.shape}")
print(f"   Target (y) shape: {y.shape}")
print("   Feature scaling will be applied after data splitting.\n")


# -----------------------------------------------------------------------------
# II. Data Analysis (Exploring and Understanding Patterns)
# -----------------------------------------------------------------------------
print("II. Data Analysis (Exploring and Understanding Patterns)")
print("======================================================\n")

# --- Descriptive Statistics: ---
print("1. Descriptive Statistics (Revisited):")
print(df.describe())
print("\n")

# --- Exploratory Data Analysis (EDA): ---
print("2. Exploratory Data Analysis (EDA):")

# Visualization:
print("   Visualization:")

# a) Distribution of the target variable (Median House Value)
plt.figure(figsize=(10, 6))
sns.histplot(df['MedHouseVal'], kde=True, bins=30)
plt.title('Distribution of Median House Value (Target Variable)')
plt.xlabel('Median House Value ($100,000s)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("plot_01_target_distribution.png")
plt.close() # Close the figure
print("   - Saved distribution of Median House Value to plot_01_target_distribution.png.")

# b) Histograms for all numerical features
print("   - Saving histograms for numerical features.")
# Let X.hist() create its own figure and axes array
X.hist(bins=30, figsize=(20,15), layout=(-1, 3)) # layout for auto-arrangement, e.g., 3 columns
# The figure is implicitly the current figure after X.hist() runs
plt.suptitle('Histograms of Numerical Features', fontsize=16) # Removed y, let tight_layout handle
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # rect to make space for suptitle, adjust as needed
plt.savefig("plot_02_feature_histograms.png")
plt.close() # Close the figure
print("   - Saved histograms for numerical features to plot_02_feature_histograms.png.")


# c) Correlation Analysis:
print("   Correlation Analysis:")
correlation_matrix = df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Features and Target')
plt.tight_layout()
plt.savefig("plot_03_correlation_heatmap.png")
plt.close() # Close the figure
print("   - Saved correlation heatmap to plot_03_correlation_heatmap.png.")

print("   Correlation with Median House Value (MedHouseVal):\n", correlation_matrix['MedHouseVal'].sort_values(ascending=False))
print("\n")

# d) Scatter plots of highly correlated features with the target
plt.figure(figsize=(10, 6))
sns.scatterplot(x='MedInc', y='MedHouseVal', data=df, alpha=0.5)
plt.title('Median Income vs. Median House Value')
plt.xlabel('Median Income')
plt.ylabel('Median House Value ($100,000s)')
plt.tight_layout()
plt.savefig("plot_04_medinc_vs_medhouseval.png")
plt.close() # Close the figure
print("   - Saved scatter plot: Median Income vs. Median House Value to plot_04_medinc_vs_medhouseval.png.")
print("\n")


# -----------------------------------------------------------------------------
# III. Regression Model Execution
# -----------------------------------------------------------------------------
print("III. Regression Model Execution")
print("===============================\n")

# --- Choosing the Right Regression Model: ---
print("1. Choosing the Right Regression Model:")
print("   - Selected Model: Linear Regression (as a starting point).\n")

# --- Splitting the Data: ---
print("2. Splitting the Data (Train/Test Split):")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"   X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"   X_test shape: {X_test.shape}, y_test shape: {y_test.shape}\n")

# --- Feature Scaling (Applied after splitting) ---
print("   Applying Feature Scaling (StandardScaler):")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
print("   First 5 rows of scaled training features:\n", X_train_scaled_df.head())
print("\n")


# --- Model Training: ---
print("3. Model Training:")
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("   Linear Regression model trained successfully.\n")

print("   Model Coefficients (Weights):")
coefficients = pd.Series(model.coef_, index=X.columns)
print(coefficients)
print(f"\n   Model Intercept: {model.intercept_:.4f}\n")


# --- Model Evaluation: ---
print("4. Model Evaluation:")
y_pred_test = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

print(f"   Mean Squared Error (MSE) on Test Set: {mse:.4f}")
print(f"   Root Mean Squared Error (RMSE) on Test Set: {rmse:.4f}")
print(f"   R-squared (R²) on Test Set: {r2:.4f}\n")

# Visualize Predictions vs. Actuals
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.7, edgecolors='k', s=50)
plt.plot([min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())], # Adjusted line for better fit
         [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())], 'r--', lw=2)
plt.xlabel('Actual Median House Value')
plt.ylabel('Predicted Median House Value')
plt.title('Actual vs. Predicted Median House Value (Test Set)')
plt.tight_layout()
plt.savefig("plot_05_actual_vs_predicted.png")
plt.close() # Close the figure
print("   - Saved Actual vs. Predicted values plot to plot_05_actual_vs_predicted.png.")

# Residual Plot
residuals = y_test - y_pred_test
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals (Actual - Predicted)')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.axvline(0, color='r', linestyle='--') # Add a line at zero for reference
plt.tight_layout()
plt.savefig("plot_06_residuals_distribution.png")
plt.close() # Close the figure
print("   - Saved distribution of residuals to plot_06_residuals_distribution.png.")
print("\n")

# -----------------------------------------------------------------------------
# Documentation and Reporting:
# -----------------------------------------------------------------------------
print("IV. Documentation and Reporting")
print("===============================\n")
print("   - The steps taken in this script, including comments and printed outputs, form the basis of documentation.")
print("   - Key findings include:")
print(f"     - The dataset has {df.shape[0]} samples and {df.shape[1]-1} features plus 1 target variable.")
print(f"     - 'MedInc' (Median Income) showed the highest positive correlation with 'MedHouseVal' (Target).")
print(f"     - A Linear Regression model was trained and evaluated.")
print(f"     - The model achieved an R-squared of approximately {r2:.2f} on the test set.")
print("   - Visualizations (histograms, correlation heatmap, scatter plots, actual vs. predicted plot, residuals plot) were generated and saved as PNG files.")
print("   - Key metrics (MSE, RMSE, R²) were calculated and reported.\n")

print("--------------------------------------------------")
print("End of Project Script")
print("--------------------------------------------------")
