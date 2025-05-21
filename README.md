
## Workflow Implemented

The project covers the following key stages as outlined in the `ml_project_sem10.py` script:

1.  **Data Wrangling (Preparing the Data)**
    *   **Data Acquisition:** Loads the California Housing dataset from `sklearn.datasets`.
    *   **Data Inspection & Exploration:** Initial checks using `.head()`, `.info()`, `.describe()`.
    *   **Data Cleaning:** Checks for missing values and duplicates.
    *   **Data Transformation:** Separates features (X) and target (y); prepares for feature scaling.

2.  **Data Analysis (Exploring and Understanding Patterns)**
    *   **Descriptive Statistics:** Revisits summary statistics.
    *   **Exploratory Data Analysis (EDA):**
        *   Visualization of target variable distribution.
        *   Histograms for all numerical features.
        *   Correlation analysis (matrix and heatmap).
        *   Scatter plot of the most correlated feature (`MedInc`) against the target.

3.  **Regression Model Execution**
    *   **Model Selection:** Linear Regression chosen as a baseline.
    *   **Data Splitting:** Data split into 80% training and 20% testing sets.
    *   **Feature Scaling:** `StandardScaler` applied to training and test features.
    *   **Model Training:** Linear Regression model trained on the scaled training data.
    *   **Model Evaluation:** Model performance assessed on the test set using:
        *   Mean Squared Error (MSE)
        *   Root Mean Squared Error (RMSE)
        *   R-squared (R²)
        *   Visualizations: Actual vs. Predicted plot, Residuals distribution plot.

4.  **Documentation and Reporting**
    *   The Python script includes print statements for step-by-step output.
    *   Key findings and metrics are reported.
    *   Visualizations are saved as PNG files (listed above).

## Key Findings (from the script output)

*   The dataset contains 20,640 samples and 8 numerical features.
*   Median Income (`MedInc`) has the strongest positive correlation (approx. +0.69) with Median House Value (`MedHouseVal`).
*   The Linear Regression model achieved an R-squared value of approximately **0.58** on the test set.
*   The Root Mean Squared Error (RMSE) on the test set was approximately **0.7456** (in units of $100,000s).

## How to Run

1.  **Prerequisites:**
    *   Python 3.x
    *   A Python virtual environment is highly recommended.

2.  **Setup Virtual Environment (Recommended):**
    ```bash
    # Navigate to the project directory
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Ensure you have the necessary Python libraries. If a `requirements.txt` file is provided:
    ```bash
    pip install -r requirements.txt
    ```
    Otherwise, install them manually (while the virtual environment is active):
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

4.  **Execute the Script:**
    ```bash
    python ml_project_sem10.py
    ```

5.  **Outputs:**
    *   The script will print detailed output to the console for each step.
    *   Plot images (listed in Project Structure) will be saved in the project directory.

## Tools and Libraries Used

*   **Python 3**
*   **Pandas:** For data manipulation and analysis.
*   **NumPy:** For numerical operations.
*   **Matplotlib:** For plotting.
*   **Seaborn:** For enhanced statistical visualizations.
*   **Scikit-learn:** For the dataset, preprocessing tools, machine learning model (Linear Regression), and evaluation metrics.

## Future Considerations / Potential Improvements

*   Explore more complex regression models (e.g., Random Forest, Gradient Boosting) to potentially improve the R² score.
*   Investigate the impact of outliers (e.g., the $500,000 cap on house values) and consider strategies to handle them.
*   Implement more advanced feature engineering techniques or explore regional segmentation.
*   Perform hyperparameter tuning for more complex models.

---
*(This README provides a summary. For detailed documentation, please refer to the project report and the commented Python script.)*
