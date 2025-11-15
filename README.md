# Data Science SBMP - Customer Churn Prediction

## Overview
This project implements a customer churn prediction model using machine learning techniques. The notebook demonstrates a complete workflow from data loading through model training and evaluation.

## Project Objectives
- Load and explore customer dataset
- Preprocess and clean the data
- Build and train a Logistic Regression model
- Evaluate model performance using classification metrics

## Dataset
The project uses a customer dataset stored in `archive.zip`. The dataset contains various customer attributes including:
- **Customer ID**: Unique identifier for each customer
- **Churn**: Target variable (Yes/No - whether the customer churned)
- **Tenure**: Duration of customer relationship
- **Monthly Charges**: Monthly subscription charges
- **Total Charges**: Total charges accumulated
- **Other categorical features**: Customer demographics and service attributes

## Technologies & Libraries
- **Python 3.x**
- **pandas**: Data manipulation and preprocessing
- **scikit-learn**: Machine learning model and metrics
  - LogisticRegression: Classification model
  - train_test_split: Data splitting
  - StandardScaler: Feature scaling
  - classification_report, accuracy_score: Model evaluation

## Workflow

### 1. Data Loading
The dataset is loaded from the CSV file within `archive.zip` using pandas.

### 2. Data Cleaning & Preprocessing
- Remove non-predictive columns (e.g., customerID)
- Convert `TotalCharges` to numeric format, handling errors
- Map categorical target variable `Churn` to binary (0/1)
- Handle missing values
- Encode categorical features using one-hot encoding
- Scale numeric features (tenure, MonthlyCharges, TotalCharges) using StandardScaler

### 3. Model Training
- Split data into 80% training and 20% testing sets (stratified split)
- Train Logistic Regression model with 1000 maximum iterations
- Use random_state=42 for reproducibility

### 4. Model Evaluation
- Generate predictions on test set
- Calculate accuracy score
- Produce detailed classification report with precision, recall, and F1-score

## Results
The notebook outputs:
- **Accuracy Score**: Overall model performance metric
- **Classification Report**: Detailed metrics for both classes (Churned/Not Churned)

## How to Run
1. Ensure all required libraries are installed:
   ```bash
   pip install pandas scikit-learn
   ```

2. Place `archive.zip` in the same directory as the notebook

3. Execute the notebook cells in order or run the entire notebook:
   ```bash
   jupyter notebook datascienceSBMP.ipynb
   ```

## File Structure
```
data_science/
├── datascienceSBMP.ipynb    # Main Jupyter notebook
├── archive.zip              # Dataset (required)
└── README.md                # This file
```

## Model Details
- **Algorithm**: Logistic Regression
- **Test Size**: 20% of data
- **Random State**: 42 (for reproducibility)
- **Stratification**: Yes (maintains class distribution in splits)
- **Feature Scaling**: StandardScaler applied to numeric features

## Future Improvements
- Experiment with other algorithms (Random Forest, SVM, etc.)
- Perform hyperparameter tuning
- Add cross-validation for robustness
- Implement feature importance analysis
- Create visualizations for exploratory data analysis

## Author
Project workspace: data_science

## Notes
- Ensure the dataset file is properly formatted before running
- Missing values in 'TotalCharges' are dropped during preprocessing
- Random state ensures reproducible results across runs
