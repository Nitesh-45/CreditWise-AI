import pypandoc, os, textwrap
readme = r'''<div align="center">

# 💳 CreditWise AI
### Machine Learning-Based Loan Approval Prediction System

**A supervised machine learning project that analyzes applicant and financial attributes to predict loan approval outcomes.**

</div>

---

## 📌 Project Overview

**CreditWise AI** explores how machine learning classification models can be used to predict whether a loan application will be approved. The project uses applicant information such as income, age, credit score, debt-to-income ratio, savings, loan details, and categorical profile attributes.

The notebook follows a typical ML workflow: data inspection, missing-value handling, exploratory data analysis (EDA), categorical encoding, correlation analysis, train-test splitting, feature scaling, model training, evaluation, and feature engineering.

> **Important:** This is an educational prediction project, not a production lending system. Its predictions should not be used to make real credit decisions without substantial validation, fairness analysis, explainability, security, and regulatory review.

## 🎯 Objectives

- Inspect and prepare loan applicant data for machine learning.
- Explore distributions, class balance, potential outliers, and relationships between features.
- Convert categorical variables into numerical features.
- Train and compare classification models.
- Evaluate predictions using precision, recall, F1-score, accuracy, and confusion matrices.
- Test whether squared Credit Score and DTI Ratio features improve model performance.

## 🧰 Tech Stack

| Tool / Library | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations and feature engineering |
| Matplotlib | Data visualization |
| Seaborn | Statistical plots and correlation heatmap |
| Scikit-learn | Imputation, encoding, scaling, data splitting, models, and metrics |
| Jupyter Notebook | Interactive analysis and experimentation |

## 📂 Project Structure

```text
CreditWise-AI/
├── credit_wise.ipynb
├── loan_approval_data.csv
└── README.md
```

`loan_approval_data.csv` is expected by the notebook. The notebook file alone is not sufficient to reproduce the analysis unless the dataset is also available.

## 📊 Dataset Overview

The notebook loads `loan_approval_data.csv`. Its initial inspection reports **1,000 rows and 20 columns**, with missing values present in multiple fields. The target column is `Loan_Approved`.

Features used in the notebook include:

- **Applicant profile:** Age, Gender, Marital Status, Education Level, Dependents
- **Financial attributes:** Applicant Income, Coapplicant Income, Credit Score, DTI Ratio, Savings
- **Loan attributes:** Existing Loans, Loan Amount, Loan Term, Loan Purpose
- **Other attributes:** Employment Status, Employer Category, Property Area, Collateral Value
- **Target:** `Loan_Approved`

`Applicant_ID` is removed before model training because it is an identifier rather than a meaningful predictive attribute.

The exact dataset source, license, and data dictionary are not documented in the notebook. Add them here if known before publishing the repository.

## 🔍 Workflow

### 1. Data inspection and preprocessing

- Loads the CSV into a Pandas DataFrame.
- Inspects column types, summary statistics, and missing-value counts.
- Fills numerical missing values using the **mean**.
- Fills categorical missing values using the **most frequent value**.
- Removes `Applicant_ID` from the feature set.

### 2. Exploratory Data Analysis

The notebook explores:

- Loan approval class distribution.
- Education-level counts.
- Applicant and co-applicant income distributions.
- Income, credit score, DTI ratio, and savings across approval classes using box plots.
- Credit score and applicant income distributions grouped by the target.
- A correlation heatmap and feature correlations with `Loan_Approved`.

The saved notebook output shows a dataset with 1,000 records after imputation. The initial inspection reported 950 non-null values in several columns, meaning missingness was present before preprocessing.

### 3. Categorical encoding

- Uses `LabelEncoder` for `Education_Level` and `Loan_Approved`.
- Uses `OneHotEncoder(drop="first", handle_unknown="ignore")` for selected categorical columns: `Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, and `Employer_Category`.

### 4. Train-test split and scaling

- Splits the data into **80% training and 20% testing**.
- Uses `random_state=42` for repeatability.
- Applies `StandardScaler`, fitting it on the training features and transforming the test features.

### 5. Models evaluated

The notebook trains and evaluates three classifiers:

- Logistic Regression
- K-Nearest Neighbors (KNN), with `n_neighbors=5`
- Gaussian Naive Bayes

Metrics include precision, recall, F1-score, accuracy, and the confusion matrix.

### 6. Feature engineering experiment

The notebook adds two polynomial features:

- `DTI_Ratio_sq = DTI_Ratio ** 2`
- `Credit_Score_sq = Credit_Score ** 2`

It then removes the original `DTI_Ratio` and `Credit_Score` columns from the feature matrix and repeats the train-test split, scaling, and model evaluation. The results below compare the two notebook experiments.

## 📈 Model Results

The following results are copied from the saved notebook outputs. The first set is the baseline experiment; the second set is after the notebook's feature-engineering change.

### Baseline experiment

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 86.50% | 78.33% | 77.05% | 77.69% |
| KNN (k=5) | 76.00% | 62.75% | 52.46% | 57.14% |
| Gaussian Naive Bayes | 86.50% | 80.36% | 73.77% | 76.92% |

### After feature engineering

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | **88.00%** | 78.46% | **83.61%** | **80.95%** |
| KNN (k=5) | 78.50% | 67.31% | 57.38% | 61.95% |
| Gaussian Naive Bayes | 86.00% | **81.13%** | 70.49% | 75.44% |

### Interpretation

- After the feature-engineering change, **Logistic Regression has the highest recorded accuracy (88.00%) and F1-score (80.95%)**.
- Gaussian Naive Bayes has the highest recorded precision after feature engineering (81.13%).
- KNN has the lowest scores among the three models in both experiments.
- The model with the highest precision is not automatically the best model overall. The appropriate choice depends on the relative cost of false approvals and false rejections.
- These scores are from one 80/20 split. They are not evidence of production-level performance, and the improvement should be validated using cross-validation and a leakage-safe preprocessing pipeline.

## ▶️ Run Locally

### Prerequisites

- Python 3.9 or later recommended
- Jupyter Notebook or JupyterLab
- The dataset file `loan_approval_data.csv`

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

Replace the URL and folder name with your actual GitHub repository details.

### 2. Create and activate a virtual environment (recommended)

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Add the dataset

Place `loan_approval_data.csv` in the same directory as `credit_wise.ipynb`, unless you update the CSV path in the notebook.

### 5. Launch Jupyter

```bash
jupyter notebook
```

Open `credit_wise.ipynb` and run the cells in order.

## ⚠️ Limitations and Recommended Improvements

This repository is a useful learning experiment, but several issues should be addressed before treating it as a robust ML solution:

1. **Preprocessing leakage:** Imputers and encoders are fitted to the full dataset before the train-test split. This allows information from the test set to influence preprocessing. Fit preprocessing only on training data, ideally using a Scikit-learn `Pipeline` and `ColumnTransformer`.
2. **Target encoding:** `LabelEncoder` is used on `Loan_Approved`. Confirm the mapping explicitly so that the positive class used by precision and recall is the intended approval outcome.
3. **Feature engineering consistency:** The notebook creates squared features, removes the original Credit Score and DTI Ratio features, and then re-evaluates the models. Compare this against a controlled experiment that retains the original features and adds the squared terms.
4. **Repeated experiment discipline:** Use the same split and evaluation protocol for fair comparisons. Consider stratified splitting and cross-validation.
5. **Metrics:** Report the full classification report, confusion matrix, and preferably ROC-AUC or precision-recall AUC. Select a model based on the actual costs of false positives and false negatives, not a single metric alone.
6. **Fairness and responsible lending:** Features such as gender and marital status can create fairness and legal concerns. Evaluate disparate impact and applicable regulations before any real-world use. Do not use this notebook to make actual lending decisions.
7. **Reproducibility:** Record library versions, document the dataset source and license, and add a `requirements.txt`.
8. **Deployment:** If extending the project, save the trained pipeline and build a simple Streamlit interface only after the evaluation process is reliable.

## 🛣️ Future Enhancements

- Build a reusable preprocessing and model-training pipeline.
- Add cross-validation and hyperparameter tuning.
- Compare additional models such as Random Forest and Gradient Boosting.
- Add feature importance or SHAP-based explanations where appropriate.
- Create a Streamlit dashboard for interactive predictions.
- Add model persistence, input validation, tests, and clear model/version metadata.
- Evaluate fairness, calibration, and performance on independent data.

## 🤝 Contributing

Contributions and suggestions are welcome. For substantial changes, open an issue first to discuss the proposed improvement.

## 📄 License

No license is specified in the notebook. Add a license file before accepting reuse or redistribution.

---

<div align="center">

**CreditWise AI**  
*Exploring machine learning for loan approval prediction.*

</div>
'''
out='/mnt/data/README.md'
pypandoc.convert_text(readme, 'md', format='md', outputfile=out, extra_args=['--standalone'])
print(os.path.exists(out), os.path.getsize(out))
