# Drug Type Prediction using Machine Learning

This repository implements the provided Mini Project: Drug Type Prediction using Machine Learning.

## Project structure

- `Drug_Type_Prediction.ipynb` — complete Task 1–10 implementation
- `app.py` — Streamlit prediction UI
- `requirements.txt` — Python dependencies
- `.gitignore` — excludes credentials and generated Python files

## Required dataset

Place the professor's `drug200.csv` file in:

`/content/drive/MyDrive/ML Project/drug200.csv`

or in the notebook working directory.

## Tasks

1. **Task 1 — Import and Load the Data**  
   Load the dataset and explore it using `.head()`, `.info()`, `.describe()`, `.shape`, and `.columns`.

2. **Task 2 — Exploratory Data Analysis (EDA)**  
   Perform numerical and categorical visualization, analyze the Drug distribution, study feature relationships, generate a correlation heatmap, and summarize findings.

3. **Task 3 — Missing Values and Outlier Treatment**  
   Check missing values, detect outliers using Boxplots, IQR and Z-score methods, and treat outliers when necessary.

4. **Task 4 — Feature Engineering and Preprocessing**  
   Encode categorical variables, check skewness, apply transformations when required, scale numerical features, and split the dataset into training and testing sets.

5. **Task 5 — Data Augmentation**  
   Apply tabular data augmentation to generate at least 800 synthetic samples and obtain a final dataset of at least 1,000 samples. Compare performance before and after augmentation.

6. **Task 6 — Model Building**  
   Train and evaluate multiple classifiers including Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes, AdaBoost, Gradient Boosting, XGBoost, Extra Trees, Voting, and Stacking Classifiers.

7. **Task 7 — Model Evaluation and Overfitting Check**  
   Evaluate Accuracy, Precision, Recall, F1 Score, ROC-AUC, Confusion Matrix, and Classification Report. Compare training and testing performance.

8. **Task 8 — Hyperparameter Tuning**  
   Use GridSearchCV or RandomizedSearchCV to optimize the best-performing models and record the best hyperparameters and performance improvement.

9. **Task 9 — Model Comparison**  
   Compare the models using training accuracy, testing accuracy, precision, recall, F1 score, augmentation usage, and overfitting status, then identify the best model.

10. **Task 10 — Feature Importance Analysis**  
    Identify the factors influencing drug selection using Feature Importance, Permutation Importance, and optional SHAP analysis.

## Run the Streamlit app

```bash
pip install -r requirements.txt
streamlit run app.py
