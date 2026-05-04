# Loan Approval Prediction Model
Built a complete ML pipeline to predict loan approval decisions on a dataset of 58,645 applicants. 
The project covers everything from raw data cleaning through to model comparison, hyperparameter 
tuning, and ensemble learning. A regression component also predicts the maximum loan amount 
for approved applicants.

## Notebooks
| # | Notebook | Description |
|---|---|---|
| 1 | Data Understanding & Preprocessing | Feature selection, missing value treatment, outlier handling, encoding, and scaling |
| 2 | Classification & Hyperparameter Tuning | Logistic Regression, Naïve Bayes, KNN — tuned via GridSearchCV, evaluated on Precision, Recall, F1, AUC-ROC |
| 3 | Regression & Ensemble Learning | Decision Tree regressors (fully grown vs pruned) and a soft-voting ensemble classifier |

## Key Results

| Model | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|
| Naïve Bayes | 0.48 | 0.62 | 0.54 | 0.86 |
| Logistic Regression | 0.71 | 0.40 | 0.51 | 0.87 |
| KNN Tuned (K=7) | 0.82 | 0.55 | 0.66 | 0.86 |
| **Ensemble (KNN + LR)** | **0.82** | 0.50 | 0.62 | **0.90** |

> KNN selected as best single model (F1=0.66). Ensemble achieved the highest AUC-ROC of 0.90.

## Tools
Python · pandas · scikit-learn · Matplotlib · Seaborn · Jupyter Notebook

## Author
**Rashmi Tharuka** · BSc Business Data Analysis, University of Westminster  
