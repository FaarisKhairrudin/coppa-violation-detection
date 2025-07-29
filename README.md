
# 🚸 COPPA Violation Detection
> 🏆 A data science project for FindIT Competition by Universitas Gadjah Mada

## 📌 Overview
In the digital age, mobile apps have become widely accessible to children, raising significant concerns about data privacy. This project aims to build a machine learning model to detect mobile apps that are likely to **violate the Children’s Online Privacy Protection Act (COPPA)**, based on various metadata and attributes of the app.

## 🎯 Objective
Develop a predictive model that flags potentially non-compliant apps to help platforms, developers, and parents create a safer online environment for children.

## 📂 Dataset Description
The dataset **COPPARisk** contains metadata for various mobile applications, including:
- App genre
- Developer country
- Number of downloads
- Presence of privacy policy
- Target audience information
- User ratings
- And more...

The target variable is `coppaRisk` (True/False) indicating whether the app is potentially violating COPPA.

## 🔍 Exploratory Data Analysis (EDA)

Key steps in EDA (see `EDA_final.ipynb`):
- Missing value handling: Addressed placeholders like "ADDRESS NOT LISTED IN PLAYSTORE" and "CANNOT IDENTIFY COUNTRY".
- Skewness correction: Applied log transformation to skewed features like `userRatingCount` and `downloads`.
- Feature encoding:
  - Ordinal columns mapped to integers
  - High-cardinality categoricals encoded using TargetEncoder (CatBoost)
  - Low-cardinality categoricals encoded with One-Hot Encoding
- Insights:
  - Top developer countries include: United States, Hong Kong, China
  - Certain genres and developer regions show higher COPPA risk rates

## 🤖 Modeling

Implemented in `modeling_coppaRISK.ipynb`, using tree-based classifiers with `class_weight` to address data imbalance.  
All models were tuned using **Optuna** for optimal hyperparameters, and evaluated using **AUC** as the primary metric.

| Model                   | AUC Score |
|------------------------|-----------|
| XGBoost                | 0.8978    |
| LightGBM               | 0.8923    |
| CatBoost               | 0.8911    |
| Gradient Boosting Classifier | 0.8881    |

✅ Best Model: **XGBoost**, selected based on highest AUC (0.8978).

## ⚙️ Project Structure

```
coppa-violation-detection/
│
├── code/
│   ├── EDA_raw.ipynb                 # Original/raw exploratory analysis
│   ├── EDA_final.ipynb               # Cleaned EDA with preprocessing & feature engineering
│   └── Modeling_coppaRISK.ipynb      # Model training, hyperparameter tuning, evaluation
│
├── data/
│   ├── train.csv                     # Original training data
│   ├── test.csv                      # Original test data
│   ├── target.csv                    # Labels for training data
│   ├── test_with_target.csv          # Test data with ground truth (for evaluation)
│   ├── train_with_target.csv         # Merged train + label (convenience)
│   ├── new_train_2_no_encode.csv     # Processed train (pre-encoded)
│   └── new_test_2_no_encode.csv      # Processed test (pre-encoded)
│
├── submission/                       # Folder for competition submission files
│
└── README.md                         # Project documentation

```

## 🚀 How to Run

1. Clone the repository:
```bash
   git clone https://github.com/FaarisKhairrudin/coppa-violation-detection.git
   cd coppa-violation-detection
```
2. Open the Jupyter notebooks:
   - EDA_final.ipynb
   - modeling_coppaRISK.ipynb

✅ No special environment setup needed — just open the notebooks.

## 🧠 Challenges & Considerations

- Missing & ambiguous values: Addressed string placeholders for unknown data.
- Feature Engineering: Encoding strategy adjusted based on cardinality and semantics.
- Class imbalance: Mitigated using tree models with `class_weight`.
- Model interpretability: Tree models offer balance between performance and explainability.

## 📎 Dependencies

- pandas, numpy, matplotlib, seaborn
- scikit-learn
- xgboost, lightgbm, catboost
- category_encoders (CatBoostEncoder)
