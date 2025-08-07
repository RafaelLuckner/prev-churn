# Telco Customer Churn Prediction

## Project Overview
This project aims to analyze customer data from a telecommunications company to identify behavioral and demographic patterns associated with churn (service cancellation). The goal is to build a machine learning model that predicts which customers are at risk of churning, supporting proactive retention strategies.

## Current Phase: Exploratory Data Analysis (EDA)
- Data loaded and split into training and test sets
- Data dictionary and variable descriptions documented
- Descriptive statistics and visualizations for numerical and categorical features
- Initial insights on churn patterns:
  - Customers without a partner or dependents have higher churn rates
  - Churn is concentrated among customers with phone service
  - Customers with fiber optic internet are more likely to churn than those with DSL or no internet
  - Lack of security, backup, device protection, and technical support services is strongly associated with churn

## Next Steps
- Feature engineering: encoding, scaling, and creation of new features
- Model training and evaluation
- Model interpretation and business recommendations

## Project Structure
```
├── data/
│   └── raw/
│       └── Telco_Customer_Churn.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Modeling_and_Evaluation.ipynb
├── src/
│   ├── data_preparation.py
│   ├── feature_engineering.py
│   └── model.py
├── tests/
│   └── test_model.py
├── requirements.txt
└── README.md
```

## Requirements
- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open and run the notebooks in order:
   - 01_EDA.ipynb
   - 02_Feature_Engineering.ipynb
   - 03_Modeling_and_Evaluation.ipynb

## Data Source
- [Kaggle: Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [IBM Community: Telco Customer Churn](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)

## Author
Rafael Luckner
