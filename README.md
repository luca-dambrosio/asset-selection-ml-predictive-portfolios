# 📈 Machine Learning for Asset Selection – Predictive Portfolios from Factor Models

This project explores stock selection using machine learning on a large financial factor dataset. By training both classification and regression models, the approach focuses on identifying stocks likely to significantly outperform, then constructing dynamic monthly portfolios.

## 📁 Files

- `Download_and_Preparation.ipynb` – Downloads the dataset and prepares it for modeling  
- `custom_preprocess.py` – Pipeline for handling missing values, encoding, normalization, and outlier capping  
- `utils.py` – Utility functions for splitting, saving, and loading large datasets in batches  
- `neural_networks.ipynb` – Defines and trains feed-forward neural nets on classification/regression tasks  
- `xgboost.ipynb` – Implements and tunes XGBoost models for both return classification and prediction  
- `FINAL_PROJECT_REPORT#GROUP4.pdf` – Detailed methodology, results, and portfolio evaluation  

## 📊 Summary

### 1. Data
- Sourced ~320k monthly U.S. stock records from Wharton's Global Factor Data (1995–2023)  
- Includes ~160 features: accounting, market-based, industry-specific, and factor-derived  
- Data split:  
  - Train: 1995–2017  
  - Validation: 2018–2021  
  - Test: 2022–2023  

### 2. Models
- **XGBoost classifier** tuned to detect stocks with >20% next-month return probability  
- **Regression variants** for predicting raw returns and ranking stocks  
- **Neural Networks (1–5 layers)** with dropout and batch norm for robustness  
- Selection based on predicted probabilities or return ranks

### 3. Evaluation
- Simulated monthly portfolios using top 2% predicted stocks  
- Benchmarked against the market and Fama-French 3-factor model  
- Regression-based strategies showed higher Sharpe ratios (up to 2.6)  
- Robustness checks across 6 years and different splits confirm consistency

## 🔧 Tools & Libraries

- **Python** (pandas, numpy, matplotlib, scikit-learn, xgboost, tensorflow/keras)  
- Wharton WRDS data + Kenneth French Fama-French factors  
- Custom preprocessing and batch processing to manage memory efficiently

## 📌 Notes

- Binary classification target: next-month return > 20%  
- Regression models outperformed classifiers by avoiding threshold bias  
- Strong empirical performance despite short holding periods and class imbalance  
- Not accounting for transaction costs or reporting lag of firm-level features

## 📚 References

- Gu, Kelly & Xiu (2020). *Empirical Asset Pricing via Machine Learning*  
- Kelly et al. (2023). *Is There a Replication Crisis in Finance?*  
- Kenneth French Fama-French Data: [Link](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)

---

👨‍💻 Completed as part of a final project for Finance with Big Data course.
