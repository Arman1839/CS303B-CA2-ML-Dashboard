# 📱 Mobile Price Prediction - ML Pipeline Dashboard

## 📌 Overview
This project is an interactive Machine Learning web application built using **Streamlit**. It was developed as part of the Continuous Assessment 2 (CA-2) for the Machine Learning & ANN (CS-303B) course to translate theoretical AIML knowledge into a practical, visually appealing web dashboard.

The application provides a complete, end-to-end Machine Learning pipeline allowing users to upload a dataset, perform Exploratory Data Analysis (EDA), clean data, train models, and evaluate performance—all from an intuitive UI.

## 🎯 Problem Statement
The primary objective of this specific deployment is to analyze and predict **Mobile Prices**. By uploading a mobile specifications dataset, the dashboard can be used to either classify phones into price tiers (Classification) or predict their exact market value (Regression).

## ✨ Key Features
The pipeline is divided horizontally into easy-to-navigate tabs:
1. **Data Input:** Upload CSV datasets and visualize the overall data shape using 2D PCA.
2. **Exploratory Data Analysis (EDA):** Auto-generate descriptive statistics, missing value reports, and correlation heatmaps.
3. **Data Cleaning & Engineering:** Handle missing values via imputation and remove outliers using advanced methods like Isolation Forest and DBSCAN.
4. **Feature Selection:** Filter important specifications using Variance Threshold or Information Gain.
5. **Data Splitting:** Custom train-test split sliders.
6. **Model Selection:** Choose from Linear/Logistic Regression, Support Vector Machines (SVM), Random Forest, and KMeans.
7. **Model Training:** Train models with K-Fold Cross-Validation and Hyperparameter Tuning (GridSearch/RandomSearch).
8. **Performance Metrics:** Real-time evaluation (Accuracy, R-Squared, MSE) with automated overfitting/underfitting checks.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Plotly, Seaborn, Matplotlib
* **Machine Learning:** Scikit-Learn

## 🚀 How to Run Locally

1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
