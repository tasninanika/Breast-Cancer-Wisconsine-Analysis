<h1 align="center">🩺 Breast Cancer Wisconsin Analysis</h1> 
<p align="center"> A complete machine learning pipeline for predicting whether a breast tumor is <strong>malignant</strong> or <strong>benign</strong> using the Breast Cancer Wisconsin (Diagnostic) dataset. </p> 
<p align="center"> 
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/> 
<img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white"/> 
<img src="https://img.shields.io/badge/Matplotlib-007ACC?style=flat-square&logo=plotly&logoColor=white"/> 
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/> 
</p>

## 📝 Project Overview

The Breast Cancer Wisconsin Analysis project is a machine learning-based analysis aimed at classifying breast tumors as benign (B) or malignant (M) using the Breast Cancer Wisconsin (Diagnostic) Dataset. This project, implemented in a Jupyter Notebook (breast-cancer.ipynb), showcases a complete pipeline including data preprocessing, exploratory data analysis (EDA), model training, evaluation, and visualization of results using Python and popular data science libraries.
The goal is to build and compare multiple classification models to predict tumor diagnosis with high accuracy, leveraging features derived from digitized images of fine needle aspirates (FNA) of breast masses. The project demonstrates the application of machine learning in medical diagnostics, focusing on interpretability and performance evaluation.

## 🚀 Features

Data Loading and Preprocessing:
  - Loads the dataset from a CSV file (breast-cancer-wisconsin.csv).
  - Handles missing values and prepares data for modeling.

Exploratory Data Analysis (EDA):
  - Summarizes dataset characteristics (e.g., shape, data types, missing values).
  - Visualizes feature distributions and relationships.

Model Training:
  - Implements three classifiers: Logistic Regression, Decision Tree, and K-Nearest Neighbors (KNN).
  - Splits data into training and testing sets for evaluation.

Model Evaluation:
  - Uses metrics like accuracy, confusion matrix, and classification report.
  - Applies stratified k-fold cross-validation for robust performance assessment.

Visualization:
  - Generates 2D decision boundary plots using a custom plot2d_classif_model function to visualize model performance.
Custom Utilities:
  - Utilizes a custom ml_utils module for streamlined preprocessing and visualization.

## 📊 Dataset
The dataset used is the Breast Cancer Wisconsin (Diagnostic) Dataset from the UCI Machine Learning Repository. Key details:
  - **Source:** Loaded from breast-cancer-wisconsin.csv.
  - **Size:** 569 samples, 32 columns.
  - **Features:**
  - id: Unique identifier for each sample.
  - diagnosis: Target variable (M = Malignant, B = Benign).
  - 30 numerical features describing cell nuclei characteristics (e.g., radius_mean, texture_mean, perimeter_mean, area_mean, etc.).
- No Missing Values: The dataset is clean, with no null values reported.

## 🛠️ Technologies Used
| Technology           | Description                              |
| -------------------- | ---------------------------------------- |
| **Python**           | Programming language                     |
| **Jupyter Notebook** | Interactive development environment      |
| **Pandas**           | Data manipulation and analysis           |
| **Matplotlib**       | Visualization                            |
| **Scikit-learn**     | Machine learning models and metrics      |
| **Custom Utilities** | `ml_utils.py` for plot and preprocessing |

## 📉 Results

              precision    recall  f1-score   support
           B       0.95      0.97      0.96        71
           M       0.95      0.91      0.93        43
    accuracy                           0.95       114
    macro avg      0.95      0.94      0.94       114
    weighted avg   0.95      0.95      0.95       114

### Cross-Validation
Stratified k-fold cross-validation ensures robust performance estimates across different data splits.

### Visualizations
The plot2d_classif_model function generates 2D decision boundary plots for selected features, aiding in understanding model behavior.

## 📜 License

This project is licensed under the MIT License. Feel free to use, modify, and distribute the code as per the license terms.

## 🙌 Acknowledgments

UCI Machine Learning Repository for providing the Breast Cancer Wisconsin Dataset.
Scikit-learn: For robust machine learning algorithms and evaluation tools.
Pandas and Matplotlib: For data manipulation and visualization.
Jupyter: For an interactive development environment.
