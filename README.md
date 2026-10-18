# Chronic Kidney Disease Prediction

An educational machine-learning project that compares classification models for chronic kidney disease (CKD) and provides a Flask web interface for making predictions from patient measurements.

> **Medical disclaimer:** This project is for educational and research purposes only. It is not a medical device and must not be used to diagnose, treat, or make decisions about a person's health.

## Project Overview

The project explores supervised classification and unsupervised clustering on the CKD dataset included in this repository. It handles missing values, compares nine classifiers, evaluates a two-cluster K-Means model, and saves a Random Forest model for the Flask application.

The web form accepts 24 input features and returns a predicted class (`CKD` or `Not CKD`) and the model's maximum class probability. That probability is not a clinically validated or calibrated measure of certainty.

## Dataset

The bundled file is [`csv_result-chronic_kidney_disease.csv`](csv_result-chronic_kidney_disease.csv). The training script parses 399 usable records and 24 input features: 14 numeric and 10 categorical. Rows with an unexpected number of fields are skipped. The target labels are `ckd` and `notckd`; `?` values are treated as missing.

Numeric features: `age`, `bp`, `sg`, `al`, `su`, `bgr`, `bu`, `sc`, `sod`, `pot`, `hemo`, `pcv`, `wbcc`, and `rbcc`.

Categorical features: `rbc`, `pc`, `pcc`, `ba`, `htn`, `dm`, `cad`, `appet`, `pe`, and `ane`.

## Methodology

1. Split the records into training and test sets using an 80/20 stratified split (`random_state=42`): 319 training records and 80 test records.
2. Impute numeric missing values with the training-set median and categorical missing values with the most frequent category.
3. Encode categorical features with `LabelEncoder`. Standardize inputs for KNN, SVM, Logistic Regression, and the clustering analysis.
4. Evaluate the classifiers with stratified 5-fold cross-validation on the training set, then report metrics on the held-out test set.
5. Select the saved classifier by the highest test-set F1 score. The current training script selects Random Forest.

### Supervised Models

| Model | 5-fold CV accuracy | Test accuracy | Test precision | Test recall | Test F1 |
|---|---:|---:|---:|---:|---:|
| K-Nearest Neighbors (KNN) | 0.972 | 0.975 | 1.000 | 0.960 | 0.980 |
| Gaussian Naive Bayes | 0.962 | 0.975 | 1.000 | 0.960 | 0.980 |
| Decision Tree | 0.972 | 0.975 | 0.980 | 0.980 | 0.980 |
| Support Vector Machine (RBF) | 0.997 | 0.988 | 1.000 | 0.980 | 0.990 |
| Logistic Regression | 0.991 | 0.988 | 1.000 | 0.980 | 0.990 |
| Random Forest | 0.991 | 1.000 | 1.000 | 1.000 | 1.000 |
| Gradient Boosting | 0.972 | 0.963 | 1.000 | 0.940 | 0.969 |
| AdaBoost | 0.988 | 0.988 | 1.000 | 0.980 | 0.990 |
| Soft Voting (Logistic Regression, Random Forest, SVM) | 0.991 | 0.963 | 0.980 | 0.960 | 0.970 |

Metrics above were reproduced by running the repository's training and evaluation logic with the bundled CSV and the current Python environment; values are rounded to three decimals. CKD is the positive class. Results can vary with library versions or data changes. The best model is selected using the same held-out test set shown here, so its score may be optimistic; independent external validation is needed.

### Unsupervised Analysis

K-Means (`k=2`) is run on the scaled training data. The script reports Adjusted Rand Index and silhouette score and creates a PCA visualization alongside the true labels.

## Visualizations

The training script creates these figures:

- `1_EDA_Analysis.png`: exploratory data analysis
- `2_Confusion_Matrices.png`: confusion matrices for selected classifiers
- `3_Feature_Importance.png`: Random Forest feature importance
- `4_KMeans_Clustering.png`: K-Means clusters and PCA projection

## Run the Web App

Python 3.11 or a compatible Python version is recommended. From the repository root, install the required packages and start Flask:

```powershell
python -m pip install Flask joblib numpy pandas scikit-learn matplotlib seaborn
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in a browser. The saved `.pkl` artifacts are included in the repository and are loaded from the current working directory.

To rerun the analysis and regenerate the saved model artifacts and plots:

```powershell
python ckd_ml.py
```

The script uses `matplotlib` plots during execution and may pause until each plot window is closed.

## Repository Contents

- `app.py`: Flask routes and prediction processing
- `templates/index.html`: patient input form and results interface
- `ckd_ml.py`: data preparation, model evaluation, clustering, and artifact generation
- `best_model.pkl` and supporting `.pkl` files: saved model and preprocessing objects
- `csv_result-chronic_kidney_disease.csv`: dataset used by the training script
- `MACHINE LEARNING PROJECT WAJID_ZAIN_GROUP.pdf`: accompanying university project report
- `*_Analysis.png`, `*_Matrices.png`, and related PNG files: generated figures

## Important Implementation Note

The training script fits the saved Random Forest on imputed and encoded but **unscaled** features. The current Flask route scales the input before passing it to that model. This preprocessing mismatch can make web predictions differ from the reported evaluation; align training and inference preprocessing before relying on the web output.

## License and Attribution

The project code is released under the MIT License. The license does not replace any terms that may apply to the bundled dataset, report, or third-party material; check their original sources before reusing them.

Project report prepared by Wajid Ali and Zain Babar, Lahore Garrison University.