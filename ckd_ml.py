"""
================================================================================
CHRONIC KIDNEY DISEASE (CKD) PREDICTION - COMPLETE ML PIPELINE
================================================================================
Section A : Data Loading & Cleaning (handles quoted headers, trailing commas)
Section B : Exploratory Data Analysis (EDA)
Section C : Data Preprocessing (Imputation, Encoding, Scaling)
Section D : Model Development (9 Algorithms)
Section E : Model Evaluation (Cross-Validation + Test Metrics)
Section F : Model Comparison & Best Model Selection
Section G : Confusion Matrices & Feature Importance
Section H : Unsupervised Learning - K-Means Clustering
Section I : Save Artifacts for Web Interface
================================================================================
"""

# ---------------------------- IMPORTS ----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             adjusted_rand_score, silhouette_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                               AdaBoostClassifier, VotingClassifier)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


# ==============================================================================
# SECTION A : DATA LOADING & CLEANING
# ==============================================================================
print("="*70)
print("SECTION A : DATA LOADING & CLEANING")
print("="*70)

raw_data = []
with open('csv_result-chronic_kidney_disease.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    header = [h.strip().strip("'").strip('"') for h in header]
    for row in reader:
        while row and row[-1] == '':
            row.pop()
        if len(row) != len(header):
            continue
        raw_data.append(row)

df = pd.DataFrame(raw_data, columns=header)
df.replace('?', np.nan, inplace=True)
df.drop('id', axis=1, inplace=True)

numeric_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 
                'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']
categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 
                    'appet', 'pe', 'ane']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['class'] = df['class'].map({'ckd': 1, 'notckd': 0})

print(f"Dataset Shape: {df.shape}")
print(f"CKD: {(df['class']==1).sum()} | Not CKD: {(df['class']==0).sum()}")


# ==============================================================================
# SECTION B : EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
print("\n" + "="*70)
print("SECTION B : EXPLORATORY DATA ANALYSIS (EDA)")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Target Distribution
df['class'].value_counts().plot(kind='bar', ax=axes[0,0], color=['#e74c3c', '#2ecc71'])
axes[0,0].set_title('CKD Target Distribution', fontsize=14, fontweight='bold')
axes[0,0].set_xticklabels(['CKD (1)', 'Not CKD (0)'], rotation=0)
axes[0,0].set_ylabel('Count')

# 2. Missing Values
missing_data = df.isnull().sum().sort_values(ascending=False)
missing_data = missing_data[missing_data > 0]
missing_data.plot(kind='barh', ax=axes[0,1], color='#3498db')
axes[0,1].set_title('Missing Values by Feature', fontsize=14, fontweight='bold')
axes[0,1].set_xlabel('Count')

# 3. Age Distribution by Class
df[df['class']==1]['age'].dropna().plot(kind='hist', bins=20, alpha=0.7, ax=axes[1,0], 
                                        color='#e74c3c', label='CKD')
df[df['class']==0]['age'].dropna().plot(kind='hist', bins=20, alpha=0.7, ax=axes[1,0], 
                                        color='#2ecc71', label='Not CKD')
axes[1,0].set_title('Age Distribution by CKD Status', fontsize=14, fontweight='bold')
axes[1,0].legend()
axes[1,0].set_xlabel('Age')

# 4. Correlation Heatmap
numeric_df = df[numeric_cols + ['class']]
corr = numeric_df.corr()
im = axes[1,1].imshow(corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
axes[1,1].set_xticks(range(len(corr.columns)))
axes[1,1].set_yticks(range(len(corr.columns)))
axes[1,1].set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
axes[1,1].set_yticklabels(corr.columns, fontsize=8)
axes[1,1].set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=axes[1,1])

plt.tight_layout()
plt.savefig('1_EDA_Analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nTop 5 Correlations with CKD:")
target_corr = corr['class'].drop('class').sort_values(key=abs, ascending=False)
for feat, val in target_corr.head(5).items():
    print(f"  {feat}: {val:.3f}")


# ==============================================================================
# SECTION C : DATA PREPROCESSING
# ==============================================================================
print("\n" + "="*70)
print("SECTION C : DATA PREPROCESSING")
print("="*70)

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Imputation
num_imputer = SimpleImputer(strategy='median')
X_train_num = pd.DataFrame(num_imputer.fit_transform(X_train[numeric_cols]), 
                           columns=numeric_cols, index=X_train.index)
X_test_num = pd.DataFrame(num_imputer.transform(X_test[numeric_cols]), 
                          columns=numeric_cols, index=X_test.index)

cat_imputer = SimpleImputer(strategy='most_frequent')
X_train_cat = pd.DataFrame(cat_imputer.fit_transform(X_train[categorical_cols]), 
                           columns=categorical_cols, index=X_train.index)
X_test_cat = pd.DataFrame(cat_imputer.transform(X_test[categorical_cols]), 
                          columns=categorical_cols, index=X_test.index)

# Label Encoding
label_encoders = {}
X_train_cat_enc = X_train_cat.copy()
X_test_cat_enc = X_test_cat.copy()

for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X_train_cat[col], X_test_cat[col]], axis=0)
    le.fit(combined.astype(str))
    X_train_cat_enc[col] = le.transform(X_train_cat[col].astype(str))
    X_test_cat_enc[col] = le.transform(X_test_cat[col].astype(str))
    label_encoders[col] = le

# Combine & Scale
X_train_proc = pd.concat([X_train_num, X_train_cat_enc], axis=1)
X_test_proc = pd.concat([X_test_num, X_test_cat_enc], axis=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_proc)
X_test_scaled = scaler.transform(X_test_proc)

print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print("Preprocessing: Imputation -> Encoding -> Scaling [DONE]")


# ==============================================================================
# SECTION D : MODEL DEVELOPMENT
# ==============================================================================
print("\n" + "="*70)
print("SECTION D : MODEL DEVELOPMENT")
print("="*70)

models = {
    # Academic Curriculum Algorithms
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'SVM': SVC(probability=True, random_state=42, kernel='rbf'),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    
    # Required Ensemble Methods
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42, max_depth=5),
    'AdaBoost': AdaBoostClassifier(n_estimators=200, random_state=42),
    
    # Voting Classifier
    'Voting Classifier': VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)),
            ('svm', SVC(probability=True, random_state=42, kernel='rbf'))
        ],
        voting='soft'
    )
}

for name in models.keys():
    print(f"  -> {name}")


# ==============================================================================
# SECTION E : MODEL EVALUATION (5-Fold Cross-Validation)
# ==============================================================================
print("\n" + "="*70)
print("SECTION E : MODEL EVALUATION")
print("="*70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, model in models.items():
    if name in ['KNN', 'SVM', 'Logistic Regression']:
        X_tr = X_train_scaled
        X_te = X_test_scaled
    else:
        X_tr = X_train_proc.values
        X_te = X_test_proc.values
    
    cv_acc = cross_val_score(model, X_tr, y_train, cv=cv, scoring='accuracy')
    cv_prec = cross_val_score(model, X_tr, y_train, cv=cv, scoring='precision')
    cv_rec = cross_val_score(model, X_tr, y_train, cv=cv, scoring='recall')
    cv_f1 = cross_val_score(model, X_tr, y_train, cv=cv, scoring='f1')
    
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    
    results.append({
        'Model': name,
        'CV_Accuracy': cv_acc.mean(),
        'CV_Precision': cv_prec.mean(),
        'CV_Recall': cv_rec.mean(),
        'CV_F1': cv_f1.mean(),
        'Test_Accuracy': accuracy_score(y_test, y_pred),
        'Test_Precision': precision_score(y_test, y_pred, zero_division=0),
        'Test_Recall': recall_score(y_test, y_pred, zero_division=0),
        'Test_F1': f1_score(y_test, y_pred, zero_division=0)
    })
    
    print(f"{name:20s} | CV Acc: {cv_acc.mean():.3f} | Test Acc: {accuracy_score(y_test, y_pred):.3f} | Test F1: {f1_score(y_test, y_pred, zero_division=0):.3f}")

results_df = pd.DataFrame(results)
print("\n" + "="*70)
print("COMPLETE RESULTS:")
print("="*70)
print(results_df.round(3).to_string(index=False))


# ==============================================================================
# SECTION F : MODEL COMPARISON & BEST MODEL SELECTION
# ==============================================================================
print("\n" + "="*70)
print("SECTION F : BEST MODEL SELECTION")
print("="*70)

best_idx = results_df['Test_F1'].idxmax()
best_name = results_df.loc[best_idx, 'Model']
print(f"\n>>> BEST MODEL: {best_name} <<<")
print(f"Test Accuracy : {results_df.loc[best_idx, 'Test_Accuracy']:.3f}")
print(f"Test Precision: {results_df.loc[best_idx, 'Test_Precision']:.3f}")
print(f"Test Recall   : {results_df.loc[best_idx, 'Test_Recall']:.3f}")
print(f"Test F1-Score : {results_df.loc[best_idx, 'Test_F1']:.3f}")


# ==============================================================================
# SECTION G : CONFUSION MATRICES & FEATURE IMPORTANCE
# ==============================================================================
print("\n" + "="*70)
print("SECTION G : CONFUSION MATRICES & FEATURE IMPORTANCE")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

key_models = ['KNN', 'Naive Bayes', 'Decision Tree', 'SVM', 'Random Forest', 'Gradient Boosting']

for idx, name in enumerate(key_models):
    model = models[name]
    if name in ['KNN', 'SVM']:
        X_te = X_test_scaled
    else:
        X_te = X_test_proc.values
    
    y_pred = model.predict(X_te)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Not CKD', 'CKD'], yticklabels=['Not CKD', 'CKD'])
    axes[idx].set_title(f'{name}\nAcc: {accuracy_score(y_test, y_pred):.3f}', 
                        fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('True')
    axes[idx].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('2_Confusion_Matrices.png', dpi=150, bbox_inches='tight')
plt.show()

# Feature Importance
rf_model = models['Random Forest']
rf_model.fit(X_train_proc.values, y_train)
importances = rf_model.feature_importances_
feat_imp = pd.DataFrame({
    'Feature': X_train_proc.columns,
    'Importance': importances
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(12, 8))
plt.barh(feat_imp['Feature'], feat_imp['Importance'], color='#2ecc71')
plt.title('Random Forest - Feature Importance', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('3_Feature_Importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nTop 10 Features:")
print(feat_imp.tail(10).to_string(index=False))


# ==============================================================================
# SECTION H : UNSUPERVISED LEARNING - K-MEANS CLUSTERING
# ==============================================================================
print("\n" + "="*70)
print("SECTION H : K-MEANS CLUSTERING")
print("="*70)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_train_scaled)

ari = adjusted_rand_score(y_train, clusters)
sil = silhouette_score(X_train_scaled, clusters)

print(f"Adjusted Rand Index : {ari:.3f}")
print(f"Silhouette Score    : {sil:.3f}")
print(f"Cluster 0: {np.sum(clusters==0)} | Cluster 1: {np.sum(clusters==1)}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='coolwarm', 
                alpha=0.7, edgecolors='k')
axes[0].set_title('True Labels', fontsize=14, fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')

axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='coolwarm', 
                alpha=0.7, edgecolors='k')
axes[1].set_title('K-Means (k=2)', fontsize=14, fontweight='bold')
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')

plt.tight_layout()
plt.savefig('4_KMeans_Clustering.png', dpi=150, bbox_inches='tight')
plt.show()


# ==============================================================================
# SECTION I : SAVE ARTIFACTS FOR WEB INTERFACE
# ==============================================================================
print("\n" + "="*70)
print("SECTION I : SAVING MODEL ARTIFACTS")
print("="*70)

joblib.dump(models['Random Forest'], 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(num_imputer, 'num_imputer.pkl')
joblib.dump(cat_imputer, 'cat_imputer.pkl')
joblib.dump(X_train_proc.columns.tolist(), 'feature_columns.pkl')

print("Saved: best_model.pkl, scaler.pkl, label_encoders.pkl")
print("Saved: num_imputer.pkl, cat_imputer.pkl, feature_columns.pkl")
print("\n>>> PIPELINE COMPLETE <<<")
print("="*70)