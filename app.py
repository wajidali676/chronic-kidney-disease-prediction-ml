from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load all artifacts
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
num_imputer = joblib.load('num_imputer.pkl')
cat_imputer = joblib.load('cat_imputer.pkl')
feature_columns = joblib.load('feature_columns.pkl')

numeric_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 
                'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']
categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 
                    'appet', 'pe', 'ane']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        
        # Build DataFrame with one row
        input_df = pd.DataFrame([data])
        
        # Convert numeric fields
        for col in numeric_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
        
        # Impute numeric
        input_num = pd.DataFrame(num_imputer.transform(input_df[numeric_cols]), 
                                 columns=numeric_cols)
        
        # Impute categorical
        input_cat = pd.DataFrame(cat_imputer.transform(input_df[categorical_cols]), 
                                 columns=categorical_cols)
        
        # Encode categorical
        for col in categorical_cols:
            le = label_encoders[col]
            val = str(input_cat[col].iloc[0])
            if val not in le.classes_:
                val = le.classes_[0]  # fallback
            input_cat[col] = le.transform([val])[0]
        
        # Combine and scale
        input_proc = pd.concat([input_num, input_cat], axis=1)
        input_proc = input_proc[feature_columns]  # Ensure column order
        input_scaled = scaler.transform(input_proc)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        result = {
            'prediction': 'CKD (Chronic Kidney Disease)' if prediction == 1 else 'Not CKD (Healthy)',
            'confidence': f"{max(probability)*100:.1f}%",
            'risk_color': '#e74c3c' if prediction == 1 else '#2ecc71'
        }
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)