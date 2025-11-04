import os
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Set the number of cores to use for joblib
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

app = Flask(__name__)

# Load the models and encoders
model = joblib.load('depression_model_best.pkl')
scaler = joblib.load('scaler_best.pkl')
label_encoder_new_degree = joblib.load('label_encoder_new_degree_best.pkl')
label_encoder_depression = joblib.load('label_encoder_depression_best.pkl')

# City encoding
city_encoding = {
    'Agra': 0, 'Ahmedabad': 1, 'Bangalore': 2, 'Bhopal': 3, 'Chennai': 4, 'Delhi': 5, 'Faridabad': 6, 'Ghaziabad': 7,
    'Hyderabad': 8, 'Indore': 9, 'Jaipur': 10, 'Kalyan': 11, 'Kanpur': 12, 'Kolkata': 13, 'Lucknow': 14, 'Ludhiana': 15,
    'Meerut': 16, 'Mumbai': 17, 'Nagpur': 18, 'Nashik': 19, 'Patna': 20, 'Pune': 21, 'Rajkot': 22, 'Srinagar': 23,
    'Surat': 24, 'Thane': 25, 'Vadodara': 26, 'Varanasi': 27, 'Vasai-Virar': 28, 'Visakhapatnam': 29
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/depress')
def depress():
    return render_template('depress.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    df = pd.DataFrame([data])

    # Preprocess the input data
    try:
        df['Gender'] = df['gender'].astype(int)
        df['Age'] = df['age'].astype(int)
        df['Academic Pressure'] = df['academic_pressure'].astype(float)
        df['CGPA'] = df['cgpa'].astype(float)
        df['Study Satisfaction'] = df['study_satisfaction'].astype(float)
        df['Sleep Duration'] = df['sleep_duration'].astype(int)
        df['Work/Study Hours'] = df['work_study_hours'].astype(float)
        df['Financial Stress'] = df['financial_stress'].astype(float)
        df['Dietary Habits'] = df['dietary_habits'].astype(int)
        df['New_Degree'] = df['new_degree'].astype(int)
        df['Have you ever had suicidal thoughts ?'] = df['suicidal_thoughts'].astype(int)
        df['Family History of Mental Illness'] = df['family_history'].astype(int)
        df['City_encoded'] = df['city'].map(city_encoding).astype(int)

        
        df['CGPA'] = df['CGPA'].astype(float)

        # Cluster CGPA into specified ranges
        bins = [0, 5, 6, 7, 8, 9, 10]
        labels = ['<5', '5 - <6', '6 - <7', '7 - <8', '8 - <9', '9 - 10']
        df['CGPA_range'] = pd.cut(df['CGPA'], bins=bins, labels=labels, right=False)

        

        # Convert CGPA labels to numerical values
        cgpa_map = {'<5': 0, '5 - <6': 1, '6 - <7': 2, '7 - <8': 3, '8 - <9': 4, '9 - 10': 5}
        df['CGPA'] = df['CGPA_range'].map(cgpa_map)
    except ValueError as e:
        return jsonify({'error': f"Error in input data: {e}"})

    
    column_order = ['Gender', 'Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Sleep Duration', 
                    'Dietary Habits', 'Have you ever had suicidal thoughts ?', 'Work/Study Hours', 'Financial Stress', 
                    'Family History of Mental Illness', 'New_Degree', 'City_encoded']
    df = df[column_order]

    # Scale numerical features
    numerical_features = [
        'Gender', 'Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Sleep Duration', 
        'Dietary Habits', 'Have you ever had suicidal thoughts ?', 'Work/Study Hours', 
        'Financial Stress', 'Family History of Mental Illness', 'New_Degree', 'City_encoded'
    ]
    df[numerical_features] = scaler.transform(df[numerical_features])

    # Predict the result
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)

    prediction_label = label_encoder_depression.inverse_transform(prediction)[0]
    prediction_probability = prediction_proba[0][prediction[0]]

    # Convert prediction_label to a native Python type
    prediction_label = int(prediction_label)

    return jsonify({'prediction': prediction_label, 'probability': prediction_probability})

if __name__ == '__main__':
    app.run(debug=True)