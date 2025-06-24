from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model and scaler
model = load_model('heart_attack_prediction_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user input
        features = [float(request.form[key]) for key in request.form.keys()]
        
        # Scale the features
        features_scaled = scaler.transform([features])
        
        # Predict probability
        probability = model.predict(features_scaled)[0][0]
        
        # Convert to percentage
        probability_percentage = round(probability * 100, 2)
        
        # Display the result
        prediction_text = f"The predicted risk of heart attack is {probability_percentage}%."
        return render_template('result.html', prediction_text=prediction_text)
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
