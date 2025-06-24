from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('heart_attack_prediction_model.h5')

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
  
# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get input from form
    features = [float(x) for x in request.form.values()]
    features = np.array(features).reshape(1, -1)

    # Normalize the features using the loaded scaler
    features_scaled = scaler.transform(features)

    # Predict using the model
    prediction = model.predict(features_scaled)[0][0]

    # Determine the result message
    if prediction > 0.5:
        result = "This person is at risk of a heart attack."
    else:
        result = "This person is not at risk of a heart attack.Please take the following measures"

    return render_template('result.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
