from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  
import joblib
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

try:
    model = load_model('model/new_diabetes_model.h5') 
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")

try:
    scaler = joblib.load('scaler.pkl')  
    logging.info("Scaler loaded successfully.")
except Exception as e:
    logging.error(f"Error loading scaler: {e}")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
  
        pregnancies = request.form.get('pregnancies', type=float)
        glucose = request.form.get('glucose', type=float)
        blood_pressure = request.form.get('bloodPressure', type=float)
        skin_thickness = request.form.get('skinThickness', type=float)
        insulin = request.form.get('insulin', type=float)
        bmi = request.form.get('bmi', type=float)
        diabetes_pedigree_function = request.form.get('diabetesPedigreeFunction', type=float)
        age = request.form.get('age', type=float)

   
        if None in [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]:
            return render_template('index.html', prediction_text="Please provide valid input values for all fields.")
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        result = "You have diabetes!" if prediction[0] > 0.5 else "You do not have diabetes."
        logging.info(f"Prediction value: {prediction}")

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return render_template('index.html', prediction_text="Error in input or processing. Please try again.")

if __name__ == "__main__":
    app.run(debug=True)
