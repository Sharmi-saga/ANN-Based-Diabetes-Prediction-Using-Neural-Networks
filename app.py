from flask import Flask, render_template, request
import numpy as np
import joblib  
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

try:
    model = joblib.load('model/diabetes_model.pkl')  
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def predict():
    try:
     
        try:
            pregnancies = float(request.form['pregnancies'])
            glucose = float(request.form['glucose'])
            blood_pressure = float(request.form['bloodPressure'])
            skin_thickness = float(request.form['skinThickness'])
            insulin = float(request.form['insulin'])
            bmi = float(request.form['bmi'])
            diabetes_pedigree_function = float(request.form['diabetesPedigreeFunction'])
            age = float(request.form['age'])

            logging.info(f"Form data received: {pregnancies}, {glucose}, {blood_pressure}, {skin_thickness}, {insulin}, {bmi}, {diabetes_pedigree_function}, {age}")
        except ValueError as ve:
            logging.error(f"ValueError: {ve}")
            return render_template('index.html', prediction_text="Please enter valid numbers for all fields.")

     
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

     
        logging.info(f"Input data shape: {input_data.shape}, content: {input_data}")

        if model is None:
            return render_template('index.html', prediction_text="Model is not loaded properly.")
   
        prediction = model.predict(input_data)

        logging.info(f"Model Prediction Output: {prediction}")

        result = "You have diabetes!" if prediction[0] > 0.5 else "You do not have diabetes."
        logging.info(f"Prediction Result: {result}")


        return render_template('index.html', prediction_text=result)

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return render_template('index.html', prediction_text="Error in input or processing. Please try again.")

if __name__ == "__main__":
    app.run(debug=True)
