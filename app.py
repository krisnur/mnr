# Import necessary libraries
from flask import Flask, request, jsonify
from tensorflow import keras
import joblib
import pandas as pd

# Create a Flask web application
app = Flask(__name__)

# Load your pre-trained ML model from .pkl file
model = joblib.load('maintenance_modelrf_v_002.pkl')

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.get_json()

    # Ensure the input data is in the format expected by your model
    # Perform any necessary preprocessing here
    # For example, you can convert the input data to a DataFrame if needed
    input_data = pd.DataFrame(data, index=[0])

    # Make predictions using your model
    predictions = model.predict(input_data)

    # Return the predictions as JSON
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)