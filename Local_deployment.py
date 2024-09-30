from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import numpy as np

# Load the model
model = joblib.load('churn_model.pkl')

# Create a Flask app
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def home():
    return render_template_string('''
        <h1>Bank Churn Prediction Testing(Development Environment Testing)</h1>
        <form action="/predict" method="post">
            <label>Customer ID:</label><input type="text" name="CustomerId" required><br>
            <label>Surname:</label><input type="text" name="Surname" required><br>
            <label>Credit Score:</label><input type="text" name="CreditScore" required><br>
            <label>Geography:</label><input type="text" name="Geography" required><br>
            <label>Gender:</label><input type="text" name="Gender" required><br>
            <label>Age:</label><input type="text" name="Age" required><br>
            <label>Tenure:</label><input type="text" name="Tenure" required><br>
            <label>Balance:</label><input type="text" name="Balance" required><br>
            <label>Num Of Products:</label><input type="text" name="Num Of Products" required><br>
            <label>Has Credit Card:</label><input type="text" name="Has Credit Card" required><br>
            <label>Is Active Member:</label><input type="text" name="Is Active Member" required><br>
            <label>Estimated Salary:</label><input type="text" name="Estimated Salary" required><br>
            <input type="submit" value="Submit">
        </form>
    ''')

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    data = request.form.to_dict()
    
    # Debugging: Print the data received
    print("Received data:", data)

    # Convert input data to DataFrame
    input_df = pd.DataFrame([data])

    # Convert categorical variables to numerical values
    input_df['Geography'] = input_df['Geography'].replace({'France': 2, 'Germany': 1, 'Spain': 0})
    input_df['Gender'] = input_df['Gender'].replace({'Male': 0, 'Female': 1})
    input_df['Num Of Products'] = input_df['Num Of Products'].replace({1: 0, 2: 1, 3: 1, 4: 1})
    input_df['Has Credit Card'] = input_df['Has Credit Card'].replace({'Yes': 1, 'No': 0})
    input_df['Is Active Member'] = input_df['Is Active Member'].replace({'Yes': 1, 'No': 0})

    # Add the Zer Balance feature
    input_df['Zer Balance'] = np.where(input_df['Balance'].astype(float) > 0, 1, 0)

    # Ensure that all features match the trained model
    expected_features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
                         'Balance', 'Num Of Products', 'Has Credit Card', 
                         'Is Active Member', 'Estimated Salary', 'Zer Balance']
    
    # Check for missing features
    for feature in expected_features:
        if feature not in input_df.columns:
            input_df[feature] = 0  # Add missing features with default value

    # Prepare the response with customer details and prediction
    try:
        customer_details = {
            'CustomerId': data['CustomerId'],
            'Surname': data['Surname'],
            'Age': data['Age'],
            'Prediction': int(model.predict(input_df[expected_features])[0])
        }
    except KeyError as e:
        return jsonify({'error': f'Missing field: {e}'}), 400

    return jsonify(customer_details)

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
