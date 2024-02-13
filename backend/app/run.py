from flask import Flask, render_template, request, jsonify
import pandas as pd
import xgboost
import logging
import joblib  # Assuming you have saved your XGBoost model using joblib

app = Flask(__name__)

# Load the XGBoost model
xgb_model = joblib.load('C:/Users/HP/Desktop/EstateVerse-Base-Model/backend/app/models/xgboost_model.pkl')
X_train = pd.read_csv('C:/Users/HP/Desktop/EstateVerse-Base-Model/backend/app/models/X_train.csv')  

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'An error occurred during prediction.'})

@app.route('/log', methods=['GET'])
def log():
    try:
        log_data = request.json.get('log_data')
        print('JavaScript Log:', log_data)
        return jsonify({'success': True})
    except Exception as e:
        print('Error handling log:', str(e))
        return jsonify({'error': 'An error occurred handling the log.'})

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        if request.method == 'POST':
            print("Entered /predict route")
            # Handle POST request as usual
            input_data = request.json
            
            categorical_columns = ['active', 'furnishingDesc', 'isMaintenance', 'loanAvailable', 'parking', 'propertyType', 'sharedAccomodation', 'swimmingPool', 'type_bhk', 'LIFT', 'GYM', 'INTERNET', 'AC', 'CLUB', 'INTERCOM', 'POOL', 'CPA', 'FS', 'SERVANT', 'SECURITY', 'SC', 'GP', 'PARK', 'RWH', 'STP', 'HK', 'PB', 'VP']
            numeric_columns = ['bathroom', 'floor', 'property_age', 'property_size', 'totalFloor']

            # One-hot encode input data using the columns from the training set
            input_df = pd.DataFrame([input_data])
            input_df = pd.get_dummies(input_df, columns=categorical_columns)

            # Extract latitude and longitude from 'location' tuple
            input_df[['latitude', 'longitude']] = pd.DataFrame([eval(input_df['location'].iloc[0])])

            # Drop the original 'location' column
            input_df = input_df.drop('location', axis=1)

            # Align columns with the training set DataFrame
            input_df = input_df.reindex(columns=X_train.columns, fill_value=0)

            # Convert relevant columns to numeric types
            input_df[numeric_columns] = input_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
            input_df = input_df.drop('Unnamed: 0', axis=1)

            # Make predictions using the model
            predictions = xgb_model.predict(input_df)

            # Unpack the array of predictions
            deposit_pred, rent_pred = predictions[0][0], predictions[0][1]

            # Return the prediction results
            return jsonify({'deposit_pred': float(deposit_pred), 'rent_pred': float(rent_pred)})
        else:
            # Render a template for GET requests
            return render_template('index.html')        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)