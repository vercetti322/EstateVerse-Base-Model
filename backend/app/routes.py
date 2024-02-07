from flask import render_template, request, jsonify
from . import app, model
import pandas as pd

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # get input data
        input_data = request.get_json()
        
        # create df
        input_df = pd.DataFrame(input_data)
        
        # make predicition using model
        prediction = model.predict(input_df)
        
        # get deposit_pred, rent_pred
        deposit_pred, rent_pred = prediction[0][0], prediction[0][1]
        
        return jsonify({'expected deposit': deposit_pred, 'expected rent': rent_pred})
    
    except Exception as e:
        return jsonify({'error': str(e)})