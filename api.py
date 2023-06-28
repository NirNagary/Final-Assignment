# -*- coding: utf-8 -*-


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('traiend_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = request.form.getlist('feature')
    
    input_data = pd.DataFrame([features], columns=['City', 'type', 'city_area', 'Street', 'condition ', 'room_number',
                                                   'Area', 'hasElevator ', 'hasParking ', 'hasStorage ',
                                                   'hasAirCondition ', 'hasBalcony ', 'hasMamad ', 'floor',
                                                   'total_floors'])

    prediction = model.predict(input_data)

    output_text = prediction[0]

    return render_template('index.html', prediction_text='{}'.format(output_text))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    
    app.run(host='0.0.0.0', port=port, debug=True)
