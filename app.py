from crypt import methods
from distutils.log import debug
import pickle
from flask import Flask, jsonify, request, render_template, url_for
from flask_cors import cross_origin
import pandas as pd
import numpy as np

app = Flask(__name__)
@app.route('/')
@cross_origin()
def home():
    return render_template('home.html')

model = pickle.load(open('New_model.pkl', 'rb'))    
@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    data = request.json['data']
    predict_data = [list(data.values())]
    output = model.predict(predict_data)[0]
    return jsonify('{:.2f}'.format(output) )

@app.route('/predict_web', methods=['POST'])
@cross_origin()
def predict_web():
    New_data = [float(x) for x in request.form.values()]
    final_feature = [np.array(New_data)]
    output = model.predict(final_feature)[0]
    return render_template('home.html', prediction_text = 'Airfoil Pressure {:.2f} '.format(output) )


if __name__ =="__main__":
    app.run(debug=True)