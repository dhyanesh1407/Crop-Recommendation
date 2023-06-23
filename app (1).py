
from flask import Flask, render_template, request
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

with open("api/model.pkl", "rb") as f:
    model = pickle.load(f)

encoder = LabelEncoder()
encoder.classes_ = np.load('api/classes.npy', allow_pickle=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosporus'])
    K = int(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])

    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    
    y_pred = model.predict([feature_list])
    pred_label = encoder.inverse_transform(y_pred)[0]
    

    result = f"{pred_label} is the best crop to be cultivated right there"
    
    return render_template('index.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)    