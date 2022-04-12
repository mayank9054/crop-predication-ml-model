import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


# Create flask app
app = Flask(__name__)
model = pickle.load(open("model_crop.pkl", "rb"))

@app.route("/")
def Home():
    return "Welcome To Crop predection Model"

@app.route("/predict", methods = ["POST","GET"])
def predict():
    if request.method == "POST":
        data = request.get_json(force=True) 
        prediction = model.predict([[data['n'],data['p'],data['k'],data['t'],data['h'],data['ph'],data['r']]]) 
        print(prediction)
        output = prediction[0]
        print(output)
        return jsonify(output)
    else : 
        return "predict"
    
 
if __name__ == "__main__":
    app.run(debug=True)
