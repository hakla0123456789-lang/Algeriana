from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ridge_model = pickle.load(open("models/ridge.pkl","rb"))
scaler_model =pickle.load(open("models/scaler.pkl","rb"))

application = Flask(__name__)
app=application

@app.route("/")
def welcome():
    return render_template("index.html")

@app.route("/predict", methods=['GET','POST'])
def prediction():
    if request.method =="POST":
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC= float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
    
        # Create DataFrame with correct feature names
        feature_names = [
            "Temperature", "RH", "Ws", "Rain", 
            "FFMC", "DMC", "ISI", "Classes", "Region"
        ]

        input_df = pd.DataFrame([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]],
                                columns=feature_names)

        # Scale input
        scaled_data = scaler_model.transform(input_df)

        # Model prediction
        result = ridge_model.predict(scaled_data)

        
        return render_template("home.html", results = result[0])

    else:
        return render_template("home.html")
        
        

if __name__=="__main__":
    app.run(host = "0.0.0.0")


