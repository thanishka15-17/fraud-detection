
from flask import Flask, request, jsonify, render_template
import pickle, json
import numpy as np
import pandas as pd

app = Flask(__name__)

model  = pickle.load(open("fraud_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
with open("feature_cols.json") as f:
    feature_cols = json.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    Amount_Log   = np.log1p(data["Transaction_Amount"])
    Failed_Ratio = data["Failed_Transaction_Count_7d"] / (data["Daily_Transaction_Count"] + 1)

    input_dict = {
        "Transaction_Amount":          data["Transaction_Amount"],
        "Transaction_Type":            data["Transaction_Type"],
        "Account_Balance":             data["Account_Balance"],
        "Device_Type":                 data["Device_Type"],
        "Location":                    data["Location"],
        "Merchant_Category":           data["Merchant_Category"],
        "Daily_Transaction_Count":     data["Daily_Transaction_Count"],
        "Avg_Transaction_Amount_7d":   data["Avg_Transaction_Amount_7d"],
        "Failed_Transaction_Count_7d": data["Failed_Transaction_Count_7d"],
        "Card_Type":                   data["Card_Type"],
        "Card_Age":                    data["Card_Age"],
        "Transaction_Distance":        data["Transaction_Distance"],
        "Authentication_Method":       data["Authentication_Method"],
        "Is_Weekend":                  data["Is_Weekend"],
        "Hour":                        data["Hour"],
        "Day_Of_Week":                 data["Day_Of_Week"],
        "Month":                       data["Month"],
        "Amount_Log":                  Amount_Log,
        "Is_High_Value":               data["Is_High_Value"],
        "Failed_Ratio":                Failed_Ratio,
    }

    input_df     = pd.DataFrame([input_dict])[feature_cols]
    input_scaled = scaler.transform(input_df)
    prediction   = model.predict(input_scaled)[0]
    probability  = float(model.predict_proba(input_scaled)[0][1])

    if probability >= 0.85:
        risk, action = "P1 - CRITICAL", "Block Transaction"
    elif probability >= 0.60:
        risk, action = "P2 - HIGH", "Manual Review"
    elif probability >= 0.40:
        risk, action = "P3 - MEDIUM", "OTP Verification"
    else:
        risk, action = "P4 - LOW", "Auto Approve"

    return jsonify({
        "prediction": int(prediction),
        "probability": round(probability * 100, 2),
        "risk": risk,
        "action": action
    })

if __name__ == "__main__":
    app.run(debug=True)
