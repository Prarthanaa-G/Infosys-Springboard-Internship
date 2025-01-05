import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("xgb_model.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected.")

        data = pd.read_csv(file)

        if "Time" in data.columns:
            data.drop(columns=["Time"], inplace=True)
        if "Class" in data.columns:
            data.drop(columns=["Class"], inplace=True)

        if "Amount_Interval" in data.columns:
            data.drop(columns=["Amount_Interval"], inplace=True)

        data = data.astype(float)

        predictions = model.predict(data)

        data["Prediction"] = predictions

        table_html = data.to_html(classes="table table-striped", index=False).replace("\n", "")

        return render_template("index.html", table_html=table_html)

    except Exception as e:
        return render_template("index.html", error=f"Error: {str(e)}")
    
    
    
if __name__ == "__main__":
    app.run(debug=True)
