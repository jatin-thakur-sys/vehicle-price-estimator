import pandas as pd
import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained pipeline (model with preprocessing)
model = pickle.load(open("car_price_prediction.pkl", "rb"))

# Load dataset to get unique values for dropdowns
df = pd.read_csv("car_price_dataset.csv")
companies = df['Brand'].unique().tolist()
models = df['Model'].unique().tolist()
years = sorted(df['Year'].unique().tolist(), reverse=True)
fuel_types = df['Fuel_Type'].unique().tolist()
transmissions = df['Transmission'].unique().tolist()
owners = df['Owner_Count'].unique().tolist()

@app.route("/")
def home():
    return render_template("index.html", companies=companies, models=models, year=years, 
                           fuel=fuel_types, transmission=transmissions, owner=owners)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("Received Data:", data)  # Debugging statement

        # Ensure column names match model expectations
        formatted_data = pd.DataFrame([{
            "Brand": data["Company"],
            "Model": data["Model"],
            "Year": data["Year"],
            "Fuel_Type": data["Fuel_Type"],
            "Engine_Size": data["Engine_Size"],
            "Transmission": data["Transmission"],
            "Doors": data["Doors"],
            "Owner_Count": data["Owner"],
            "Mileage": data["Mileage"]
        }])

        # Predict using the pipeline (automatically applies OneHotEncoding & Scaling)
        predicted_price = model.predict(formatted_data)[0]

        return jsonify({"predicted_price": round(predicted_price, 2)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
