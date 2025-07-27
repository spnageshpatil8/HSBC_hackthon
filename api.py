from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("mlp_model.pkl")
scaler = joblib.load("scaler.pkl")

# Categorical encodings 
customer_map = {'C12345': 0} 
age_map = {'30-39': 1, '20-29': 0, '40-49': 2}  
gender_map = {'F': 0, 'M': 1}
merchant_map = {'M98765': 0}  
category_map = {'shopping': 0, 'travel': 1, 'entertainment': 2}  

@app.route('/')
def home():
    return render_template("form.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        step = float(request.form['step'])
        customer = customer_map.get(request.form['customer'], 0)
        age = age_map.get(request.form['age'], 0)
        gender = gender_map.get(request.form['gender'], 0)
        merchant = merchant_map.get(request.form['merchant'], 0)
        category = category_map.get(request.form['category'], 0)
        amount = float(request.form['amount'])

        # Create input vector
        input_data = np.array([[step, customer, age, gender, merchant, category, amount]])

        # Scale step and amount
        input_data[:, [0, 6]] = scaler.transform(input_data[:, [0, 6]])

        # Predict
        prediction = model.predict(input_data)[0]
        label = "Fraudulent Transaction" if prediction == 1 else "Legitimate Transaction"
        return render_template("form.html", prediction_text=label)

    except Exception as e:
        return render_template("form.html", prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
