from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load Naive Bayes model
model = joblib.load("naive_bayes_model.joblib")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari form
        data = [float(x) for x in request.form.values()]
        input_data = np.array([data])

        # Prediksi menggunakan model
        prediction = model.predict(input_data)
        result = "Positive" if prediction[0] == 1 else "Negative"

        # Render hasil ke template
        return render_template('index.html', prediction_text=f'Diabetes Prediction: {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")


if __name__ == '__main__':
    app.run(debug=True)
