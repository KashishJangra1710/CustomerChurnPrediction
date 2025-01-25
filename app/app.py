from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Paths to the saved model
MODEL_PATH = "models/gradient_boost_model.joblib"

def load_model():
    """Load the trained model"""
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
        return model
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

def predict_churn(new_data):
    """Preprocess the data and make predictions."""
    model= load_model()
    predictions = model.predict(new_data) 
    probability = model.predict_proba(new_data)
    return predictions, probability

@app.route('/')
def home():
    return render_template('home.html', churn=None, upload=None)

@app.route('/choose', methods=['POST'])
def choose():
    selected_method = request.form['method']
    if selected_method == 'upload':
        return render_template('home.html', churn=None, upload=1) 
    elif selected_method == 'form':
        return render_template('home.html', churn=None, upload=0)
    else:
        return "Invalid choice", 400

@app.route('/file_submit', methods=['POST'])
def file_submit():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file and file.filename.endswith('.csv'):
        data = pd.read_csv(file)
        return "File uploaded and processed successfully"

@app.route('/form_submit', methods=['POST'])
def form_submit():
    # Retrieve form values
    age = int(request.form.get('age'))
    gender = request.form.get('gender')
    tenure = int(request.form.get('tenure'))
    subscription_type = request.form.get('subscription')
    contract_length = request.form.get('contract')
    total_spend = float(request.form.get('total_spend'))
    last_interaction = int(request.form.get('last_interaction'))
    usage_frequency = int(request.form.get('usage_frequency'))
    support_calls = int(request.form.get('support_calls'))
    payment_delay = int(request.form.get('payment_delay'))

    # New Customer Data
    new_customer_data = pd.DataFrame({
            "Age": age,
            "Gender": gender,
            "Tenure": tenure,
            'Usage Frequency': usage_frequency,
            'Support Calls': support_calls,
            'Payment Delay': payment_delay, 
            'Subscription Type': subscription_type,
            'Contract Length': contract_length, 
            'Total Spend': total_spend, 
            'Last Interaction': last_interaction,
        }, index=[0]) 
    
    pred, prob = predict_churn(new_customer_data)
    prob = prob[0, 1] if pred else prob[0, 0]
    return render_template('home.html', churn=pred, probability=np.round(prob*100, 2)) 

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__=='__main__':
    app.run(debug=True) 