import pandas as pd
import joblib

# Paths to the saved model
MODEL_PATH = "models/gradient_boost_model.joblib"

# New Customer Data
new_customer_data = pd.DataFrame({
        "Age": [46],
        "Gender": ["Male"],
        "Tenure": [50],
        'Usage Frequency': [22],
        'Support Calls': [4],
        'Payment Delay': [0], 
        'Subscription Type': ['Basic'],
        'Contract Length': ['Annual'], 
        'Total Spend': [683.9], 
        'Last Interaction': [12],
    })

def load_model():
    """Load the trained model"""
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
        return model
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

def predict(new_data):
    """Preprocess the data and make predictions."""
    model= load_model()
    predictions = model.predict(new_data) 
    return predictions

if __name__ == "__main__":
    results = predict(new_customer_data)
    print("Prediction Results:")
    print("Churn" if results[0] else "No Churn") 