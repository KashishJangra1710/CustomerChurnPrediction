import pandas as pd 
import os

def load_data(path:str):
    """Loading original data"""
    try:
        data = pd.read_csv(path)
        print("Data loaded successfully")
        return data
    except Exception as e:
        print(f"Exception occured: {e}")
        exit(1)

def clean_data(path:str):
    """Cleaning data by removing null values and unnecessary columns, and changing column datatypes"""
    data = load_data(path)
    working_data = data.drop(["Unnamed: 0","CustomerID"], axis=1)
    working_data = working_data.dropna().reset_index(drop=True)
    columns_float_to_int = ['Age', 'Tenure', 'Usage Frequency','Support Calls', 'Payment Delay', 'Last Interaction', 'Churn']
    working_data[columns_float_to_int] = working_data[columns_float_to_int].astype('int')
    return working_data

def save_cleaned_data(data_path:str, save_path:str):
    "Saving cleaned data"
    try:
        if os.path.exists(save_path):
            print(f"{save_path} file already exists.")
        else:
            cleaned_data = clean_data(data_path)
            cleaned_data.to_csv(save_path, index=False)
            print(f"Cleaned data saved successfully to {save_path}")
    except Exception as e:
        print(f"Exception Occured: {e}")
        exit(1)

if __name__=='__main__':
    data_path = "data/churn_data.csv"
    save_path = "data/cleaned_data.csv"
    save_cleaned_data(data_path, save_path) 