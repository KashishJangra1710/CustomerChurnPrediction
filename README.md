# Customer Churn Prediction 🚀

## **Overview**
Customer churn prediction is a project aimed at identifying customers who are likely to leave a service or product subscription. This predictive analysis can help businesses take proactive measures to retain valuable customers, ultimately improving customer satisfaction and revenue.

--- 

## **Objective**
The primary objective of this project is to build a machine learning model that accurately predicts whether a customer will churn or not, based on historical data.

---

## **Dataset**
The dataset used for this project contains the following key features:
- **Demographic Information**: Gender, Age.
- **Account Information**: Tenure, Subscription Type, Contract Length
- **Behavioral Information**: Total Spend, Last Interaction, Usage Frequency, Support calls, Payment delay
- **Target Variable**: `Churn` (binary: 0 - No churn, 1 - Churn).

Source: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset

---

## **Technologies Used**
- **Programming Language**: Python 🐍
- **Libraries/Tools**:
  - Pandas, NumPy for data manipulation.
  - Matplotlib for visualization.
  - Scikit-learn for model building and evaluation.
  - XGBoost for advanced gradient boosting models.

---

## **Workflow**
1. **Data Understanding and Exploration**  
   - Explored the dataset (EDA.ipynb file) and checked for missing values and duplicates in the data.
   
2. **Exploratory Data Analysis (EDA)**  
   - Performed univariate analysis by plotting distribution of numerical features using histplot and categorical features using pie charts.
   - Performed bivariate analysis by observing "Churn" rates with respect to features.
   - Showed heatmap of correlations between different numerical features.

3. **Data Preprocessing**  
   - Encoded categorical features - Gender, Subscription Type, Contract Length using OneHotEncoder .
   - Scaled numerical features using StandardScaler.
   - Created a preprocessor for performing above tasks.

4. **Feature Engineering**  
   - Selected the most relevant features for the model.

5. **Model Building**  
   - Trained multiple classification models, including:
     - Logistic Regression
     - Random Forest
     - Gradient Boosting (and XGboost) 
   - Selected the best-performing model based on evaluation metrics (accuracy score).

6. **Model Evaluation**  
   - Evaluated model using different evaluation metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

---

## **Folder Structure**
- raw data and cleaned data are in data folder.
- notebooks for exploratory data analysis (EDA.ipynb) and different classification models (Model.ipynb) are in notebooks folder.
- In src folder, there are 3 python files- for data preprocessing (preprocess.py), training model and evaluation (train.py) and prediction (predict.py)
- Selected model (Gradient Boosting Classifier) is saved in models folder (model.joblib).
- Evaluation metrics for choosen model are in results folder (metrics.txt).