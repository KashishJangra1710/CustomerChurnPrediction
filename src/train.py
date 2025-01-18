import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Function for Model Training
def train(trainX, trainY):    
    """Transform columns, train a random forest model using pipeline and save the model"""
    numeric_features = list(trainX.select_dtypes(include=['number']).columns)
    categorical_features = list(trainX.select_dtypes(include=['object','category']).columns) 
    # Transforming columns
    preprocessor = ColumnTransformer(
        transformers = [
            ('numeric', StandardScaler(), numeric_features),
            ('categ', OneHotEncoder(), categorical_features)
        ]
    )
    # Gradient Boost Classifier
    gradient_boost_classifier = GradientBoostingClassifier(max_depth = 10, random_state=42, verbose=1)
    # Pipeline for preprocessing and model
    gradient_boost_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', gradient_boost_classifier)
    ]) 
    # Training model
    print("Training started...")
    gradient_boost_pipeline.fit(trainX, trainY)
    print("Training done...")
    # Saving the model
    pipeline_path = "models/gradient_boost_model.joblib"
    joblib.dump(gradient_boost_pipeline, pipeline_path)
    print(f"Pipeline dumped successfully to {pipeline_path}") 
    return gradient_boost_pipeline

# Function for Model Evaluation
def evaluate(testX, testY, model):
    """Predict using the trained and saved model, calculate evaluation metrics and save them"""
    predY = model.predict(testX)
    accuracy = accuracy_score(testY, predY)
    class_report = classification_report(testY, predY) 
    conf_matrix = confusion_matrix(testY, predY) 
    with open("results/metrics.txt", 'w') as file:
        file.write(f"Accuracy Score: {accuracy}\n")
        file.write(f"Classification Report:\n {class_report}\n")
        file.write(f"Confusion Matrix:\n {conf_matrix}") 
    print(f"Model Accuracy: {accuracy}") 

if __name__=='__main__':
    # Loading Cleaned data
    data_path = 'data/cleaned_data.csv'
    data = pd.read_csv(data_path)
    # Extracting feature and target
    features = data.drop("Churn", axis=1)
    target = data["Churn"]
    # Train and Test data
    trainX, testX, trainY, testY = train_test_split(features, target, test_size=0.2, random_state=64)
    # Model training and evaluation
    model = train(trainX, trainY)
    evaluate(testX, testY, model) 