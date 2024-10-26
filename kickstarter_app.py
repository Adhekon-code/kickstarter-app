
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Load the dataset (ensure the CSV file is in the same directory as this script)
@st.cache_data
def load_data():
    data = pd.read_csv('kickstarter_2016.csv')
    return data

# Main function to run the app
def main():
    st.title("Kickstarter Campaign Success Prediction App")

    # Load the data
    data = load_data()

    # Display a message about the data
    st.write("In this project, I developed a predictive model to determine the success of Kickstarter campaigns from the year 2016. The primary goal was to create and evaluate machine learning models that classify whether a campaign will be successful based on specific features. This report outlines the steps taken to prepare the dataset, engineer features, train models, and analyze feature impact using backward elimination. ")

    # Show the first few rows of the dataset
    st.write("Creating two additional features: the duration of the campaign in days, and the length of the project name in words.Understanding the first few rows of the dataset:")
    st.write(data.head())

if __name__ == '__main__':
    main()


def main():
    st.subheader("Kickstarter Campaign Success Prediction")

    # Load the data
    data = load_data()

    # Display all the column names
    st.subheader("Understanding columns in the dataset:")
    st.write(data.columns)

if __name__ == '__main__':
    main()


# 1. Feature creation
def create_features(data):
    # Create two additional features: the duration of the campaign in days, and the length of the project name in words.
    data['Success'] = np.where(data['Pledged'] >= data['Goal'], 1, 0)
    data['Log_Goal'] = np.log(data['Goal'] + 1)

    # columns 'Launched' and 'Deadline'
    data['Launched'] = pd.to_datetime(data['Launched'])
    data['Deadline'] = pd.to_datetime(data['Deadline'])
    data['Duration_Days'] = (data['Deadline'] - data['Launched']).dt.days
    
    # Project name length (number of words)
    data['Name_Length'] = data['Name'].apply(lambda x: len(str(x).split()))
    
    # Selecting the relevant features
    features = ['Log_Goal', 'Duration_Days', 'Name_Length']
    target = 'Success'

    # Dropping rows with missing values
    data = data.dropna(subset=features + [target])
    
    return data, features, target


# Load the dataset
df = pd.read_csv('kickstarter_2016.csv')

# Create the target variable 'success' (1 for 'successful', 0 for other states)
df['success'] = df['State'].apply(lambda x: 1 if x == 'successful' else 0)

# Convert 'Launched' and 'Deadline' to datetime and calculate campaign duration in days
df['Launched'] = pd.to_datetime(df['Launched'])
df['Deadline'] = pd.to_datetime(df['Deadline'])
df['campaign_duration'] = (df['Deadline'] - df['Launched']).dt.days

# Calculate the length of the project name in words
df['name_length'] = df['Name'].apply(lambda x: len(str(x).split()))

# Display table with Streamlit
st.subheader("Kickstarter Project Analysis:Creating target variable")

st.subheader("Preview of the dataset with new features:")
st.dataframe(df[['Name', 'success', 'campaign_duration', 'name_length']].head())

# Optionally, if you want to show specific statistics or insights:
st.subheader("Basic Statistics of the new features:Optional")
st.write(df[['success', 'campaign_duration', 'name_length']].describe())


# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('kickstarter_2016.csv')
    return data

# Feature engineering
def create_features(data):
    # Adjusting for correct column names
    data['Success'] = np.where(data['Pledged'] >= data['Goal'], 1, 0)
    data['Log_Goal'] = np.log(data['Goal'] + 1)

    # Use the correct columns 'Launched' and 'Deadline'
    data['Launched'] = pd.to_datetime(data['Launched'])
    data['Deadline'] = pd.to_datetime(data['Deadline'])
    data['Duration_Days'] = (data['Deadline'] - data['Launched']).dt.days
    
    # Project name length (number of words)
    data['Name_Length'] = data['Name'].apply(lambda x: len(str(x).split()))
    
    # Selecting the relevant features
    features = ['Log_Goal', 'Duration_Days', 'Name_Length']
    target = 'Success'

    # Dropping rows with missing values
    data = data.dropna(subset=features + [target])
    
    return data, features, target

# Preprocessing pipeline
def preprocess_data(data, features, target):
    # Splitting the data into training and test sets
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing pipeline (scaling)
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())  # Standardize numeric features
    ])
    
    # Applying the preprocessing pipeline to the training data
    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)

    return X_train, X_test, y_train, y_test

# Main function to run the app
def main():
    st.subheader("Kickstarter Campaign Success Prediction: Model creation and evaluation!")

    # Load the data
    data = load_data()

    # Perform feature engineering
    data, features, target = create_features(data)

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(data, features, target)

    # Display the transformed features
    st.write(" Preprocessed Training Data (First 5 Rows")
    st.write(X_train[:5])

if __name__ == '__main__':
    main()

# Model creation and evaluation
def model_evaluation(X_train, y_train, classifier):
    # Create classifier based on user input
    if classifier == 'Logistic Regression':
        model = LogisticRegression()
    elif classifier == 'Random Forest':
        model = RandomForestClassifier(max_samples=0.1) 
    else:
        model = GradientBoostingClassifier()

    # Performing 5-fold cross-validation
    y_pred = cross_val_predict(model, X_train, y_train, cv=5)

    # Compute evaluation metrics
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)

    return accuracy, precision, recall, f1

def main():
    st.subheader(" Kickstarter Campaign Success Prediction:Feature impact analysis!")

    # Load and preprocess data
    data = load_data()
    data, features, target = create_features(data)
    X_train, X_test, y_train, y_test = preprocess_data(data, features, target)

    # Model selection
    classifier = st.selectbox("Choose Classifier", ("Logistic Regression", "Random Forest", "Gradient Boosting"))

    # Evaluating the model
    accuracy, precision, recall, f1 = model_evaluation(X_train, y_train, classifier)

    # Display results
    st.write(f"Classifier: {classifier}")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

if __name__ == '__main__':
    main()


def feature_impact_analysis(X_train, y_train, selected_features, classifier):
    if classifier == 'Logistic Regression':
        model = LogisticRegression()
    elif classifier == 'Random Forest':
        model = RandomForestClassifier(max_samples=0.1)
    else:
        model = GradientBoostingClassifier()

    current_features = selected_features.copy()
    st.write(f"Starting with features: {current_features}")

    while len(current_features) > 1:
        model.fit(X_train[current_features], y_train)  # Use column names
        accuracy = cross_val_score(model, X_train[current_features], y_train, cv=5).mean()
        st.write(f"Current Features: {current_features}, Accuracy: {accuracy:.2f}")
        
        feature_impact = {}
        for feature in current_features:
            reduced_features = [f for f in current_features if f != feature]
            reduced_accuracy = cross_val_score(model, X_train[reduced_features], y_train, cv=5).mean()
            feature_impact[feature] = reduced_accuracy
        
        worst_feature = min(feature_impact, key=feature_impact.get)
        st.write(f"Removing feature: {worst_feature} (Accuracy drop: {accuracy - feature_impact[worst_feature]:.2f})")
        current_features.remove(worst_feature)
    
        return current_features

    

