import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Create a DataFrame
data = {
    'Age': [25, 30, 22, 35, 29, 40],
    'Income': [50000, 60000, 45000, 70000, 55000, 80000],
    'Purchased': [0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Features (X) and Labels (y)
X = df[['Age', 'Income']]
y = df['Purchased']

# Split the dataset into training and testing sets (80% training, 20% testing) Random-state is a constant of 42 in programming
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for scaling the numerical features
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Create a ColumnTransformer that applies the scaling to the numerical columns
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, ['Age', 'Income'])
])

# Create a pipeline that first preprocesses the data, then fits a Random Forest model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Fit the model to the training data
model_pipeline.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model_pipeline.predict(X_test)

# Evaluate the model's performance
print(classification_report(y_test, y_pred))
