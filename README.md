import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset (Replace 'road_accidents.csv' with your actual dataset file)
df = pd.read_csv("road_accidents.csv")

# Display first few rows of the dataset
print("Dataset Preview:")
print(df.head())

# Selecting relevant columns
columns = ["Accident_Severity", "Weather_Conditions", "Road_Type", "Speed_Limit", 
           "Number_of_Vehicles_Involved", "Lighting_Conditions", "Day_of_Week", "Time_of_Accident"]
df = df[columns]

# Convert categorical variables into numerical using one-hot encoding
df = pd.get_dummies(df, columns=["Weather_Conditions", "Road_Type", "Lighting_Conditions", "Day_of_Week"], drop_first=True)

# Splitting data into X (independent variables) and y (dependent variable)
X = df.drop("Accident_Severity", axis=1)
y = df["Accident_Severity"]

# Splitting into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Model Evaluation - Mean Squared Error: {mse}")

# Save the trained model for future use
joblib.dump(model, "accident_severity_model.pkl")
print("Model saved as accident_severity_model.pkl")

# Load saved model for prediction
loaded_model = joblib.load("accident_severity_model.pkl")

# Create a hypothetical accident scenario for prediction
hypothetical_data = pd.DataFrame({
    "Speed_Limit": [60],
    "Number_of_Vehicles_Involved": [3],
    "Time_of_Accident": [15],  # Assuming 15:00 (3 PM)
    "Weather_Conditions_Clear": [1],  
    "Weather_Conditions_Rainy": [0],  
    "Road_Type_Major": [1],  
    "Lighting_Conditions_Daylight": [1],  
    "Lighting_Conditions_Night": [0],  
    "Day_of_Week_Monday": [1],  
    "Day_of_Week_Friday": [0]
})

# Predict accident severity
predicted_severity = loaded_model.predict(hypothetical_data)
print(f"Predicted Accident Severity: {predicted_severity[0]}")

# Visualizing the predictions
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test, y=y_pred, color="blue", alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="dashed")
plt.xlabel("Actual Severity")
plt.ylabel("Predicted Severity")
plt.title("Actual vs Predicted Accident Severity")
plt.show()
