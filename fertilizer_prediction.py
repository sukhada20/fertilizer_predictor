from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
# Edit this path to your dataset
data = pd.read_csv('/content/drive/path/FertilizerPrediction.csv')

# Define features and target variable
# Updated the column names based on the actual names in the dataset
X = data[['Temparature', 'Humidity ', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
y = data['Fertilizer Name']

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X, columns=['Soil Type', 'Crop Type'], drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Function to predict fertilizer
def predict_fertilizer(temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous):
    # Create a DataFrame for the input data
    input_data = pd.DataFrame([[temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous]],
                               columns=['Temparature', 'Humidity ', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous'])

    # Apply one-hot encoding to the categorical features
    input_data = pd.get_dummies(input_data, columns=['Soil Type', 'Crop Type'], drop_first=True)

    # Align the columns of the input data with the training data
    missing_cols = set(X_train.columns) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0
    input_data = input_data[X_train.columns]

    # Make prediction
    prediction = model.predict(input_data)
    return prediction[0]

# Example usage
# Replace the parameters with actual values
t = input('Enter temperature: ')
h = input('Enter humidity: ')
m = input('Enter moisture: ')
s = input('Enter soil type: ')
c = input('Enter crop type: ')
n = input('Enter nitrogen: ')
k = input('Enter potassium: ')
p = input('Enter phosphorus: ')
predicted_fertilizer = predict_fertilizer(t, h, m, s, c, n, k, p)
print(f'The recommended fertilizer is: {predicted_fertilizer}')
