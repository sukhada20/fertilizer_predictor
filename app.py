import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and preprocess dataset (replace with your dataset path)
data = pd.read_csv('FertilizerPrediction.csv')
X = data[['Temparature', 'Humidity ', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
y = data['Fertilizer Name']
X = pd.get_dummies(X, columns=['Soil Type', 'Crop Type'], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# Fertilizer prediction function
def predict_fertilizer(temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous):
    input_data = pd.DataFrame([[temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous]],
                               columns=['Temparature', 'Humidity ', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous'])
    input_data = pd.get_dummies(input_data, columns=['Soil Type', 'Crop Type'], drop_first=True)
    missing_cols = set(X_train.columns) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0
    input_data = input_data[X_train.columns]
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app
st.title("Fertilizer Recommendation System")
st.write("Get personalized fertilizer recommendations based on your soil and crop information.")

# Input fields
soil_type = st.selectbox("Select Soil Type:", ["Sandy", "Loamy", "Clayey", "Red", "Black"])
crop_type = st.selectbox("Select Crop Type:", ["Maize", "Sugarcane", "Cotton", "Tobacco", "Paddy", "Barley", "Wheat", "Millets", "Oil Seeds", "Pulses", "Ground Nuts"])
temperature = st.number_input("Temperature (°C):", min_value=-10, max_value=50, value=25, step=1)
humidity = st.number_input("Humidity (%):", min_value=0, max_value=100, value=50, step=1)
moisture = st.number_input("Moisture (%):", min_value=0, max_value=100, value=30, step=1)
nitrogen = st.number_input("Nitrogen (ppm):", min_value=0, max_value=100, value=50, step=1)
potassium = st.number_input("Potassium (ppm):", min_value=0, max_value=100, value=40, step=1)
phosphorous = st.number_input("Phosphorous (ppm):", min_value=0, max_value=100, value=30, step=1)

# Recommendation button
if st.button("Get Recommendation"):
    recommendation = predict_fertilizer(temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous)
    st.success(f"The recommended fertilizer is: {recommendation}")

# Display model accuracy
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
