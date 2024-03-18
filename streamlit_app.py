import streamlit as st
import pickle
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.train_model import train_model
from steps.evaluation import evaluate_model

# Define our input data class
class InputData(BaseModel):
    YearMade: int
    MachineHoursCurrentMeter: int
    UsageBand: str
    ProductSize: str
    fiProductClassDesc: str
    state: str
    ProductGroup: str
    ProductGroupDesc: str
    Drive_System: str
    Enclosure: str

# Define default values for input features
defaults = {
    'YearMade': 2010,
    'MachineHoursCurrentMeter': 5000,
    'UsageBand': 'High',
    'ProductSize': 'Large',
    'fiProductClassDesc': 'Construction',
    'state': 'CA',
    'ProductGroup': 'Wheel Loader',
    'ProductGroupDesc': 'Wheel Loader',
    'Drive_System': 'Wheel',
    'Enclosure': 'Open'
}

# Group related inputs
groups = {
    'Basic Information': ['YearMade', 'MachineHoursCurrentMeter', 'UsageBand', 'ProductSize', 'fiProductClassDesc', 'state'],
    'Product Details': ['ProductGroup', 'ProductGroupDesc', 'Drive_System', 'Enclosure']
}

# Streamlit App
def main():
    st.title("Machine Learning Model Input")

    # Ingest data
    data_path = st.text_input("Enter data path", value="data.csv")
    if st.button("Ingest Data"):
        ingest_df.run(parameters={"data_path": data_path})
        st.success("Data ingestion complete!")

    # Clean data
    if st.button("Clean Data"):
        X_train, X_test, y_train, y_test = clean_df.run()
        st.success("Data cleaning complete!")

    # Train model
    if st.button("Train Model"):
        trained_model = train_model.run(parameters={"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test})
        st.success("Model training complete!")

    # Evaluate model
    if st.button("Evaluate Model"):
        r2, rmse = evaluate_model.run(parameters={"model": trained_model, "X_test": X_test, "y_test": y_test})
        st.success(f"Model evaluation complete! R2 score: {r2}, RMSE: {rmse}")

    # User input for prediction
    st.header("Enter Data for Prediction")

    # Create input form
    input_data = {}
    for group_name, features in groups.items():
        with st.expander(group_name):
            for feature in features:
                if feature == 'YearMade':
                    input_data[feature] = st.slider(f"{feature}", min_value=1950, max_value=2023, value=defaults[feature], step=1)
                elif feature == 'MachineHoursCurrentMeter':
                    input_data[feature] = st.number_input(f"{feature}", min_value=0, max_value=100000, value=defaults[feature], step=1)
                elif feature == 'UsageBand':
                    input_data[feature] = st.selectbox(f"{feature}", ['High', 'Medium', 'Low'], index=0)
                elif feature == 'ProductSize':
                    input_data[feature] = st.selectbox(f"{feature}", ['Large', 'Medium', 'Small'], index=0)
                elif feature == 'fiProductClassDesc':
                    input_data[feature] = st.text_input(f"{feature}", value=defaults[feature])
                elif feature == 'state':
                    input_data[feature] = st.text_input(f"{feature}", value=defaults[feature])
                elif feature == 'ProductGroup':
                    input_data[feature] = st.selectbox(f"{feature}", ['Wheel Loader', 'Excavator', 'Tractor', 'Skid Steer', 'Backhoe'], index=0)
                elif feature == 'ProductGroupDesc':
                    input_data[feature] = st.selectbox(f"{feature}", ['Wheel Loader', 'Excavator', 'Tractor', 'Skid Steer', 'Backhoe'], index=0)
                elif feature == 'Drive_System':
                    input_data[feature] = st.selectbox(f"{feature}", ['Wheel', 'Track'], index=0)
                elif feature == 'Enclosure':
                    input_data[feature] = st.selectbox(f"{feature}", ['Open', 'Enclosed'], index=0)

    if st.button("Predict"):
        # Create InputData object with user input
        input_data_obj = InputData(**input_data)

        # Encode categorical features
        categorical_features = ['UsageBand', 'ProductSize', 'fiProductClassDesc', 'state', 'ProductGroup', 'ProductGroupDesc', 'Drive_System', 'Enclosure']
        encoded_data = []
        for feature in categorical_features:
            value = getattr(input_data_obj, feature)
            encoded_value = label_encoders[feature].transform([value])[0]
            encoded_data.append(encoded_value)

        # Prepare input data for prediction
        input_features = [input_data_obj.YearMade,
                          input_data_obj.MachineHoursCurrentMeter] + encoded_data

        # Use the loaded model to make predictions
        prediction = trained_model.predict([input_features])

        # Display the prediction
        st.header("Prediction Result")
        st.write(prediction)

if __name__ == "__main__":
    main()