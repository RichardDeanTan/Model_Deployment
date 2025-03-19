import streamlit as st
import pandas as pd
import joblib

def load_model():
    """Load the trained Random Forest model"""
    return joblib.load('rf_model.pkl')

def main():
    # Load the model
    model = load_model()

    # Title of the app
    st.title("Iris Species Prediction")

    # Add a description
    st.write("""
    This app predicts the Iris species based on sepal and petal measurements using a Random Forest Classifier.
    Enter the measurements below to get a prediction!
    """)

    # Create input fields for the features
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.5)

    # Create a button to make prediction
    if st.button("Predict"):
        # Prepare the input data
        input_data = pd.DataFrame({
            'sepal length (cm)': [sepal_length],
            'sepal width (cm)': [sepal_width],
            'petal length (cm)': [petal_length],
            'petal width (cm)': [petal_width]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Map prediction to species name
        species = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
        predicted_species = species[prediction]
        
        # Display the result
        st.success(f"The predicted Iris species is: **{predicted_species}**")

    # Add some additional information
    st.write("""
    ### About the Model
    - Model Type: Random Forest Classifier
    - Features Used: Sepal Length, Sepal Width, Petal Length, Petal Width
    - Trained on the Iris dataset
    """)

if __name__ == "__main__":
    main()