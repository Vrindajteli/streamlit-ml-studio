import streamlit as st
import pandas as pd
import joblib

def show():
    st.title("Model Testing")

    st.write("Upload a trained model (.joblib) OR upload a dataset to test.")

    model_file = st.file_uploader("Upload Model (.joblib)", type=["joblib"])
    dataset_file = st.file_uploader("Upload Dataset (.csv or .xlsx)", type=["csv","xlsx"])

    # ----------------------------
    # MODEL UPLOAD SECTION
    # ----------------------------
    if model_file is not None:

        model_data = joblib.load(model_file)

        model = model_data["model"]
        scaler = model_data["scaler"]
        features = model_data["features"]
        target = model_data["target"]
        task = model_data["task"]

        rmse = model_data.get("rmse")
        r2 = model_data.get("r2")
        accuracy = model_data.get("accuracy")

        st.success("Model Loaded Successfully")

        # ----------------------------
        # Display Model Metrics
        # ----------------------------
        st.subheader("Model Performance")

        if task == "Regression":

            col1, col2 = st.columns(2)

            if rmse is not None:
                col1.metric("RMSE", round(rmse, 4))

            if r2 is not None:
                col2.metric("R² Score", round(r2, 4))

        else:

            if accuracy is not None:
                st.metric("Accuracy", round(accuracy, 4))

        # ----------------------------
        # Manual Prediction
        # ----------------------------
        st.subheader("Manual Prediction")

        input_data = {}

        for feature in features:
            input_data[feature] = st.number_input(f"Enter {feature}")

        if st.button("Predict"):

            input_df = pd.DataFrame([input_data])

            if scaler is not None:
                input_scaled = scaler.transform(input_df)
            else:
                input_scaled = input_df

            prediction = model.predict(input_scaled)

            st.success(f"Predicted {target}: {prediction[0]}")

    # ----------------------------
    # DATASET UPLOAD SECTION
    # ----------------------------
    elif dataset_file is not None:

        # Detect file type
        if dataset_file.name.endswith(".csv"):
            df = pd.read_csv(dataset_file)

        elif dataset_file.name.endswith(".xlsx"):
            df = pd.read_excel(dataset_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        target_column = st.selectbox("Select Target Column", df.columns)

        X = df.drop(columns=[target_column])
        y = df[target_column]

        st.write("Features detected:", list(X.columns))

        if st.button("Run Prediction"):

            st.warning("Dataset uploaded but no trained model available to run predictions.")

            st.info("Please upload a trained .joblib model to make predictions.")
