def show():

    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import joblib

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix

    from background import apply_background

    # --------------------------------------------------
    # PAGE CONFIG
    # --------------------------------------------------

    st.set_page_config(page_title="ML Testing Studio", layout="wide")
    apply_background()

    st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        width: 320px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Machine Learning Testing Studio")

    # --------------------------------------------------
    # CREATE TABS
    # --------------------------------------------------

    tab1, tab2, tab3 = st.tabs([
        "Use Trained Model (.joblib)",
        "Batch Prediction",
        "Train From Dataset"
    ])

    # ==================================================
    # TAB 1 → JOBLIB MANUAL PREDICTION
    # ==================================================

    with tab1:

        st.header("Use Trained Model")

        model_file = st.file_uploader(
            "Upload Trained Model (.joblib)",
            type=["joblib"],
            key="joblib_model"
        )

        if model_file:

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

            st.subheader("Model Performance")

            if task == "Regression":
                c1, c2 = st.columns(2)

                if rmse is not None:
                    c1.metric("RMSE", round(rmse, 3))

                if r2 is not None:
                    c2.metric("R² Score", round(r2, 3))

            else:
                if accuracy is not None:
                    st.metric("Accuracy", round(accuracy, 3))

            st.subheader("Manual Prediction")

            user_input = {}

            for col in features:
                user_input[col] = st.number_input(f"Enter {col}")

            if st.button("Predict", key="manual_predict"):

                input_df = pd.DataFrame([user_input])

                input_scaled = scaler.transform(input_df)

                pred = model.predict(input_scaled)

                st.success(f"Predicted {target}: {pred[0]}")

    # ==================================================
    # TAB 2 → BATCH PREDICTION
    # ==================================================

    with tab2:

        st.header("Batch Prediction")

        model_file = st.file_uploader(
            "Upload Trained Model (.joblib)",
            type=["joblib"],
            key="batch_model"
        )

        data_file = st.file_uploader(
            "Upload Dataset for Prediction",
            type=["csv", "xlsx"],
            key="batch_data"
        )

        if model_file and data_file:

            model_data = joblib.load(model_file)

            model = model_data["model"]
            scaler = model_data["scaler"]
            features = model_data["features"]
            target = model_data["target"]

            if data_file.name.endswith(".csv"):
                df = pd.read_csv(data_file)
            else:
                df = pd.read_excel(data_file)

            st.subheader("Dataset Preview")
            st.dataframe(df.head())

            if not all(col in df.columns for col in features):
                st.error("Dataset does not contain required features.")
            else:

                X = df[features]

                X_scaled = scaler.transform(X)

                preds = model.predict(X_scaled)

                df["Prediction"] = preds

                st.success("Predictions generated")

                st.dataframe(df.head())

                csv = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download predictions.csv",
                    csv,
                    "predictions.csv",
                    "text/csv"
                )

    # ==================================================
    # TAB 3 → YOUR ORIGINAL TRAINING PIPELINE
    # ==================================================

    with tab3:

        st.header("Train Model From Dataset")

        file = st.file_uploader(
            "Upload Dataset (CSV or XLSX)",
            type=["csv", "xlsx"],
            key="train_dataset"
        )

        if not file:
            st.info("Upload dataset to begin training")
            st.stop()

        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

        df.columns = df.columns.str.strip()

        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        num_cols = df.select_dtypes(include="number").columns.tolist()

        if len(num_cols) < 2:
            st.error("Dataset must contain at least 2 numeric columns.")
            st.stop()

        # SIDEBAR CONFIG
        st.sidebar.header("Model Configuration")

        task = st.sidebar.selectbox(
            "Select ML Task",
            ["Regression", "Classification"]
        )

        target = st.sidebar.selectbox(
            "Select Target Column",
            num_cols
        )

        features = st.sidebar.multiselect(
            "Select Input Features",
            [c for c in num_cols if c != target]
        )

        REGRESSION_MODELS = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(random_state=42),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
            "SVR": SVR()
        }

        CLASSIFICATION_MODELS = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest Classifier": RandomForestClassifier(random_state=42),
            "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
            "KNN": KNeighborsClassifier()
        }

        if task == "Regression":
            algo_name = st.sidebar.selectbox(
                "Select Regression Algorithm",
                list(REGRESSION_MODELS.keys())
            )
            model = REGRESSION_MODELS[algo_name]

        else:
            algo_name = st.sidebar.selectbox(
                "Select Classification Algorithm",
                list(CLASSIFICATION_MODELS.keys())
            )
            model = CLASSIFICATION_MODELS[algo_name]

        train_button = st.sidebar.button(
            "Train Model",
            disabled=(len(features) == 0)
        )

        if train_button:

            X = df[features]
            y = df[target]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            if task == "Classification":
                le = LabelEncoder()
                y = le.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            st.success("Model trained successfully")

            if task == "Regression":

                rmse = np.sqrt(mean_squared_error(y_test, preds))
                r2 = r2_score(y_test, preds)

                c1, c2 = st.columns(2)

                c1.metric("RMSE", round(rmse, 3))
                c2.metric("R² Score", round(r2, 3))

            else:

                acc = accuracy_score(y_test, preds)

                st.metric("Accuracy", round(acc, 3))
