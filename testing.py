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
    from sklearn.metrics import (
        mean_squared_error,
        r2_score,
        accuracy_score,
        confusion_matrix
    )

    from background import apply_background

    # --------------------------------------------------
    # PAGE CONFIG
    # --------------------------------------------------

    st.set_page_config(page_title="ML Testing Studio", layout="wide")
    apply_background()

    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {
            width: 320px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Machine Learning Testing Studio")
    st.write("Train once, then make live manual predictions on a fresh dataset.")

    # --------------------------------------------------
    # JOBLIB MODEL UPLOAD
    # --------------------------------------------------

    model_file = st.file_uploader(
        "Upload Trained Model (.joblib)",
        type=["joblib"],
        key="model_upload"
    )

    # --------------------------------------------------
    # IF USER UPLOADS MODEL
    # --------------------------------------------------

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
        # SHOW MODEL METRICS
        # ----------------------------

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

        # ----------------------------
        # MANUAL PREDICTION
        # ----------------------------

        st.subheader("Manual Prediction")

        user_input = {}

        for col in features:
            user_input[col] = st.number_input(f"Enter {col}")

        input_df = pd.DataFrame([user_input])

        if st.button("Predict"):

            input_scaled = scaler.transform(input_df)

            pred = model.predict(input_scaled)

            st.success(f"Predicted {target}: {pred[0]}")

        st.stop()

    # --------------------------------------------------
    # SESSION STATE INIT
    # --------------------------------------------------

    if "trained" not in st.session_state:
        st.session_state.trained = False
        st.session_state.model = None
        st.session_state.scaler = None
        st.session_state.features = None
        st.session_state.target = None
        st.session_state.task = None
        st.session_state.label_encoder = None

    # --------------------------------------------------
    # DATASET UPLOAD (YOUR ORIGINAL FLOW)
    # --------------------------------------------------

    file = st.file_uploader(
        "Upload Dataset for Testing (CSV or XLSX)",
        type=["csv", "xlsx"],
        key="testing_file"
    )

    if not file:
        st.info("Upload a dataset to begin testing")
        st.stop()

    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

    df.columns = df.columns.str.strip()

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # --------------------------------------------------
    # BASIC CHECK
    # --------------------------------------------------

    num_cols = df.select_dtypes(include="number").columns.tolist()

    if len(num_cols) < 2:
        st.error("Dataset must contain at least 2 numeric columns.")
        st.stop()

    # --------------------------------------------------
    # SIDEBAR CONFIG
    # --------------------------------------------------

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

    # --------------------------------------------------
    # MODEL REGISTRY
    # --------------------------------------------------

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

    # --------------------------------------------------
    # TRAIN BUTTON
    # --------------------------------------------------

    train_button = st.sidebar.button(
        "Train Model",
        disabled=(len(features) == 0)
    )

    # --------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------

    if train_button:

        X = df[features]
        y = df[target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if task == "Classification":
            le = LabelEncoder()
            y = le.fit_transform(y)
            st.session_state.label_encoder = le

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

            fig, ax = plt.subplots(figsize=(7, 6))

            ax.scatter(y_test, preds, alpha=0.7)

            min_val = min(y_test.min(), preds.min())
            max_val = max(y_test.max(), preds.max())

            ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")

            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")

            st.pyplot(fig)

        else:

            acc = accuracy_score(y_test, preds)

            st.metric("Accuracy", round(acc, 3))

            cm = confusion_matrix(y_test, preds)

            fig, ax = plt.subplots()

            ax.imshow(cm)

            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha="center", va="center")

            st.pyplot(fig)

        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.features = features
        st.session_state.target = target
        st.session_state.task = task
        st.session_state.trained = True

    # --------------------------------------------------
    # MANUAL PREDICTION
    # --------------------------------------------------

    if st.session_state.trained:

        st.subheader("Manual Prediction")

        user_input = {}

        for col in st.session_state.features:

            user_input[col] = st.number_input(
                f"Enter {col}",
                value=float(df[col].mean())
            )

        input_df = pd.DataFrame([user_input])

        input_scaled = st.session_state.scaler.transform(input_df)

        pred = st.session_state.model.predict(input_scaled)

        if st.session_state.task == "Classification":
            pred = st.session_state.label_encoder.inverse_transform(pred)

        st.success(f"Predicted {st.session_state.target}: {pred[0]}")
