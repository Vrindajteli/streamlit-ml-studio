def show():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import joblib
    import tempfile

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

    from background import apply_background

    # --------------------------------------------------
    # PAGE CONFIG
    # --------------------------------------------------
    st.set_page_config(page_title="ML Training Studio", layout="wide")
    apply_background()

    st.title("Machine Learning Training Studio")
    st.write("Train models, auto-detect the best algorithm, and export results.")

    # --------------------------------------------------
    # SESSION STATE INIT (MODEL ONLY)
    # --------------------------------------------------
    defaults = {
        "trained": False,
        "model": None,
        "scaler": None,
        "features": None,
        "target": None,
        "task": None,
        "rmse": None,
        "r2": None,
        "accuracy": None,
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,
        "label_encoder": None,
        "auto_model_name": None,
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

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

    # --------------------------------------------------
    # AUTO MODEL SELECTION
    # --------------------------------------------------
    def auto_select_model(task, X, y, test_size):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if task == "Classification":
            le = LabelEncoder()
            y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )

        models = REGRESSION_MODELS if task == "Regression" else CLASSIFICATION_MODELS
        scores = {}

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                score = (
                    np.sqrt(mean_squared_error(y_test, preds))
                    if task == "Regression"
                    else accuracy_score(y_test, preds)
                )

                scores[name] = score
            except:
                continue

        return min(scores, key=scores.get) if task == "Regression" else max(scores, key=scores.get)

    # --------------------------------------------------
    # DATA UPLOAD (ISOLATED)
    # --------------------------------------------------
    file = st.file_uploader(
        "Upload Dataset for Training (CSV or XLSX)",
        type=["csv", "xlsx"],
        key="training_file"   # 🔑 IMPORTANT
    )

    if not file:
        st.info("Upload a dataset to begin training")
        st.stop()

    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

    # 🔧 COLUMN SAFETY
    df.columns = df.columns.str.strip()

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # --------------------------------------------------
    # BASIC CHECK
    # --------------------------------------------------
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if len(num_cols) < 2:
        st.error("Dataset must contain at least two numeric columns.")
        st.stop()

    # --------------------------------------------------
    # SIDEBAR CONFIG
    # --------------------------------------------------
    st.sidebar.header("Training Configuration")

    task = st.sidebar.selectbox("Select ML Task", ["Regression", "Classification"])
    target = st.sidebar.selectbox("Select Target Column", num_cols)

    features = st.sidebar.multiselect(
        "Select Input Features",
        [c for c in num_cols if c != target],
        default=[c for c in num_cols if c != target][:2]
    )

    split_option = st.sidebar.selectbox(
        "Train / Test Split",
        ["80% Train / 20% Test", "70% Train / 30% Test", "60% Train / 40% Test"]
    )

    split_map = {
        "80% Train / 20% Test": 0.2,
        "70% Train / 30% Test": 0.3,
        "60% Train / 40% Test": 0.4
    }

    test_size = split_map[split_option]

    auto_detect = st.sidebar.checkbox("Auto-detect Best Model", value=True)

    # --------------------------------------------------
    # ALGORITHM SELECTION
    # --------------------------------------------------
    if auto_detect and features:
        st.session_state.auto_model_name = auto_select_model(
            task, df[features], df[target], test_size
        )

    model_list = list(
        REGRESSION_MODELS.keys()
        if task == "Regression"
        else CLASSIFICATION_MODELS.keys()
    )

    default_index = (
        model_list.index(st.session_state.auto_model_name)
        if auto_detect and st.session_state.auto_model_name in model_list
        else 0
    )

    algo_name = st.sidebar.selectbox("Select Algorithm", model_list, index=default_index)

    model = (
        REGRESSION_MODELS[algo_name]
        if task == "Regression"
        else CLASSIFICATION_MODELS[algo_name]
    )

    # --------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------
    if st.sidebar.button("Train Model"):

        X = df[features]
        y = df[target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if task == "Classification":
            le = LabelEncoder()
            y = le.fit_transform(y)
            st.session_state.label_encoder = le

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        st.session_state.update({
            "model": model,
            "scaler": scaler,
            "features": features,
            "target": target,
            "task": task,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "trained": True
        })

        if task == "Regression":
            st.session_state.rmse = np.sqrt(mean_squared_error(y_test, preds))
            st.session_state.r2 = r2_score(y_test, preds)
        else:
            st.session_state.accuracy = accuracy_score(y_test, preds)

        st.success(f"Model trained successfully using {algo_name}")

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------
    if st.session_state.trained:

        st.subheader("Model Performance")

        if st.session_state.task == "Regression":
            c1, c2 = st.columns(2)
            c1.metric("RMSE", round(st.session_state.rmse, 4))
            c2.metric("R² Score", round(st.session_state.r2, 4))
        else:
            st.metric("Accuracy", round(st.session_state.accuracy, 4))

    # --------------------------------------------------
    # DOWNLOADS
    # --------------------------------------------------
    if st.session_state.trained:

        st.subheader("Download Outputs")

        X_train_df = pd.DataFrame(st.session_state.X_train, columns=st.session_state.features)
        X_test_df = pd.DataFrame(st.session_state.X_test, columns=st.session_state.features)

        st.download_button("Download Training Data", X_train_df.to_csv(index=False), "training_data.csv")
        st.download_button("Download Testing Data", X_test_df.to_csv(index=False), "testing_data.csv")

        model_path = tempfile.NamedTemporaryFile(delete=False, suffix=".joblib").name
        joblib.dump(
            {
                "model": st.session_state.model,
                "scaler": st.session_state.scaler,
                "features": st.session_state.features,
                "target": st.session_state.target,
                "task": st.session_state.task
            },
            model_path
        )

        with open(model_path, "rb") as f:
            st.download_button("Download Trained Model", f, "trained_model.joblib")
