def show():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px

    from background import apply_background

    # --------------------------------------------------
    # PAGE CONFIG
    # --------------------------------------------------
    st.set_page_config(page_title="Data Visualization Studio", layout="wide")
    apply_background()

    st.title("Data Visualization Studio")
    st.write("Upload any dataset and freely create interactive visualizations.")

    # --------------------------------------------------
    # DATA UPLOAD
    # --------------------------------------------------
    file = st.file_uploader(
        "Upload Dataset for Visualization (CSV or XLSX)",
        type=["csv", "xlsx"],
        key="visualization_file"
    )

    if not file:
        st.info("Upload a dataset to start visualizing")
        st.stop()

    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    df.columns = df.columns.str.strip()

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols = df.columns.tolist()

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # --------------------------------------------------
    # SIDEBAR CONTROLS
    # --------------------------------------------------
    st.sidebar.header("Visualization Controls")

    chart_type = st.sidebar.selectbox(
        "Select Chart Type",
        [
            "Bar Chart",
            "Line Chart",
            "Scatter Plot",
            "Histogram",
            "Box Plot",
            "Pie Chart",
            "Correlation Heatmap"
        ]
    )

    x_col = st.sidebar.selectbox("X Axis", ["-- Select --"] + all_cols)
    y_col = st.sidebar.selectbox("Y Axis", ["-- Select --"] + all_cols)

    color_col = st.sidebar.selectbox(
        "Color (Optional)",
        ["None"] + all_cols
    )

    agg_func = st.sidebar.selectbox(
        "Aggregation (for categorical X)",
        ["None", "Count", "Sum", "Mean"]
    )

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------
    def can_show_button(chart, x, y):
        if chart in ["Histogram", "Pie Chart", "Correlation Heatmap"]:
            return x != "-- Select --" or y != "-- Select --"
        return x != "-- Select --" and y != "-- Select --"

    # --------------------------------------------------
    # PLOT BUTTON (CONDITIONAL)
    # --------------------------------------------------
    plot_clicked = False

    if can_show_button(chart_type, x_col, y_col):
        plot_clicked = st.sidebar.button("Plot Chart")

    # --------------------------------------------------
    # CHART RENDERING (ONLY AFTER BUTTON CLICK)
    # --------------------------------------------------
    if plot_clicked:

        fig = None

        # ---------------- BAR ----------------
        if chart_type == "Bar Chart":

            if agg_func == "None":
                fig = px.bar(
                    df,
                    x=x_col,
                    y=None if y_col == "-- Select --" else y_col,
                    color=None if color_col == "None" else color_col
                )

            else:
                if agg_func == "Count":
                    temp = df.groupby(x_col).size().reset_index(name="Count")
                    fig = px.bar(temp, x=x_col, y="Count")

                elif agg_func in ["Sum", "Mean"]:
                    temp = df.groupby(x_col)[y_col].agg(
                        "sum" if agg_func == "Sum" else "mean"
                    ).reset_index()
                    fig = px.bar(temp, x=x_col, y=y_col)

        # ---------------- LINE ----------------
        elif chart_type == "Line Chart":
            fig = px.line(
                df,
                x=x_col,
                y=y_col,
                color=None if color_col == "None" else color_col
            )

        # ---------------- SCATTER ----------------
        elif chart_type == "Scatter Plot":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=None if color_col == "None" else color_col
            )

        # ---------------- HISTOGRAM ----------------
        elif chart_type == "Histogram":
            fig = px.histogram(
                df,
                x=x_col if x_col != "-- Select --" else y_col,
                color=None if color_col == "None" else color_col
            )

        # ---------------- BOX ----------------
        elif chart_type == "Box Plot":
            fig = px.box(
                df,
                x=x_col,
                y=y_col,
                color=None if color_col == "None" else color_col
            )

        # ---------------- PIE ----------------
        elif chart_type == "Pie Chart":
            if y_col == "-- Select --":
                temp = df[x_col].value_counts().reset_index()
                temp.columns = [x_col, "Count"]
                fig = px.pie(temp, names=x_col, values="Count")
            else:
                fig = px.pie(df, names=x_col, values=y_col)

        # ---------------- HEATMAP ----------------
        elif chart_type == "Correlation Heatmap":
            if len(num_cols) < 2:
                st.warning("Need at least two numeric columns.")
            else:
                corr = df[num_cols].corr()
                fig = px.imshow(
                    corr,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu_r"
                )

        # --------------------------------------------------
        # DISPLAY
        # --------------------------------------------------
        if fig:
            fig.update_layout(
                height=650,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)

