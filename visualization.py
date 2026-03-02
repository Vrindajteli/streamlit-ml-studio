def show():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import tempfile

    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px

    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image
    )
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors

    from background import apply_background

    # --------------------------------------------------
    # PAGE CONFIG
    # --------------------------------------------------
    st.set_page_config(page_title="EDA Studio", layout="wide")
    apply_background()

    st.title("📊 Exploratory Data Analysis")

    # --------------------------------------------------
    # FILE UPLOAD
    # --------------------------------------------------
    file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if not file:
        st.info("📂 Upload a dataset to continue")
        st.stop()

    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # --------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------
    st.sidebar.title("EDA Options")

    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        [
            "Dataset Health",
            "Univariate Analysis",
            "Bivariate Analysis",
            "Multivariate Analysis"
        ]
    )

    generate_pdf = st.sidebar.button("Generate PDF Report")

    # --------------------------------------------------
    # DATA PREVIEW
    # --------------------------------------------------
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))

    # --------------------------------------------------
    # UTILITIES
    # --------------------------------------------------
    def format_df_for_pdf(dataframe, max_rows=10):
        df_copy = dataframe.head(max_rows).copy()
        for col in df_copy.select_dtypes(include=np.number).columns:
            df_copy[col] = df_copy[col].round(2)
        return df_copy.astype(str)

    def dataset_health(dataframe):
        return pd.DataFrame({
            "Column": dataframe.columns,
            "Data Type": dataframe.dtypes.astype(str),
            "Missing %": (dataframe.isna().mean() * 100).round(2),
            "Unique Values": dataframe.nunique()
        })

    # --------------------------------------------------
    # PDF GENERATOR
    # --------------------------------------------------
    def generate_pdf_report(title, df, health_df, image_path=None):
        pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        doc = SimpleDocTemplate(pdf.name, pagesize=A4)

        styles = getSampleStyleSheet()
        elements = []

        usable_width = A4[0] - doc.leftMargin - doc.rightMargin

        elements.append(Paragraph(title, styles["Title"]))
        elements.append(Spacer(1, 12))

        # ---- Summary ----
        summary = pd.DataFrame({
            "Metric": ["Rows", "Columns", "Missing Values"],
            "Value": [df.shape[0], df.shape[1], int(df.isnull().sum().sum())]
        })

        elements.append(Paragraph("Dataset Summary", styles["Heading2"]))
        elements.append(Spacer(1, 6))

        summary_table = Table(
            [summary.columns.tolist()] + summary.values.tolist(),
            colWidths=[usable_width * 0.5, usable_width * 0.5]
        )

        summary_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 1, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("PADDING", (0, 0), (-1, -1), 6)
        ]))

        elements.append(summary_table)
        elements.append(Spacer(1, 12))

        # ---- Preview Table ----
        elements.append(Paragraph("Dataset Preview", styles["Heading2"]))
        elements.append(Spacer(1, 6))

        preview_df = format_df_for_pdf(df)
        col_count = len(preview_df.columns)
        col_widths = [usable_width / col_count] * col_count

        preview_table = Table(
            [preview_df.columns.tolist()] + preview_df.values.tolist(),
            colWidths=col_widths,
            repeatRows=1
        )

        preview_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("WORDWRAP", (0, 0), (-1, -1), True),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))

        elements.append(preview_table)
        elements.append(Spacer(1, 12))

        # ---- Health Table ----
        elements.append(Paragraph("Dataset Health", styles["Heading2"]))
        elements.append(Spacer(1, 6))

        health_df = format_df_for_pdf(health_df)
        health_widths = [
            usable_width * 0.3,
            usable_width * 0.25,
            usable_width * 0.2,
            usable_width * 0.25,
        ]

        health_table = Table(
            [health_df.columns.tolist()] + health_df.values.tolist(),
            colWidths=health_widths,
            repeatRows=1
        )

        health_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("WORDWRAP", (0, 0), (-1, -1), True),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))

        elements.append(health_table)

        if image_path:
            elements.append(Spacer(1, 20))
            elements.append(Image(image_path, width=420, height=300))

        doc.build(elements)
        return pdf.name

    # --------------------------------------------------
    # ANALYSIS FUNCTIONS
    # --------------------------------------------------
    def univariate_analysis(df):
        col = st.sidebar.selectbox("Select Column", df.columns)

        if col in num_cols:
            fig = px.histogram(df, x=col, marginal="box")
        else:
            fig = px.bar(df[col].value_counts().reset_index(),
                         x="index", y=col)

        st.plotly_chart(fig, use_container_width=True)
        return fig

    def bivariate_analysis(df):
        x = st.sidebar.selectbox("X Axis", df.columns)
        y = st.sidebar.selectbox("Y Axis", df.columns)

        if x == y:
            st.warning("X and Y must be different")
            return None

        if x in num_cols and y in num_cols:
            fig = px.scatter(df, x=x, y=y, trendline="ols")
        else:
            fig = px.box(df, x=x, y=y)

        st.plotly_chart(fig, use_container_width=True)
        return fig

    def multivariate_analysis(df):
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="RdBu_r", ax=ax)
        st.pyplot(fig)
        return fig

    # --------------------------------------------------
    # RENDER
    # --------------------------------------------------
    health_df = dataset_health(df)
    fig = None

    if analysis_type == "Dataset Health":
        st.dataframe(health_df, use_container_width=True)

    elif analysis_type == "Univariate Analysis":
        fig = univariate_analysis(df)

    elif analysis_type == "Bivariate Analysis":
        fig = bivariate_analysis(df)

    elif analysis_type == "Multivariate Analysis":
        fig = multivariate_analysis(df)

    # --------------------------------------------------
    # PDF DOWNLOAD
    # --------------------------------------------------
    if generate_pdf:
        img_path = None

        if fig is not None:
            img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name

            if hasattr(fig, "write_image"):   # Plotly
                fig.write_image(img_path, scale=2)
            else:                             # Matplotlib
                fig.savefig(img_path, bbox_inches="tight")
                plt.close()

        pdf_path = generate_pdf_report(
            f"{analysis_type} Report",
            df,
            health_df,
            img_path
        )

        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download PDF",
                f,
                file_name="eda_report.pdf"
            )
