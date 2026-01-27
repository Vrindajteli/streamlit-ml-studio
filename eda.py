def show():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import tempfile
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px

    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    )
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors

    from background import apply_background

    apply_background()

    st.title("Exploratory Data Analysis")

    file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if not file:
        st.info("Upload a dataset to continue")
        st.stop()

    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.sidebar.title("EDA Options")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Dataset Health", "Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"]
    )
    generate_pdf = st.sidebar.button("Generate PDF Report")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))

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

    def generate_pdf_report(title, df, health_df, image_path=None):
        pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        doc = SimpleDocTemplate(pdf.name, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []
        usable_width = A4[0] - doc.leftMargin - doc.rightMargin

        elements.append(Paragraph(title, styles["Title"]))
        elements.append(Spacer(1, 12))

        summary_data = [["Metric", "Value"], ["Rows", str(df.shape[0])], ["Columns", str(df.shape[1])]]
        summary_table = Table(summary_data, colWidths=[usable_width * 0.5] * 2)
        summary_table.setStyle(
            TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
        elements.append(summary_table)
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Dataset Preview (Horizontal Chunks)", styles["Heading2"]))
        preview_full = format_df_for_pdf(df)
        chunk_size = 5
        for i in range(0, len(preview_full.columns), chunk_size):
            chunk = preview_full.iloc[:, i:i + chunk_size]
            data = [chunk.columns.tolist()] + chunk.values.tolist()
            t = Table(data, colWidths=[usable_width / len(chunk.columns)] * len(chunk.columns))
            t.setStyle(TableStyle(
                [("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                 ("FONTSIZE", (0, 0), (-1, -1), 7)]))
            elements.append(t)
            elements.append(Spacer(1, 10))

        elements.append(Paragraph("Dataset Health", styles["Heading2"]))
        health_pdf = format_df_for_pdf(health_df)
        h_table = Table([health_pdf.columns.tolist()] + health_pdf.values.tolist(), colWidths=[usable_width * 0.25] * 4)
        h_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 7)]))
        elements.append(h_table)

        if image_path:
            elements.append(Spacer(1, 20))
            elements.append(Image(image_path, width=420, height=300))

        doc.build(elements)
        return pdf.name

    health_data = dataset_health(df)
    fig = None

    if analysis_type == "Dataset Health":
        st.dataframe(health_data, use_container_width=True)

    elif analysis_type == "Univariate Analysis":
        col = st.sidebar.selectbox("Select Column", ["Select"] + list(df.columns))
        if col != "Select":
            fig = px.histogram(df, x=col) if col in num_cols else px.bar(df[col].value_counts())
            st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Bivariate Analysis":
        x = st.sidebar.selectbox("Select X Axis", ["Select"] + list(df.columns))
        # Y axis only appears if X is selected
        if x != "Select":
            y = st.sidebar.selectbox("Select Y Axis", ["Select"] + list(df.columns))
            if y != "Select":
                fig = px.scatter(df, x=x, y=y)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please select an X-axis column to begin.")

    elif analysis_type == "Multivariate Analysis":
        if len(num_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap="RdBu_r", ax=ax)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Not enough numeric columns")

    if generate_pdf:
        img_path = None
        if fig:
            tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            if isinstance(fig, plt.Figure):
                # Fixed: bbox_inches='tight' prevents labels being cut off in PDF
                fig.savefig(tmp_img.name, bbox_inches='tight', dpi=300)
            else:
                fig.write_image(tmp_img.name)
            img_path = tmp_img.name

        pdf_path = generate_pdf_report("EDA Report", df, health_data, img_path)
        with open(pdf_path, "rb") as f:
            st.sidebar.download_button("Download PDF", f, file_name="EDA_Report.pdf")
