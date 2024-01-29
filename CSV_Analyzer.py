import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import statsmodels.api as sm
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_option('deprecation.showPyplotGlobalUse', False)

st.header(":bar_chart: CSV Exploratory Analyzer")
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file and let's see what we find!", type=["csv"])
    st.sidebar.markdown("""
                        [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
                        
                        """)
    
if uploaded_file is not None:
    def load_csv():
        csv = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        return csv
    df = load_csv()

    st.subheader("Uploaded CSV")
    st.write(df)
    st.write(df.dtypes)

    st.write(df.describe())

    columns = list(df.columns)
    numeric_col = []
    text_col = []

    for column in columns:
        if pd.api.types.is_numeric_dtype(df[column].dtype):
            numeric_col.append(column)
        else:
            text_col.append(column)

    for column in numeric_col:
        df = df[df[column].notna()]

    st.subheader("Cleaned CSV")
    st.write(df)

    st.subheader("Data Visualization")
    for column in numeric_col:
        fig = px.histogram(df[column], nbins=10)
        st.plotly_chart(fig)

    st.subheader("Linear Regression Analysis")
    Select = st.multiselect(label=f"Select the type: ",
    key=f"hola",
    options = numeric_col, max_selections=2)

    create = st.button("Generate analysis")
    if create and len(Select) == 2:
        X = sm.add_constant(df[Select[0]])

        # Realizar la regresión lineal con statsmodels
        model = sm.OLS(df[Select[1]], X).fit()

        # Imprimir el resumen de la regresión
        st.subheader("Model Summary")
        st.write(model.summary())
        intercept, slope = model.params

        #Crear el gráfico con plotly
        fig = px.scatter(df, x=Select[0], y=Select[1], trendline="ols")
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black')

        # Mostrar el gráfico
        st.subheader("Linear Regression plot")
        st.latex(f'y = {slope}x + {intercept}')
        st.plotly_chart(fig)

        x = list(df[Select[0]])
        y = list(df[Select[1]]) 
        points = [(x[_], y[_]) for _ in range(len(x))]  

        inertias = []

        for i in range(1,6):
            kmeans = KMeans(n_clusters=i, n_init="auto")
            kmeans.fit(points)
            inertias.append(kmeans.inertia_)

        df_corr = df.corr(method="pearson", numeric_only=True)
        h = px.imshow(df_corr)
        st.subheader("Correlation Heatmap")
        st.plotly_chart(h)
        
        g = px.line(inertias, markers='o')
        st.subheader("K-means clustering")
        st.plotly_chart(g)

        kmeans = KMeans(n_clusters=3, n_init="auto")
        kmeans.fit(points)

        f = px.scatter(df, x=Select[0], y=Select[1], color=kmeans.labels_)
        f.update_xaxes(showline=True, linewidth=1, linecolor='black')
        f.update_yaxes(showline=True, linewidth=1, linecolor='black')
        st.plotly_chart(f)

    st.subheader("Word Cloud")
    option = st.selectbox(
    'Select column', text_col)

    btn_cloud = st.button("Generate word cloud")

    if btn_cloud:
        text = " ".join(list(df[option]))
        word_cloud = WordCloud(width = 1600, height = 800, background_color = "black",
               colormap = "rainbow").generate(text)
        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot()

else:
    st.info('Awaiting for CSV file to be uploaded.')    
    a = pd.DataFrame(
            np.random.rand(100,5),
            columns=['a', 'b', 'c', 'd', 'e']
        )
    
    e = px.scatter(a, color='a')
    st.write("\n")
    st.write("This plot is just decoration :sweat_smile:.")
    st.plotly_chart(e)

    