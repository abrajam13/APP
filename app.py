import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="ML Export", layout="wide")
st.title("ðŸ¤– ML: RegresiÃ³n y ExportaciÃ³n")

uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # 1. ConfiguraciÃ³n de variables
    col_config, col_viz = st.columns(2)
    with col_config:
        target = st.selectbox("Variable a predecir (Y)", df.columns)
        features = st.multiselect("Variables predictoras (X)", df.columns)

    if features and target:
        # 2. Entrenamiento
        X, y = df[features], df[target]
        model = LinearRegression().fit(X, y)
        df['Prediccion_Modelo'] = model.predict(X) # AÃ±adimos predicciones al dataframe

        # 3. MÃ©tricas y GrÃ¡fica
        r2 = r2_score(y, df['Prediccion_Modelo'])
        st.metric("PrecisiÃ³n del Modelo (RÂ²)", f"{r2:.2%}")

        with col_viz:
            if len(features) == 1:
                fig, ax = plt.subplots()
                ax.scatter(X, y, color="blue", alpha=0.5)
                ax.plot(X, df['Prediccion_Modelo'], color="red")
                st.pyplot(fig)

        # 4. EXPORTACIÃ“N DE RESULTADOS
        st.divider()
        st.subheader("ðŸ“¥ Descargar Resultados")
        
        # Convertimos el DataFrame con las predicciones a CSV
        csv_data = df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Descargar CSV con Predicciones",
            data=csv_data,
            file_name="resultados_prediccion.csv",
            mime="text/csv",
        )
        
        st.write("Vista previa de la tabla a descargar:", df.head())

        # 5. PredicciÃ³n individual manual
        st.subheader("ðŸ”® PredicciÃ³n manual")
        inputs = [st.number_input(f"Ingresa {f}", value=0.0) for f in features]
        if st.button("Calcular"):
            st.write(f"Resultado: {model.predict([inputs])}")