import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_resource
def load_model():
    with open('best_ridge_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('preprocessing.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    return model, pipeline

@st.cache_data
def load_eda_data():
    df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
    df_test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')
    return df_train, df_test

df_train, df_test = load_eda_data()

model, pipeline = load_model()

st.title('Прогнозирование цен на автомобили')
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["EDA", "Загрузка файла CSV", "Ручной ввод параметров автомобиля", "Информация о модели"])

# Часть процесса обработки из ноутбука перенес сюда
def preprocess_input(df, pipeline):
    df = df.copy()
    
    df['mileage'] = df['mileage'].astype(str).str.extract('(\d+\.?\d*)').astype(float)
    df['engine'] = df['engine'].astype(str).str.extract('(\d+)').astype(float)
    df['max_power'] = df['max_power'].astype(str).str.extract('(\d+\.?\d*)').astype(float)
    
    pattern1 = r'(\d+\.?\d*)\s*(Nm|kgm)[\s@]*(\d+\.?\d*)[\s\-]*(\d+\.?\d*)?\s*[rR]?[pP]?[mM]?'
    pattern2 = r'(\d+\.?\d*)\s*@\s*[\d,]+[\s\-]*(\d+\.?\d*)?\s*\(?(kgm|Nm)'
    
    torque_split1 = df['torque'].str.extract(pattern1)
    torque_split2 = df['torque'].str.extract(pattern2)
    
    df['torque'] = torque_split1[0].fillna(torque_split2[0]).astype(float)
    df['max_torque_rpm'] = torque_split1[3].fillna(torque_split1[2]).fillna(torque_split2[1]).astype(float)
    
    unit = torque_split1[1].fillna(torque_split2[2])
    df['torque'] = df['torque'].where(unit != 'kgm', df['torque'] * 9.807)
    
    df['brand'] = df['name'].str.split().str[0]
    df['country'] = df['brand'].map(pipeline['country_mapping'])
    df['country'] = df['country'].fillna('Other')
    
    for col, median_val in pipeline['median_values'].items():
        if col in df.columns:
            df[col] = df[col].fillna(median_val)
    
    df['engine'] = df['engine'].astype(int)
    df['seats'] = df['seats'].astype(float)
    
    num_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'seats', 'max_torque_rpm']
    num_df = df[num_cols].copy()
    
    num_scaled = pipeline['scaler'].transform(num_df)
    num_scaled_df = pd.DataFrame(num_scaled, columns=num_cols)
    
    df['seats_cat'] = df['seats'].astype(str)
    df['seats'] = df['seats'].astype(str)
    
    cat_data = df[['fuel', 'seller_type', 'transmission', 'owner', 'country', 'seats']]
    
    encoded = pipeline['encoder'].transform(cat_data)
    encoded_df = pd.DataFrame(encoded, columns=pipeline['encoder'].get_feature_names_out(['fuel', 'seller_type', 'transmission', 'owner', 'country', 'seats']))
    
    num_scaled_df = num_scaled_df.drop(columns=['seats'])
    
    final_df = pd.concat([num_scaled_df, encoded_df], axis=1)
    
    for feat in pipeline['feature_order']:
        if feat not in final_df.columns:
            final_df[feat] = 0
    
    final_df = final_df[pipeline['feature_order']]
    
    return final_df

# Вкладка 1: EDA
with tab1:
    st.header("Анализ данных")
    
    st.subheader("1. Pairplot числовых признаков")
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns
    pairplot = sns.pairplot(df_train[numeric_cols])
    pairplot.fig.suptitle('Попарные распределения числовых признаков', y=1.02)
    st.pyplot(pairplot.fig)
    
    st.subheader("2. Матрица корреляций Пирсона")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df_train.corr(numeric_only=True), annot=True, ax=ax)
    ax.set_title('Матрица корреляций Пирсона')
    st.pyplot(fig)
    
    st.subheader("3. Зависимость цены от года и топлива")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=df_train, x='year', y='selling_price', hue='fuel', alpha=0.7, ax=ax)
    ax.set_title('Зависимость цены от года выпуска и типа топлива')
    ax.grid(True)
    st.pyplot(fig)

# Вкладка 2: Загрузка файла CSV
with tab2:
    st.header("Загрузка CSV файла")
    uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")
    
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.write("Предпросмотр данных:")
        st.dataframe(df_input.head())
        
        if st.button("Предсказать цены"):
            try:
                X_processed = preprocess_input(df_input, pipeline)
                
                predictions = model.predict(X_processed)
                df_input['predicted_price'] = predictions
                
                st.success("Предсказанные значения")
                st.dataframe(df_input[['name', 'predicted_price']])
                
                csv = df_input.to_csv(index=False)
                st.download_button(
                    label="Скачать результаты",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Ошибка: {str(e)}")

# Вкладка 3: Ручной ввод
with tab3:
    st.header("Ручной ввод параметров автомобиля")
    
    col1, col2 = st.columns(2)
    
    with col1:
        year = st.slider("Год выпуска", 1980, 2020, 2015)
        km_driven = st.number_input("Пробег (км)", 0, 3000000, 50000)
        mileage = st.number_input("Расход топлива", 0.0, 100.0, 15.0)
        engine = st.number_input("Объем двигателя (CC)", 500, 7000, 1500)
    
    with col2:
        max_power = st.number_input("Мощность (bhp)", 0.0, 500.0, 100.0)
        seats = st.selectbox("Количество мест", [2, 4, 5, 6, 7, 8, 9, 10, 14])
        fuel = st.selectbox("Топливо", ["Diesel", "Petrol", "CNG", "LPG"])
        transmission = st.selectbox("Коробка передач", ["Manual", "Automatic"])
    
    seller_type = st.selectbox("Тип продавца", ["Individual", "Dealer", "Trustmark Dealer"])
    owner = st.selectbox("Владелец", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
    name = st.text_input("Модель автомобиля", "Maruti Swift Dzire VDI")
    torque_input = st.text_input("Крутящий момент", "260Nm@ 1500-2750rpm")
    
    if st.button("Предсказать цену"):
        input_data = pd.DataFrame([{
            'name': name,
            'year': year,
            'km_driven': km_driven,
            'fuel': fuel,
            'seller_type': seller_type,
            'transmission': transmission,
            'owner': owner,
            'mileage': mileage,
            'engine': engine,
            'max_power': max_power,
            'torque': torque_input,  
            'seats': seats,
        }])
        
        X_processed = preprocess_input(input_data, pipeline)
        prediction = model.predict(X_processed)[0]
        
        st.success(f"Предсказанная цена: {prediction:,.0f}")

# Вкладка 4: Модель
with tab4:
    st.header("Характеристики модели")
    
    st.subheader("Важность признаков")
    
    coef_df = pd.DataFrame({
        'feature': pipeline['feature_order'],
        'coefficient': model.coef_
    })
    coef_df = coef_df.sort_values('coefficient', key=abs).head(20)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(coef_df['feature'], coef_df['coefficient'])
    ax.set_xlabel('Коэффициент')
    ax.set_title('Топ-20 самых важных признаков')
    plt.tight_layout()
    st.pyplot(fig)
    st.subheader("Метрики качества")
    st.metric("Количество признаков", len(model.coef_))
    st.metric("R2 на тесте при разработке модели", "0.682")

if __name__ == '__main__':
    pass