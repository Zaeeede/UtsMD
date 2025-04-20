import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from model_xgb import ModelXGB  # pastikan class ModelXGB disimpan di file ini

# Load objek-objek model dan preprocessing
with open('model_xgb.pkl', 'rb') as file:
    loaded_model = pkl.load(file)

with open('scaler.pkl', 'rb') as file:
    loaded_scaler = pkl.load(file)

with open('encoder.pkl', 'rb') as file:
    loaded_encoder = pkl.load(file)

with open('feature_names.pkl', 'rb') as file:
    loaded_features = pkl.load(file)

# Inisialisasi OOP
model = ModelXGB(data='Dataset_A_loan.csv', loaded_model=loaded_model)
model.data_split(target_column='loan_status')
model.data_preprocessing(scaler=loaded_scaler, encoder=loaded_encoder, load_from_pickle=True)
model.feature_names = loaded_features  # sinkronisasi fitur model

# UI Streamlit
def main():
    st.title('Machine Learning Loan Status Prediction App')
    st.subheader('Name: Dennis Purnomo Yohaidi')
    st.subheader('NIM: 2702354741')
    st.info('This app will predict your Loan Status!')

    # Contoh data
    accepted_case = {
        'person_age': 25,
        'person_gender': 'female',
        'person_education': 'high school',
        'person_income': 12438.0,
        'person_emp_exp': 3,
        'person_home_ownership': 'mortgage',
        'loan_amnt': 5500.0,
        'loan_intent': 'medical',
        'loan_int_rate': 12.87,
        'loan_percent_income': 0.44,
        'cb_person_cred_hist_length': 3,
        'credit_score': 635,
        'previous_loan_defaults_on_file': 'no'
    }

    rejected_case = {
        'person_age': 25,
        'person_gender': 'male',
        'person_education': 'high school',
        'person_income': 165792.0,
        'person_emp_exp': 4,
        'person_home_ownership': 'rent',
        'loan_amnt': 34800.0,
        'loan_intent': 'personal',
        'loan_int_rate': 16.77,
        'loan_percent_income': 0.21,
        'cb_person_cred_hist_length': 2,
        'credit_score': 662,
        'previous_loan_defaults_on_file': 'no'
    }

    st.markdown("### 🔵 Contoh Test Case Diterima")
    st.dataframe(pd.DataFrame([accepted_case]))

    st.markdown("### 🔴 Contoh Test Case Ditolak")
    st.dataframe(pd.DataFrame([rejected_case]))

    # Ambil opsi dari data
    data = model.data
    gender = st.selectbox("Gender:", sorted(data['person_gender'].dropna().unique()))
    education = st.selectbox("Pendidikan Terakhir:", sorted(data['person_education'].dropna().unique()))
    home_ownership = st.selectbox("Kepemilikan Rumah:", sorted(data['person_home_ownership'].dropna().unique()))
    loan_intent = st.selectbox("Tujuan Pinjaman:", sorted(data['loan_intent'].dropna().unique()))
    default_history = st.selectbox("Gagal Bayar Sebelumnya:", sorted(data['previous_loan_defaults_on_file'].dropna().unique()))

    # Numeric inputs
    age = st.number_input("Umur:", 20, 144, step=1)
    income = st.number_input("Pendapatan Tahunan:", 0.0)
    emp_exp = st.number_input("Pengalaman Kerja (tahun):", 0, 100, step=1)
    loan_amnt = st.number_input("Jumlah Pinjaman:", 0.0)
    loan_int_rate = st.number_input("Suku Bunga (%):", 0.0)
    loan_percent_income = st.number_input("Persentase Pendapatan utk Pinjaman:", 0.0)
    cb_length = st.number_input("Lama Histori Kredit (tahun):", 0, 100, step=1)
    credit_score = st.slider("Skor Kredit:", 0, 850)

    # Buat dataframe input
    user_input = pd.DataFrame([{
        'person_age': age,
        'person_gender': gender,
        'person_education': education,
        'person_income': income,
        'person_emp_exp': emp_exp,
        'person_home_ownership': home_ownership,
        'loan_amnt': loan_amnt,
        'loan_intent': loan_intent,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_length,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': default_history
    }])

    # Normalisasi nilai gender
    user_input['person_gender'] = user_input['person_gender'].replace({'fe male': 'female', 'Male': 'male'})

    with st.expander('Lihat Data Input'):
        st.dataframe(user_input)

    # Prediksi
    if st.button("Prediksi"):
        try:
            with st.spinner("Memproses..."):
                prediction, probas = model.predict_input(user_input)
                label = "Diterima" if prediction == 1 else "Ditolak"
                st.success(f"**Status Pinjaman: {label} ({probas[prediction]*100:.2f}%)**")
        except Exception as e:
            st.error(f"Terjadi error: {e}")

if __name__ == '__main__':
    main()
