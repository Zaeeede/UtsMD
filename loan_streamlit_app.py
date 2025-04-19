import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl

# Load model dan tools preprocessing
with open('Loan_XGB_Model.pkl', 'rb') as file:
    loaded_model = pkl.load(file)

with open('scaler.pkl', 'rb') as file:
    loaded_scaler = pkl.load(file)

with open('encoder.pkl', 'rb') as file:
    loaded_encoder = pkl.load(file)

with open('target_vals.pkl', 'rb') as file:
    loaded_target_vals = pkl.load(file)

def main():
    st.title('Machine Learning Loan Status Prediction App')
    st.subheader('Name: Dennis Purnomo Yohaidi')
    st.subheader('NIM: 2702354741')
    st.info('This app will predict your Loan_Status!')

    # Load data asli
    data = pd.read_csv('Dataset_A_loan.csv')
    
    with st.expander('**Sample Raw Data**'):
        st.dataframe(data.head())

    # Ambil kategori dari encoder agar sinkron
    cat_features = loaded_encoder.feature_names_in_
    num_features = loaded_scaler.feature_names_in_

    # --- INPUT USER ---
    age = st.slider("Umur Anda:", 18, int(data['person_age'].max()))
    gender = st.selectbox("Gender:", sorted(data['person_gender'].dropna().unique()))
    education = st.selectbox("Pendidikan Terakhir:", sorted(data['person_education'].dropna().unique()))
    income = st.slider("Pendapatan Tahunan:", 0.0, float(data['person_income'].max()))
    emp_exp = st.slider("Pengalaman Kerja (tahun):", 0, int(data['person_emp_exp'].max()))
    home_ownership = st.selectbox("Kepemilikan Rumah:", sorted(data['person_home_ownership'].dropna().unique()))
    loan_amnt = st.slider("Jumlah Pinjaman:", 0.0, float(data['loan_amnt'].max()))
    loan_intent = st.selectbox("Tujuan Pinjaman:", sorted(data['loan_intent'].dropna().unique()))
    loan_int_rate = st.slider("Suku Bunga Pinjaman:", 0.0, float(data['loan_int_rate'].max()))
    loan_percent_income = st.slider("Persentase Pendapatan untuk Pinjaman:", 0.0, float(data['loan_percent_income'].max()))
    cb_length = st.slider("Panjang Riwayat Kredit:", 0, int(data['cb_person_cred_hist_length'].max()))
    credit_score = st.slider("Skor Kredit:", 0, int(data['credit_score'].max)
