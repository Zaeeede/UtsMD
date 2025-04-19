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
    credit_score = st.slider("Skor Kredit:", 0, int(data['credit_score'].max()))
    default_history = st.selectbox("Apakah pernah gagal bayar sebelumnya?", sorted(data['previous_loan_defaults_on_file'].dropna().unique()))

    # Buat DataFrame dari input user
    user_data = pd.DataFrame([{
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

    # --- NORMALISASI INPUT KATEGORIKAL ---
    for col in cat_features:
        if col in user_data.columns:
            user_data[col] = user_data[col].astype(str).str.lower().str.strip()

    with st.expander('**Data yang Anda Masukkan**'):
        st.dataframe(user_data)

    # --- PREDIKSI ---
    try:
        processed_data = preprocess_data(user_data, loaded_encoder, loaded_scaler)
        prediction = loaded_model.predict(processed_data)[0]
        prediction_probs = loaded_model.predict_proba(processed_data)

        inverse_target_vals = {v: k for k, v in loaded_target_vals.items()}
        pred_label = inverse_target_vals[prediction]

        st.success(f"**Prediksi Loan Status: {prediction} [{pred_label}]**")
        st.subheader("Probabilitas Tiap Kelas:")
        prob_df = pd.DataFrame(prediction_probs, columns=[inverse_target_vals[i] for i in range(len(inverse_target_vals))])
        st.dataframe(prob_df)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")

# --- FUNGSI PREPROCESS ---
def preprocess_data(data, encoder, scaler):
    try:
        # Lowercase semua kategori agar konsisten
        for col in encoder.feature_names_in_:
            data[col] = data[col].str.lower().str.strip()
        
        cat_cols = encoder.feature_names_in_
        num_cols = scaler.feature_names_in_

        # Transform kategorikal
        data_encoded = pd.DataFrame(
            encoder.transform(data[cat_cols]),
            columns=encoder.get_feature_names_out(),
            index=data.index
        )

        # Transform numerikal
        data_scaled = pd.DataFrame(
            scaler.transform(data[num_cols]),
            columns=num_cols,
            index=data.index
        )

        # Gabungkan
        return pd.concat([data_encoded, data_scaled], axis=1)

# --- MAIN APP ---
if __name__ == '__main__':
    main()
