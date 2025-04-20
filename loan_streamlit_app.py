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

with open('feature_names.pkl', 'rb') as file:
    model_features = pkl.load(file)

def main():
    st.title('Machine Learning Loan Status Prediction App')
    st.subheader('Name: Dennis Purnomo Yohaidi')
    st.subheader('NIM: 2702354741')
    st.info('This app will predict your Loan Status!')

    # Menampilkan test case
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


    # Load data asli
    data = pd.read_csv('Dataset_A_loan.csv')
    
    # Ambil kategori dari encoder agar sinkron
    cat_features = loaded_encoder.feature_names_in_
    num_features = loaded_scaler.feature_names_in_

    # User Input
    age = st.number_input("Umur Anda (maksimal 144 tahun):", 20, int(data['person_age'].max()))
    gender = st.selectbox("Apa gender anda?:", sorted(clean_categories(data['person_gender'])))
    education = st.selectbox("Pendidikan Terakhir:", sorted(data['person_education'].dropna().unique()))
    income = st.number_input("Pendapatan Tahunan:", 0.0, float(data['person_income'].max()))
    emp_exp = st.number_input("Pengalaman Kerja (tahun):", 0, int(data['person_emp_exp'].max()))
    home_ownership = st.selectbox("Kepemilikan Rumah:", sorted(data['person_home_ownership'].dropna().unique()))
    loan_amnt = st.number_input("Jumlah Pinjaman:", 0.0, float(data['loan_amnt'].max()))
    loan_intent = st.selectbox("Tujuan Pinjaman:", sorted(data['loan_intent'].dropna().unique()))
    loan_int_rate = st.number_input("Suku Bunga Pinjaman:", 0.0, float(data['loan_int_rate'].max()))
    loan_percent_income = st.number_input("Persentase Pendapatan tahunan untuk Pinjaman:", 0.0, float(data['loan_percent_income'].max()))
    cb_length = st.number_input("Panjang Riwayat Kredit (tahun):", 0, int(data['cb_person_cred_hist_length'].max()))
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

    # Normalisasi input data kategorikal 
    user_data = user_data.astype(str)  # memastikan semua str dulu

# Normalisasi person_gender sesuai OOP
    if 'person_gender' in user_data.columns:
        user_data['person_gender'] = user_data['person_gender'].replace({'fe male': 'female', 'Male': 'male'})


    with st.expander('**Data yang Anda Masukkan**'):
        st.dataframe(user_data)

    # kalkulasi prediksi dengan tombol
    if st.button("Prediksi"):
        with st.spinner("Sedang memproses prediksi..."):
            try:
                processed_data = preprocess_data(user_data, loaded_encoder, loaded_scaler)
                prediction = loaded_model.predict(processed_data)[0]
                prediction_probs = loaded_model.predict_proba(processed_data)

                pred_label = "Ditolak" if prediction == 0 else "Diterima"
                prob = prediction_probs[0][prediction] * 100

                st.success(f"**Prediksi Loan Status: {pred_label} ({prob:.2f}%)**")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses data: {e}")

# membersihkan kategori
def clean_categories(series):
    return (
        series
        .dropna()
        .astype(str)
        .str.lower()
        .str.replace(' ', '')
        .unique()
    )

# preprocessing
def preprocess_data(data, encoder, scaler):
    try:
        # Normalisasi input kategorikal
        for col in encoder.feature_names_in_:
            data[col] = data[col].str.lower().str.strip()

        # Split kolom
        cat_cols = encoder.feature_names_in_
        num_cols = scaler.feature_names_in_

        # Transformasi data
        data_encoded = pd.DataFrame(
            encoder.transform(data[cat_cols]),
            columns=encoder.get_feature_names_out(),
            index=data.index
        )
        data_scaled = pd.DataFrame(
            scaler.transform(data[num_cols]),
            columns=num_cols,
            index=data.index
        )

        # Gabung hasil transform
        all_features = pd.concat([data_encoded, data_scaled], axis=1)

        # Pastikan urutan kolom sesuai dengan model
        model_features = loaded_model.get_booster().feature_names
        missing = set(model_features) - set(all_features.columns)
        if missing:
            raise ValueError(f"Data yang diproses tidak memiliki fitur: {missing}")

        return all_features[model_features]

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data (preprocessing): {e}")
        st.stop()

# main
if __name__ == '__main__':
    main()
