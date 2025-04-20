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

# Membersihkan kategori
def clean_categories(series):
    return (
        series
        .dropna()
        .astype(str)
        .str.lower()
        .str.replace(' ', '')
        .unique()
    )

# Preprocessing
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

# Aplikasi utama
def main():
    st.title('Machine Learning Loan Status Prediction App')
    st.subheader('Name: Dennis Purnomo Yohaidi')
    st.subheader('NIM: 2702354741')
    st.info('This app will predict your Loan Status!')

    # Load data asli
    data = pd.read_csv('Dataset_A_loan.csv')
    
    # Ambil fitur dari encoder dan scaler
    cat_features = loaded_encoder.feature_names_in_
    num_features = loaded_scaler.feature_names_in_

    # Inisialisasi session state
    if "test_case" not in st.session_state:
        st.session_state.test_case = None

    # Tombol Test Case
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔵 Gunakan Test Case Diterima"):
            st.session_state.test_case = "accept"
    with col2:
        if st.button("🔴 Gunakan Test Case Ditolak"):
            st.session_state.test_case = "reject"
    with col3:
        if st.button("🔁 Reset Form"):
            st.session_state.test_case = None
            st.experimental_rerun()

    # Default input berdasarkan test case
    if st.session_state.test_case == "accept":
        default_input = {
            'person_age': 35,
            'person_gender': 'male',
            'person_education': 'high school',
            'person_income': 80000.0,
            'person_emp_exp': 10,
            'person_home_ownership': 'rent',
            'loan_amnt': 5000.0,
            'loan_intent': 'medical',
            'loan_int_rate': 5.27,
            'loan_percent_income': 0.06,
            'cb_person_cred_hist_length': 14,
            'credit_score': 700,
            'previous_loan_defaults_on_file': 'no'
        }
    elif st.session_state.test_case == "reject":
        default_input = {
            'person_age': 26,
            'person_gender': 'female',
            'person_education': 'college',
            'person_income': 25000.0,
            'person_emp_exp': 2,
            'person_home_ownership': 'rent',
            'loan_amnt': 10000.0,
            'loan_intent': 'debt consolidation',
            'loan_int_rate': 15.27,
            'loan_percent_income': 0.4,
            'cb_person_cred_hist_length': 3,
            'credit_score': 510,
            'previous_loan_defaults_on_file': 'yes'
        }
    else:
        default_input = {
            'person_age': 30,
            'person_gender': 'female',
            'person_education': 'college',
            'person_income': 50000.0,
            'person_emp_exp': 5,
            'person_home_ownership': 'rent',
            'loan_amnt': 4000.0,
            'loan_intent': 'medical',
            'loan_int_rate': 10.0,
            'loan_percent_income': 0.1,
            'cb_person_cred_hist_length': 5,
            'credit_score': 650,
            'previous_loan_defaults_on_file': 'n'
        }

    # Input Form
    age = st.number_input("Umur Anda (maksimal 144 tahun):", 20, int(data['person_age'].max()), value=default_input['person_age'])
    gender = st.selectbox("Apa gender anda?:", sorted(clean_categories(data['person_gender'])), index=sorted(clean_categories(data['person_gender'])).index(default_input['person_gender']))
    education = st.selectbox("Pendidikan Terakhir:", sorted(data['person_education'].dropna().unique()), index=sorted(data['person_education'].dropna().unique()).index(default_input['person_education']))
    income = st.number_input("Pendapatan Tahunan:", 0.0, float(data['person_income'].max()), value=default_input['person_income'])
    emp_exp = st.number_input("Pengalaman Kerja (tahun):", 0, int(data['person_emp_exp'].max()), value=default_input['person_emp_exp'])
    home_ownership = st.selectbox("Kepemilikan Rumah:", sorted(data['person_home_ownership'].dropna().unique()), index=sorted(data['person_home_ownership'].dropna().unique()).index(default_input['person_home_ownership']))
    loan_amnt = st.number_input("Jumlah Pinjaman:", 0.0, float(data['loan_amnt'].max()), value=default_input['loan_amnt'])
    loan_intent = st.selectbox("Tujuan Pinjaman:", sorted(data['loan_intent'].dropna().unique()), index=sorted(data['loan_intent'].dropna().unique()).index(default_input['loan_intent']))
    loan_int_rate = st.number_input("Suku Bunga Pinjaman:", 0.0, float(data['loan_int_rate'].max()), value=default_input['loan_int_rate'])
    loan_percent_income = st.number_input("Persentase Pendapatan tahunan untuk Pinjaman:", 0.0, float(data['loan_percent_income'].max()), value=default_input['loan_percent_income'])
    cb_length = st.number_input("Panjang Riwayat Kredit (tahun):", 0, int(data['cb_person_cred_hist_length'].max()), value=default_input['cb_person_cred_hist_length'])
    credit_score = st.slider("Skor Kredit:", 0, int(data['credit_score'].max()), value=default_input['credit_score'])
    default_history = st.selectbox("Apakah pernah gagal bayar sebelumnya?", sorted(data['previous_loan_defaults_on_file'].dropna().unique()), index=sorted(data['previous_loan_defaults_on_file'].dropna().unique()).index(default_input['previous_loan_defaults_on_file']))

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

    with st.expander('**Data yang Anda Masukkan**'):
        st.dataframe(user_data)

    # Tombol Prediksi
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

# Run aplikasi
if __name__ == '__main__':
    main()
