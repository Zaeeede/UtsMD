import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl

# Load model dan preprocessing tools
with open('Loan_XGB_Model.pkl', 'rb') as file:
    loaded_model = pkl.load(file)

with open('scaler.pkl', 'rb') as file:
    loaded_scaler = pkl.load(file)

with open('encoder.pkl', 'rb') as file:
    loaded_encoder = pkl.load(file)

with open('target_vals.pkl', 'rb') as file:
    loaded_target_vals = pkl.load(file)

# Fungsi Preprocessing
def preprocess_data(data, encoder, scaler):
    # Koreksi nilai jika perlu
    data['person_gender'] = data['person_gender'].replace({'fe male': 'female', 'Male': 'male'})

    # Kolom kategorikal dan numerik
    cat_cols = encoder.feature_names_in_
    num_cols = scaler.feature_names_in_

    # Cek kolom yang hilang
    expected_cols = set(cat_cols) | set(num_cols)
    missing_cols = expected_cols - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in input: {missing_cols}")

    # Transformasi
    data_encoded = pd.DataFrame(encoder.transform(data[cat_cols]),
                                columns=encoder.get_feature_names_out(),
                                index=data.index)
    data_scaled = pd.DataFrame(scaler.transform(data[num_cols]),
                               columns=num_cols,
                               index=data.index)

    return pd.concat([data_encoded, data_scaled], axis=1)

# Streamlit App
def main():
    st.title('Machine Learning Loan_Status Prediction App')
    st.subheader('Name: Dennis Purnomo Yohaidi')
    st.subheader('NIM: 2702354741')
    st.info('This app will predict your Loan_Status!')

    # Load data untuk referensi nilai input
    data = pd.read_csv('Dataset_A_loan.csv')

    with st.expander('📊 View Raw Data'):
        st.dataframe(data)

    # Ambil nilai unik dan maksimum untuk field input
    age = st.slider("Berapa umur anda?", 0, int(data['person_age'].max()))
    gender = st.selectbox("Apa gender anda?", sorted(data['person_gender'].dropna().unique()))
    education = st.selectbox("Apa tingkat pendidikan tertinggi anda?", sorted(data['person_education'].dropna().unique()))
    income = st.slider("Berapa jumlah pendapatan tahunan anda?", 0.0, float(data['person_income'].max()))
    emp_exp = st.slider("Berapa tahun pengalaman kerja anda?", 0, int(data['person_emp_exp'].max()))
    home_ownership = st.selectbox("Apakah anda punya rumah?", sorted(data['person_home_ownership'].dropna().unique()))
    loan_amnt = st.slider("Berapa jumlah pinjaman anda?", 0.0, float(data['loan_amnt'].max()))
    loan_intent = st.selectbox("Apa tujuan pinjaman anda?", sorted(data['loan_intent'].dropna().unique()))
    loan_int_rate = st.slider("Berapa suku bunga pinjaman anda?", 0.0, float(data['loan_int_rate'].max()))
    loan_percent_income = st.slider("Berapa persen penghasilan anda untuk pinjaman?", 0.0, float(data['loan_percent_income'].max()))
    cb_person_cred_hist_length = st.slider("Berapa lama riwayat kredit anda?", 0.0, float(data['cb_person_cred_hist_length'].max()))
    credit_score = st.slider("Berapa skor kredit anda?", 0, int(data['credit_score'].max()))
    previous_loan_defaults_on_file = st.selectbox("Apakah ada riwayat gagal bayar sebelumnya?", sorted(data['previous_loan_defaults_on_file'].dropna().unique()))

    # Buat data user
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
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': previous_loan_defaults_on_file
    }])

    st.write('🔍 Data input oleh user')
    st.dataframe(user_data)

    if st.button('Predict Loan Status'):
        try:
            processed_data = preprocess_data(user_data, loaded_encoder, loaded_scaler)
            prediction = loaded_model.predict(processed_data)[0]
            prediction_probs = loaded_model.predict_proba(processed_data)[0]

            inverse_target_vals = {v: k for k, v in loaded_target_vals.items()}

            st.success(f"✅ Prediksi Status Pinjaman Anda: **{inverse_target_vals[prediction]}**")
            st.subheader("📈 Probabilitas Prediksi:")
            st.dataframe(pd.DataFrame([prediction_probs], columns=[inverse_target_vals[i] for i in range(len(prediction_probs))]))
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses data: {e}")

if __name__ == '__main__':
    main()
