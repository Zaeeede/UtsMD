import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl

# === Fungsi tambahan seperti pada OOP ===
def medianval(df, col):
    return np.median(df[col].dropna())

def fillingnawithmedian(df, col):
    median_value = medianval(df, col)
    df[col].fillna(median_value, inplace=True)
    return df

def correct_values(df):
    # Hanya memperbaiki nilai pada kolom 'person_gender'
    df['person_gender'] = df['person_gender'].replace({'fe male': 'female', 'Male': 'male'})
    return df

# === Load model dan tools preprocessing ===
with open('model_xgb.pkl', 'rb') as file:
    loaded_model = pkl.load(file)

with open('scaler.pkl', 'rb') as file:
    loaded_scaler = pkl.load(file)

with open('encoder.pkl', 'rb') as file:
    loaded_encoder = pkl.load(file)

with open('target_vals.pkl', 'rb') as file:
    loaded_target_vals = pkl.load(file)

# === Preprocessing Function ===
def preprocess_data(data, encoder, scaler):
    try:
        cat_cols = encoder.feature_names_in_
        num_cols = scaler.feature_names_in_

        data = correct_values(data)  # Memperbaiki gender tanpa normalisasi kolom kategorikal lainnya

        for col in num_cols:
            data = fillingnawithmedian(data, col)

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

        all_features = pd.concat([data_encoded, data_scaled], axis=1)

        model_features = loaded_model.get_booster().feature_names
        missing = set(model_features) - set(all_features.columns)
        if missing:
            raise ValueError(f"Data yang diproses tidak memiliki fitur: {missing}")

        return all_features[model_features]

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data (preprocessing): {e}")
        st.stop()

# === Streamlit App ===
def main():
    st.title('Machine Learning Loan Status Prediction App')
    st.subheader('Name: Dennis Purnomo Yohaidi')
    st.subheader('NIM: 2702354741')
    st.info('This app will predict your Loan Status!')

    try:
        data = pd.read_csv('Dataset_A_loan.csv')
    except FileNotFoundError:
        st.error("File Dataset_A_loan.csv tidak ditemukan!")
        return

    cat_features = loaded_encoder.feature_names_in_
    num_features = loaded_scaler.feature_names_in_

    age = st.number_input("Umur Anda (maksimal 144 tahun):", 20, int(data['person_age'].max()))
    gender = st.selectbox("Apa gender anda?:", sorted(data['person_gender'].dropna().unique()))
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

    with st.expander('Data yang Anda Masukkan'):
        st.dataframe(user_data)

    # Test Case Buttons
    col1, col2 = st.columns(2)

    if col1.button("✅ Test Case: Approved"):
        st.session_state.update({
            'person_age': 40,
            'person_gender': 'male',
            'person_education': 'Master',
            'person_income': 120000,
            'person_emp_exp': 15,
            'person_home_ownership': 'MORTGAGE',
            'loan_amnt': 3000,
            'loan_intent': 'MEDICAL',
            'loan_int_rate': 8.5,
            'loan_percent_income': 0.025,
            'cb_person_cred_hist_length': 10,
            'credit_score': 780,
            'previous_loan_defaults_on_file': 'No'
        })

    if col2.button("❌ Test Case: Rejected"):
        st.session_state.update({
            'person_age': 22,
            'person_gender': 'female',
            'person_education': 'High School',
            'person_income': 15000,
            'person_emp_exp': 1,
            'person_home_ownership': 'RENT',
            'loan_amnt': 30000,
            'loan_intent': 'DEBTCONSOLIDATION',
            'loan_int_rate': 19.5,
            'loan_percent_income': 1.5,
            'cb_person_cred_hist_length': 1,
            'credit_score': 470,
            'previous_loan_defaults_on_file': 'Yes'
        })

    # Predict Button
    if st.button("Prediksi"):
        with st.spinner("Sedang memproses prediksi..."):
            try:
                processed_data = preprocess_data(user_data, loaded_encoder, loaded_scaler)
                prediction = loaded_model.predict(processed_data)[0]
                prediction_probs = loaded_model.predict_proba(processed_data)

                pred_label = "Ditolak" if prediction == 0 else "Diterima"
                prob = prediction_probs[0][prediction] * 100

                st.success(f"Prediksi Loan Status: {pred_label} ({prob:.2f}%)")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses data: {e}")

if __name__ == '__main__':
    main()
