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

    # Menentukan nilai max yang valid untuk setiap inputan
    max_income = float(data['person_income'].max()) if isinstance(data['person_income'].max(), (int, float)) else 100000.0
    max_age = int(data['person_age'].max()) if isinstance(data['person_age'].max(), (int, float)) else 100
    max_emp_exp = int(data['person_emp_exp'].max()) if isinstance(data['person_emp_exp'].max(), (int, float)) else 50
    max_loan_amnt = float(data['loan_amnt'].max()) if isinstance(data['loan_amnt'].max(), (int, float)) else 1000000.0
    max_credit_score = int(data['credit_score'].max()) if isinstance(data['credit_score'].max(), (int, float)) else 850

    # Menambahkan tombol untuk tes case
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Test Case: Approved"):
            st.session_state.update({
                'person_age': 22,
                'person_gender': 'female',
                'person_education': 'Master',
                'person_income': 71948.0,
                'person_emp_exp': 0,
                'person_home_ownership': 'RENT',
                'loan_amnt': 35000.0,
                'loan_intent': 'PERSONAL',
                'loan_int_rate': 16.02,
                'loan_percent_income': 0.49,
                'cb_person_cred_hist_length': 3,
                'credit_score': 561,
                'previous_loan_defaults_on_file': 'No'
            })

    with col2:
        if st.button("❌ Test Case: Rejected"):
            st.session_state.update({
                'person_age': 21,
                'person_gender': 'female',
                'person_education': 'High School',
                'person_income': 12282.0,
                'person_emp_exp': 0,
                'person_home_ownership': 'OWN',
                'loan_amnt': 1000.0,
                'loan_intent': 'EDUCATION',
                'loan_int_rate': 11.14,
                'loan_percent_income': 0.08,
                'cb_person_cred_hist_length': 2,
                'credit_score': 504,
                'previous_loan_defaults_on_file': 'Yes'
            })

    # Form input user
    age = st.number_input("Umur Anda (maksimal 144 tahun):", 20, max_age, value=st.session_state.get('person_age', 40))
    gender = st.selectbox("Apa gender anda?:", sorted(data['person_gender'].dropna().unique()), index=sorted(data['person_gender'].dropna().unique()).index(st.session_state.get('person_gender', 'female')))
    education = st.selectbox("Pendidikan Terakhir:", sorted(data['person_education'].dropna().unique()), index=sorted(data['person_education'].dropna().unique()).index(st.session_state.get('person_education', 'Master')))
    income = st.number_input("Pendapatan Tahunan:", 0.0, max_income, value=st.session_state.get('person_income', 30000.0))
    emp_exp = st.number_input("Pengalaman Kerja (tahun):", 0, max_emp_exp, value=st.session_state.get('person_emp_exp', 5))
    home_ownership = st.selectbox("Kepemilikan Rumah:", sorted(data['person_home_ownership'].dropna().unique()), index=sorted(data['person_home_ownership'].dropna().unique()).index(st.session_state.get('person_home_ownership', 'RENT')))
    loan_amnt = st.number_input("Jumlah Pinjaman:", 0.0, max_loan_amnt, value=st.session_state.get('loan_amnt', 10000.0))
    loan_intent = st.selectbox("Tujuan Pinjaman:", sorted(data['loan_intent'].dropna().unique()), index=sorted(data['loan_intent'].dropna().unique()).index(st.session_state.get('loan_intent', 'PERSONAL')))
    loan_int_rate = st.number_input("Suku Bunga Pinjaman:", 0.0, float(data['loan_int_rate'].max()), value=st.session_state.get('loan_int_rate', 10.0))
    loan_percent_income = st.number_input("Persentase Pendapatan tahunan untuk Pinjaman:", 0.0, float(data['loan_percent_income'].max()), value=st.session_state.get('loan_percent_income', 0.1))
    cb_length = st.number_input("Panjang Riwayat Kredit (tahun):", 0, int(data['cb_person_cred_hist_length'].max()), value=st.session_state.get('cb_person_cred_hist_length', 5))
    credit_score = st.slider("Skor Kredit:", 0, max_credit_score, value=st.session_state.get('credit_score', 700))
    default_history = st.selectbox("Apakah pernah gagal bayar sebelumnya?", sorted(data['previous_loan_defaults_on_file'].dropna().unique()), index=sorted(data['previous_loan_defaults_on_file'].dropna().unique()).index(st.session_state.get('previous_loan_defaults_on_file', 'No')))

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

    if st.button("Prediksi"):
        with st.spinner("Sedang memproses prediksi..."):
            try:
                processed_data = preprocess_data(user_data, loaded_encoder, loaded_scaler)
                prediction = loaded_model.predict(processed_data)[0]
                prediction_probs = loaded_model.predict_proba(processed_data)

            # Mapping prediksi ke label
                label_mapping = {0: "Ditolak", 1: "Diterima"}
                pred_label = label_mapping.get(prediction, str(prediction))
                prob = prediction_probs[0][prediction] * 100

                st.success(f"Hasil Prediksi: **{pred_label}** ({prob:.2f}%)")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses data: {e}")


if __name__ == '__main__':
    main()
