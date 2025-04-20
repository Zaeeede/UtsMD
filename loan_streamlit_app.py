import streamlit as st
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from ModelXGB import ModelXGB  # pastikan class ModelXGB disimpan di file terpisah atau sesuaikan dengan import path

# Memuat model, encoder, dan scaler yang sudah disimpan
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pkl.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pkl.load(f)
    with open('encoder.pkl', 'rb') as f:
        encoder = pkl.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pkl.load(f)
    return model, scaler, encoder, feature_names

# Menyiapkan input untuk prediksi
def user_input_features():
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", options=["male", "female"])
    education = st.selectbox("Education", options=["High School", "Associate", "Bachelor", "Master", "PhD"])
    income = st.number_input("Income", min_value=0.0, value=50000.0)
    emp_exp = st.number_input("Employment Experience (years)", min_value=0, value=5)
    home_ownership = st.selectbox("Home Ownership", options=["OWN", "MORTGAGE", "RENT"])
    loan_amnt = st.number_input("Loan Amount", min_value=0.0, value=20000.0)
    loan_intent = st.selectbox("Loan Intent", options=["PERSONAL", "EDUCATION", "HOME", "AUTO", "MEDICAL"])
    loan_int_rate = st.number_input("Loan Interest Rate", min_value=0.0, value=10.0)
    loan_percent_income = st.number_input("Loan Percentage of Income", min_value=0.0, value=0.3)
    cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=5)
    credit_score = st.number_input("Credit Score", min_value=0, max_value=850, value=700)
    previous_loan_defaults = st.selectbox("Previous Loan Defaults", options=["Yes", "No"])

    data = {
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
        'cb_person_cred_hist_length': cred_hist_length,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': previous_loan_defaults
    }

    return pd.DataFrame([data])

# Fungsi untuk menampilkan hasil prediksi
def predict_loan_status():
    st.title("Loan Status Prediction")

    model, scaler, encoder, feature_names = load_model()
    new_input = user_input_features()

    # Melakukan preprocessing data baru
    encoded_input = pd.DataFrame(
        encoder.transform(new_input[encoder.get_feature_names_out()]),
        columns=encoder.get_feature_names_out(new_input.select_dtypes(include='object').columns),
        index=new_input.index
    )
    
    scaled_input = pd.DataFrame(
        scaler.transform(new_input.select_dtypes(include='number')),
        columns=new_input.select_dtypes(include='number').columns,
        index=new_input.index
    )

    processed_input = pd.concat([scaled_input, encoded_input], axis=1)
    processed_input = processed_input.reindex(columns=feature_names, fill_value=0)

    # Melakukan prediksi
    prediction = model.predict(processed_input)[0]
    prediction_proba = model.predict_proba(processed_input)[0]

    # Menampilkan hasil prediksi
    st.write(f"Prediksi Status Pinjaman: {'Disetujui' if prediction == 1 else 'Ditolak'}")
    st.write("Probabilitas per kelas:")
    proba_df = pd.DataFrame(prediction_proba, index=model.classes_, columns=["Probabilitas"])
    st.write(proba_df)

# Jalankan aplikasi Streamlit
if __name__ == "__main__":
    predict_loan_status()
