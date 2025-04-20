import streamlit as st
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from model_xgb import ModelXGB  # Pastikan file ModelXGB ada di direktori yang sama atau sesuaikan importnya

# Fungsi untuk memuat model dan komponen yang diperlukan
def load_model():
    # Memuat model, scaler, encoder, dan feature_names dari pickle
    with open('xgb_model.pkl', 'rb') as file:
        model = pkl.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pkl.load(file)
    with open('encoder.pkl', 'rb') as file:
        encoder = pkl.load(file)
    with open('feature_names.pkl', 'rb') as file:
        feature_names = pkl.load(file)

    return model, scaler, encoder, feature_names

# Fungsi untuk menampilkan antarmuka pengguna
def show_input_form():
    # Input data untuk prediksi
    st.title('Prediksi Status Pinjaman')
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    gender = st.selectbox('Gender', options=['male', 'female'])
    education = st.selectbox('Education Level', options=['High School', 'Bachelor', 'Master', 'PhD'])
    income = st.number_input('Income', min_value=0, value=50000)
    emp_exp = st.number_input('Employment Experience (Years)', min_value=0, value=2)
    home_ownership = st.selectbox('Home Ownership', options=['RENT', 'OWN', 'MORTGAGE'])
    loan_amount = st.number_input('Loan Amount', min_value=0, value=5000)
    loan_intent = st.selectbox('Loan Intent', options=['PERSONAL', 'EDUCATIONAL', 'MEDICAL', 'AUTO'])
    loan_int_rate = st.number_input('Loan Interest Rate (%)', min_value=0.0, max_value=100.0, value=5.0)
    loan_percent_income = st.number_input('Loan Percentage of Income', min_value=0.0, max_value=1.0, value=0.3)
    credit_history = st.selectbox('Credit History', options=['No', 'Yes'])
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=700)
    previous_defaults = st.selectbox('Previous Loan Defaults', options=['No', 'Yes'])

    # Menyusun data input menjadi dataframe
    input_data = pd.DataFrame([{
        'person_age': age,
        'person_gender': gender,
        'person_education': education,
        'person_income': income,
        'person_emp_exp': emp_exp,
        'person_home_ownership': home_ownership,
        'loan_amnt': loan_amount,
        'loan_intent': loan_intent,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': emp_exp,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': previous_defaults
    }])

    return input_data

# Menampilkan hasil prediksi
def display_prediction(model, scaler, encoder, feature_names):
    input_data = show_input_form()

    # Memproses data input
    label, probas = model.predict_input(input_data)

    st.subheader(f'Prediksi Status Pinjaman: {label}')
    st.write("Probabilitas per kelas:")
    probas_df = pd.DataFrame(probas, columns=['No', 'Yes'])
    st.write(probas_df)

# Main function untuk menjalankan aplikasi Streamlit
def main():
    # Memuat model, scaler, encoder, dan feature_names
    model, scaler, encoder, feature_names = load_model()

    # Menampilkan form input dan hasil prediksi
    display_prediction(model, scaler, encoder, feature_names)

# Menjalankan aplikasi Streamlit
if __name__ == '__main__':
    main()
