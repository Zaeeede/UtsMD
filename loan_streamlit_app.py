import streamlit as st
import pandas as pd
import pickle as pkl
import numpy as np
from utsmodeldeploymentoop import ModelXGB

# Load semua komponen yang sudah disimpan
with open('model_xgb.pkl', 'rb') as file:
    loaded_model = pkl.load(file)

with open('scaler.pkl', 'rb') as file:
    loaded_scaler = pkl.load(file)

with open('encoder.pkl', 'rb') as file:
    loaded_encoder = pkl.load(file)

with open('feature_names.pkl', 'rb') as file:
    loaded_feature_names = pkl.load(file)

with open('target_vals.pkl', 'rb') as file:
    loaded_target_vals = pkl.load(file)

inverse_target_vals = {v: k for k, v in loaded_target_vals.items()}

# Inisialisasi class ModelXGB
model = ModelXGB(data='Dataset_A_loan.csv', loaded_model=loaded_model)
model.data_split(target_column='loan_status')
model.data_preprocessing(loaded_scaler, loaded_encoder, load_from_pickle=True)
model.feature_names = loaded_feature_names

def main():
    st.title('🎯 Loan Status Prediction App')
    st.subheader('Name: Benjamin Eleazar Manafe')
    st.subheader('NIM: 2702340704')
    st.info('This app will predict your loan approval status!')

    with st.expander('📊 Dataset Preview'):
        data = pd.read_csv('Dataset_A_loan.csv')
        st.dataframe(data)

    st.header('📝 Input Your Loan Application Data')

    # Input user
    person_age = st.slider('Age', 18, 100, 25)
    person_gender = st.selectbox('Gender', ['male', 'female'])
    person_education = st.selectbox('Education', ['High School', 'College', 'Bachelor', 'Master'])
    person_income = st.number_input('Annual Income', min_value=0.0, value=50000.0)
    person_emp_exp = st.slider('Employment Experience (Years)', 0, 40, 2)
    person_home_ownership = st.selectbox('Home Ownership', ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
    loan_amnt = st.number_input('Loan Amount', min_value=0.0, value=10000.0)
    loan_intent = st.selectbox('Loan Intent', ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
    loan_int_rate = st.slider('Interest Rate (%)', 0.0, 40.0, 13.5)
    loan_percent_income = st.slider('Loan Percent Income', 0.0, 1.0, 0.3)
    cb_person_cred_hist_length = st.slider('Credit History Length (Years)', 1, 50, 5)
    credit_score = st.slider('Credit Score', 300, 850, 600)
    previous_loan_defaults_on_file = st.selectbox('Previous Loan Default on File', ['Yes', 'No'])

    input_data = pd.DataFrame([{
        'person_age': person_age,
        'person_gender': person_gender,
        'person_education': person_education,
        'person_income': person_income,
        'person_emp_exp': person_emp_exp,
        'person_home_ownership': person_home_ownership,
        'loan_amnt': loan_amnt,
        'loan_intent': loan_intent,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': previous_loan_defaults_on_file
    }])

    st.write('### 🔍 Input Data Preview')
    st.dataframe(input_data)

    if st.button('Predict Loan Status'):
        label, probas = model.predict_input(input_data)
        st.success(f"🎯 Predicted Loan Status: **{inverse_target_vals[label]}**")
        st.write("### 📊 Probability per Class:")
        prob_df = pd.DataFrame([probas], columns=[inverse_target_vals[i] for i in range(len(probas))])
        st.dataframe(prob_df)

if __name__ == '__main__':
    main()
