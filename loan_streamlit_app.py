import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl

with open('XGB_model.pkl', 'rb') as file:
    loaded_model = pkl.load(file)

with open('scaler.pkl', 'rb') as file:
    loaded_scaler = pkl.load(file)

with open('encoder.pkl', 'rb') as file:
    loaded_encoder = pkl.load(file)

with open('target_vals.pkl', 'rb') as file:
    loaded_target_vals = pkl.load(file)

def main():
    st.title('Machine Leaning Loan_Status Prediction App')
    st.subheader('Name: Dennis Purnomo Yohaidi')
    st.subheader('NIM: 2702354741')
    st.info('This app will predict your Loan_Status!')
    
    with st.expander('**Data**'):
        data = pd.read_csv('Dataset_A_loan.csv')
        st.write('This is a raw data')
        st.dataframe(data)
     
    with st.expander('**Data Visualization**'):
        st.scatter_chart(data, x='Height', y='Weight', color='loan_status')

    max_age = data['person_age'].max()
    age = st.slider("What is your Age?", 0, max_age)
    
    gender_data = data['person_gender'].unique()
    gender = st.selectbox(
        'What is your Gender?', 
        gender_data,
    )
    
    education_data = data['person_education'].unique()
    education = st.selectbox(
        'What is your Education?', 
        education_data,
    )
    
    max_income = data['person_income'].max()
    income = st.slider("What is your income?", 0, max_income)
    
    max_emp_exp = data['person_emp_exp'].max()
    emp_exp = st.slider("What is your emp_exp?", 0, max_emp_exp)

    home_ownership_data = data['person_home_ownership'].unique()
    home_ownership = st.selectbox(
        'Do you have a home?', 
        home_ownership_data,
    )

    max_loan_amnt = data['loan_amnt'].max()
    loan_amnt = st.slider("What is your loan_amnt?", 0, max_loan_amnt)
    
    loan_intent_data = data['loan_intent'].unique() 
    loan_intent = st.selectbox(
        'Do you have loan_intent?', 
        loan_intent_data,
    )
    
    max_loan_int_rate = data['loan_int_rate'].max()
    loan_int_rate = st.slider('What is your FCVC (frequency of consumption of vegetables)?', 0, max_loan_int_rate)
    
    max_loan_percent_income = data['loan_percent_income'].max()
    loan_percent_income = st.slider('What is your NCP (number of main meals)?', 0.0, max_loan_percent_income)

    max_cb_person_cred_hist_length = data['cb_person_cred_hist_length'].max()
    cb_person_cred_hist_length = st.slider('What is your cb_person_cred_hist_length?', 0, max_cb_person_cred_hist_length)

    max_credit_score = data['credit_score'].max()
    credit_score = st.slider('What is your credit_score?', 0, max_credit_score)
    
    previous_loan_defaults_on_file_data = data['previous_loan_defaults_on_file'].unique()
    previous_loan_defaults_on_file = st.selectbox(
        'How often do you previous_loan_defaults_on_file?', 
        previous_loan_defaults_on_file_data,
    )
    
    st.write('Data input by user')
    user_data = pd.DataFrame([{
        'person_age': age,
        'person_gender': gender, 
        'education': education,
        'income': income,
        'emp_exp': emp_exp,
        'home_ownership': home_ownership,
        'loan_amnt': loan_amnt,
        'loan_intent': loan_intent,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': previous_loan_defaults_on_file
        }])
    st.dataframe(pd.DataFrame(user_data))
    
    st.write('Loan Status Prediction')
    processed_data = preprocess_data(data=user_data, encoder=loaded_encoder, scaler=loaded_scaler)
    predictions = loaded_model.predict(processed_data)
    print(predictions)
    inverse_target_vals = {v: k for k, v in loaded_target_vals.items()}
    prediction_probs = loaded_model.predict_proba(processed_data)
    st.dataframe(pd.DataFrame(prediction_probs, columns=inverse_target_vals.values()))
    st.write('The predicted output is: ', predictions[0], '**[', inverse_target_vals[predictions[0]] ,']**')
    
    st.header('🥳')
    
def preprocess_data(data, encoder, scaler):
    data_encoded = pd.DataFrame(encoder.transform(data.select_dtypes(exclude=np.number)), columns=encoder.get_feature_names_out())
    data_scaled = pd.DataFrame(scaler.transform(data.select_dtypes(include=np.number)), columns=data.select_dtypes(include=np.number).columns)
    data = pd.concat([data_encoded, data_scaled], axis=1)
    return data
    
if __name__ == '__main__':
    main()
