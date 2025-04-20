import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl


with open('obesity-prediction-streamlit/rf_model.pkl', 'rb') as file:
    loaded_model = pkl.load(file)

with open('obesity-prediction-streamlit/scaler.pkl', 'rb') as file:
    loaded_scaler = pkl.load(file)

with open('obesity-prediction-streamlit/encoder.pkl', 'rb') as file:
    loaded_encoder = pkl.load(file)

with open('obesity-prediction-streamlit/target_vals.pkl', 'rb') as file:
    loaded_target_vals = pkl.load(file)

def main():
    st.title('Machine Leaning Loan Status Prediction App')
    st.subheader('Name: Dennis Purnomo Yohaidi')
    st.subheader('NIM: 2602354741')
    st.info('This app will predict your obesity level!')
    
    with st.expander('**Data**'):
        data = pd.read_csv('obesity-prediction-streamlit/ObesityDataSet_raw_and_data_sinthetic.csv')
        st.write('This is a raw data')
        st.dataframe(data)
    
    with st.expander('**Data Visualization**'):
        st.scatter_chart(data, x='Height', y='Weight', color='NObeyesdad')
    
    gender_data = data['Gender'].unique()
    gender = st.selectbox(
        'What is your Gender?', 
        gender_data,
    )
    
    max_age = data['Age'].max()
    age = st.slider("What is your Age?", 0, max_age)
    
    max_height = data['Height'].max()
    height = st.slider("What is your Height?", 0.0, max_height)
    
    max_weight = data['Weight'].max()
    weight = st.slider("What is your Weight?", 0.0, max_weight)
    
    family_history_data = data['family_history_with_overweight'].unique()
    family_history = st.selectbox(
        'Do you have a family history with overweight?', 
        family_history_data,
    )
    
    favc_data = data['FAVC'].unique()
    favc = st.selectbox(
        'Do you have FAVC (frequent consumption of high-caloric food)?', 
        favc_data,
    )
    
    max_fcvc = data['FCVC'].max()
    fcvc = st.slider('What is your FCVC (frequency of consumption of vegetables)?', 0.0, max_fcvc)
    
    max_ncp = data['NCP'].max()
    ncp = st.slider('What is your NCP (number of main meals)?', 0.0, max_ncp)
    
    caec_data = data['CAEC'].unique()
    caec = st.selectbox(
        'How often do you CAEC (consumption of food between meals)?', 
        caec_data,
    )
    
    smoke_data = data['SMOKE'].unique()
    smoke = st.selectbox(
        'Do you smoke?', 
        smoke_data,
    )
    
    max_ch2o = data['CH2O'].max()
    ch2o = st.slider('What is your CH2O (consumption of water daily)?', 0.0, max_ch2o)
    
    scc_data = data['SCC'].unique()
    scc = st.selectbox(
        'Do you have SCC (squamous cell carcinoma)?', 
        scc_data,
    )
    
    max_faf = data['FAF'].max()
    faf = st.slider('How often do you do FAF (Physical activity frequency)?', 0.0, max_faf)
    
    max_tue = data['TUE'].max()
    tue = st.slider('How often do you do TUE?', 0.0, max_tue)
    
    calc_data = data['CALC'].unique()
    calc = st.selectbox(
        'How often do you do CALC (consumption of alcohol)?', 
        calc_data,
    )
    
    mtrans_data = data['MTRANS'].unique()
    mtrans = st.selectbox(
        'What is you main Transportation?',
        mtrans_data,
    )
    
    st.write('Data input by user')
    user_data = pd.DataFrame([{
        'Gender': gender,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history_with_overweight': family_history,
        'FAVC': favc,
        'FCVC': fcvc,
        'NCP': ncp,
        'CAEC': caec,
        'SMOKE': smoke,
        'CH2O': ch2o,
        'SCC': scc,
        'FAF': faf,
        'TUE': tue,
        'CALC': calc,
        'MTRANS': mtrans
        }])
    st.dataframe(pd.DataFrame(user_data))
    
    st.write('Obesity Prediction')
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
