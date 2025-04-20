import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# ====== Class OOP Langsung di Streamlit ======
class ModelXGB:
    def __init__(self, data, loaded_model=None):
        self.data = pd.read_csv(data)
        self.fitted_model = loaded_model

    def predict_input(self, input_data):
        if self.fitted_model is None:
            raise ValueError("Model belum dimuat!")

        encoded_input = pd.DataFrame(
            self.encoder.transform(input_data[self.cat_cols]),
            columns=self.encoder.get_feature_names_out(self.cat_cols),
            index=input_data.index
        )

        scaled_input = pd.DataFrame(
            self.scaler.transform(input_data[self.num_cols]),
            columns=self.num_cols,
            index=input_data.index
        )

        processed_input = pd.concat([scaled_input, encoded_input], axis=1)
        processed_input = processed_input.reindex(columns=self.feature_names, fill_value=0)

        prediction = self.fitted_model.predict(processed_input)[0]
        prediction_proba = self.fitted_model.predict_proba(processed_input)[0]

        return prediction, prediction_proba

# ===== Load all assets =====
@st.cache_resource
def load_all():
    with open("model.pkl", "rb") as f:
        model = pkl.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pkl.load(f)
    with open("encoder.pkl", "rb") as f:
        encoder = pkl.load(f)
    with open("feature_names.pkl", "rb") as f:
        feature_names = pkl.load(f)
    return model, scaler, encoder, feature_names

fitted_model, scaler, encoder, feature_names = load_all()

# ===== Setup class dummy untuk prediksi =====
model_obj = ModelXGB(data='Dataset_A_loan.csv', loaded_model=fitted_model)
model_obj.scaler = scaler
model_obj.encoder = encoder
model_obj.feature_names = feature_names

# Definisikan categorical dan numerical columns
model_obj.cat_cols = [
    'person_gender',
    'person_education',
    'person_home_ownership',
    'loan_intent',
    'previous_loan_defaults_on_file'
]
model_obj.num_cols = [
    'person_age',
    'person_income',
    'person_emp_exp',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length',
    'credit_score'
]

# ===== Streamlit UI =====
st.title("🔍 Prediksi Status Pinjaman")
st.markdown("Masukkan data pemohon pinjaman untuk memprediksi statusnya.")

# Form Input Pengguna
input_data = {
    'person_age': st.number_input("Umur", min_value=18, max_value=100, value=30),
    'person_gender': st.selectbox("Jenis Kelamin", ['male', 'female']),
    'person_education': st.selectbox("Pendidikan", ['High School', 'Bachelor', 'Master']),
    'person_income': st.number_input("Pendapatan", value=50000.0),
    'person_emp_exp': st.number_input("Pengalaman Kerja (tahun)", value=5),
    'person_home_ownership': st.selectbox("Status Tempat Tinggal", ['RENT', 'OWN', 'MORTGAGE', 'OTHER']),
    'loan_amnt': st.number_input("Jumlah Pinjaman", value=10000.0),
    'loan_intent': st.selectbox("Tujuan Pinjaman", ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT']),
    'loan_int_rate': st.number_input("Bunga Pinjaman (%)", value=12.5),
    'loan_percent_income': st.number_input("Persentase Pendapatan untuk Pinjaman", value=0.25),
    'cb_person_cred_hist_length': st.number_input("Lama Histori Kredit", value=5),
    'credit_score': st.number_input("Skor Kredit", value=650),
    'previous_loan_defaults_on_file': st.selectbox("Pernah Gagal Bayar?", ['Yes', 'No'])
}

# Tombol Prediksi
if st.button("Prediksi Status"):
    user_df = pd.DataFrame([input_data])
    label, probas = model_obj.predict_input(user_df)
    label_map = {0: 'Default ❌', 1: 'Approved ✅'}

    st.subheader("Hasil Prediksi:")
    if label == 1:
        st.success(f"{label_map[label]} dengan probabilitas {probas[label]*100:.2f}%")
    else:
        st.error(f"{label_map[label]} dengan probabilitas {probas[label]*100:.2f}%")

    st.markdown("### Probabilitas:")
    st.write({f"{label_map[i]}": f"{probas[i]*100:.2f}%" for i in range(len(probas))})
