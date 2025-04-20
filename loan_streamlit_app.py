import streamlit as st
import pandas as pd
import pickle as pkl
from utsmodeldeploymentoop import ModelXGB  # asumsi class disimpan di file ModelXGB.py

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

# ===== Setup class dummy for prediction only =====
model_obj = ModelXGB(data='Dataset_A_loan.csv', loaded_model=fitted_model)
model_obj.data_split(target_column='loan_status')
model_obj.data_preprocessing(scaler, encoder, load_from_pickle=True)
model_obj.feature_names = feature_names  # load dari pickle

# ===== Streamlit UI =====
st.title("Prediksi Status Pinjaman")
st.markdown("Masukkan data pemohon pinjaman untuk memprediksi statusnya.")

# ===== Form Input dari Pengguna =====
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

# Jika tombol ditekan
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
