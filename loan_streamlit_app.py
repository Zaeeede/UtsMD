import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from xgboost import XGBClassifier

# ==================== Class ModelXGB ====================
class ModelXGB:
    def __init__(self, data, loaded_model=None):
        self.data = pd.read_csv(data)
        self.fitted_model = loaded_model

    def data_split(self, target_column):
        self.X = self.data.drop([target_column], axis=1)
        self.y = self.data[target_column]
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)

    def correct_values(self):
        self.X_train['person_gender'] = self.X_train['person_gender'].replace({'fe male': 'female', 'Male': 'male'})
        self.X_test['person_gender'] = self.X_test['person_gender'].replace({'fe male': 'female', 'Male': 'male'})

    def data_preprocessing(self, scaler, encoder, load_from_pickle=False):
        self.scaler = scaler
        self.encoder = encoder

        self.correct_values()

        median_income = np.median(self.X_train['person_income'])
        self.X_train['person_income'].fillna(median_income, inplace=True)
        self.X_test['person_income'].fillna(median_income, inplace=True)

        self.num_cols = self.X_train.select_dtypes(include='number').columns
        self.cat_cols = self.X_train.select_dtypes(include='object').columns

        X_train_encoded = self.encoder.fit_transform(self.X_train[self.cat_cols])
        X_test_encoded = self.encoder.transform(self.X_test[self.cat_cols])

        X_train_encoded_df = pd.DataFrame(
            X_train_encoded, columns=self.encoder.get_feature_names_out(self.cat_cols), index=self.X_train.index)
        X_test_encoded_df = pd.DataFrame(
            X_test_encoded, columns=self.encoder.get_feature_names_out(self.cat_cols), index=self.X_test.index)

        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train[self.num_cols]),
            columns=self.num_cols, index=self.X_train.index)
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test[self.num_cols]),
            columns=self.num_cols, index=self.X_test.index)

        self.X_train = pd.concat([X_train_scaled, X_train_encoded_df], axis=1)
        self.X_test = pd.concat([X_test_scaled, X_test_encoded_df], axis=1)

        if not load_from_pickle:
            self.feature_names = self.X_train.columns.tolist()

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

# ==================== Load Pickles ====================
@st.cache_resource
def load_assets():
    with open("model.pkl", "rb") as f:
        model = pkl.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pkl.load(f)
    with open("encoder.pkl", "rb") as f:
        encoder = pkl.load(f)
    with open("feature_names.pkl", "rb") as f:
        feature_names = pkl.load(f)
    return model, scaler, encoder, feature_names

fitted_model, scaler, encoder, feature_names = load_assets()

# Buat instance class dengan data dummy
model_obj = ModelXGB(data="Dataset_A_loan.csv", loaded_model=fitted_model)
model_obj.data_split(target_column='loan_status')
model_obj.data_preprocessing(scaler, encoder, load_from_pickle=True)
model_obj.feature_names = feature_names

# ==================== UI Streamlit ====================
st.title("💰 Prediksi Status Pinjaman")
st.markdown("Masukkan data pemohon untuk memprediksi apakah pinjaman akan disetujui atau gagal bayar.")

# ====== Input Form ======
user_input = {
    'person_age': st.number_input("Umur", min_value=18, max_value=100, value=30),
    'person_gender': st.selectbox("Jenis Kelamin", ['male', 'female']),
    'person_education': st.selectbox("Pendidikan", ['High School', 'Bachelor', 'Master']),
    'person_income': st.number_input("Pendapatan", value=50000.0),
    'person_emp_exp': st.number_input("Pengalaman Kerja", value=5),
    'person_home_ownership': st.selectbox("Status Rumah", ['RENT', 'OWN', 'MORTGAGE', 'OTHER']),
    'loan_amnt': st.number_input("Jumlah Pinjaman", value=10000.0),
    'loan_intent': st.selectbox("Tujuan Pinjaman", ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT']),
    'loan_int_rate': st.number_input("Bunga Pinjaman (%)", value=12.5),
    'loan_percent_income': st.number_input("Rasio Pendapatan untuk Pinjaman", value=0.25),
    'cb_person_cred_hist_length': st.number_input("Lama Riwayat Kredit (tahun)", value=5),
    'credit_score': st.number_input("Skor Kredit", value=650),
    'previous_loan_defaults_on_file': st.selectbox("Gagal Bayar Sebelumnya?", ['Yes', 'No'])
}

if st.button("🔍 Prediksi"):
    user_df = pd.DataFrame([user_input])
    label, probas = model_obj.predict_input(user_df)

    label_map = {0: 'Default ❌', 1: 'Disetujui ✅'}
    st.subheader("📊 Hasil Prediksi:")
    if label == 1:
        st.success(f"{label_map[label]} dengan probabilitas {probas[label]*100:.2f}%")
    else:
        st.error(f"{label_map[label]} dengan probabilitas {probas[label]*100:.2f}%")

    st.markdown("### 🔢 Probabilitas:")
    for i, p in enumerate(probas):
        st.write(f"{label_map[i]}: {p*100:.2f}%")
