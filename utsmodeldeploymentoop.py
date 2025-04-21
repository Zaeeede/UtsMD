import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle as pkl

class ModelXGB:
    def __init__(self, data, loaded_model=None):
        self.data = pd.read_csv(data)
        self.fitted_model = loaded_model

    def data_split(self, target_column):
        self.X = self.data.drop([target_column], axis=1)
        self.y = self.data[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def createMedianFromColumn(self, kolom):
        return np.median(self.X_train[kolom])

    def fillingNAWithNumbers(self, kolom, number):
        self.X_train[kolom].fillna(number, inplace=True)
        self.X_test[kolom].fillna(number, inplace=True)

    def correct_values(self):
        self.X_train['person_gender'] = self.X_train['person_gender'].replace({'fe male': 'female', 'Male': 'male'})
        self.X_test['person_gender'] = self.X_test['person_gender'].replace({'fe male': 'female', 'Male': 'male'})

    def export_feature_names(self, filename):
        with open(filename, 'wb') as f:
            pkl.dump(self.feature_names, f)

    def data_preprocessing(self, scaler, encoder, load_from_pickle=False):
        self.encoder = encoder
        self.scaler = scaler

        self.correct_values()

        numerical_cols_all = self.X_train.select_dtypes(include=np.number).columns
        for col in numerical_cols_all:
            median_val = self.createMedianFromColumn(col)
            self.fillingNAWithNumbers(col, median_val)

        categories = sorted(self.y.unique())
        self.target_vals = {label: idx for idx, label in enumerate(categories)}
        self.y_train = self.y_train.map(self.target_vals)
        self.y_test = self.y_test.map(self.target_vals)

        categorical_cols = self.X_train.select_dtypes(include='object').columns
        numerical_cols = self.X_train.select_dtypes(include=np.number).columns

        X_train_encoded = pd.DataFrame(
            self.encoder.fit_transform(self.X_train[categorical_cols]),
            columns=self.encoder.get_feature_names_out(),
            index=self.X_train.index
        )
        X_test_encoded = pd.DataFrame(
            self.encoder.transform(self.X_test[categorical_cols]),
            columns=self.encoder.get_feature_names_out(),
            index=self.X_test.index
        )

        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train[numerical_cols]),
            columns=self.X_train[numerical_cols].columns,
            index=self.X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test[numerical_cols]),
            columns=self.X_test[numerical_cols].columns,
            index=self.X_test.index
        )

        self.X_train = pd.concat([X_train_encoded, X_train_scaled], axis=1)
        self.X_test = pd.concat([X_test_encoded, X_test_scaled], axis=1)

        if not load_from_pickle:
            self.feature_names = self.X_train.columns.tolist()

        return self.target_vals

    def train(self):
        self.fitted_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        self.fitted_model.fit(self.X_train, self.y_train)

    def tune_hyperparameters(self, param_grid):
        grid_search = GridSearchCV(
            estimator=XGBClassifier(),
            param_grid=param_grid,
            scoring='accuracy',
            cv=5
        )
        grid_search.fit(self.X_train, self.y_train)
        self.fitted_model = grid_search.best_estimator_
        print("Best Parameters:", grid_search.best_params_)

    def evaluate(self, pred):
        print(f'Classification Report:\n{classification_report(self.y_test, pred)}')

    def test(self):
        if self.fitted_model is None:
            raise ValueError("No Model Found!")
        self.X_test = self.X_test[self.feature_names]
        pred = self.fitted_model.predict(self.X_test)
        self.evaluate(pred)

    def export_model(self, filename):
        if self.fitted_model is None:
            raise ValueError("No Model Found!")
        with open(filename, 'wb') as file:
            pkl.dump(self.fitted_model, file)

    def export_scaler(self, filename):
        with open(filename, 'wb') as file:
            pkl.dump(self.scaler, file)

    def export_encoder(self, filename):
        with open(filename, 'wb') as file:
            pkl.dump(self.encoder, file)

    def export_target_vals(self, filename):
        with open(filename, 'wb') as file:
            pkl.dump(self.target_vals, file)

    def predict_input(self, input_data):
        if self.fitted_model is None:
            raise ValueError("Model belum dimuat!")

        categorical_cols = self.encoder.feature_names_in_
        numerical_cols = self.scaler.feature_names_in_

        encoded_input = pd.DataFrame(
            self.encoder.transform(input_data[categorical_cols]),
            columns=self.encoder.get_feature_names_out(),
            index=input_data.index
        )
        scaled_input = pd.DataFrame(
            self.scaler.transform(input_data[numerical_cols]),
            columns=numerical_cols,
            index=input_data.index
        )

        processed_input = pd.concat([encoded_input, scaled_input], axis=1)
        processed_input = processed_input.reindex(columns=self.feature_names, fill_value=0)

        prediction = self.fitted_model.predict(processed_input)[0]
        prediction_proba = self.fitted_model.predict_proba(processed_input)[0]

        inverse_target_vals = {v: k for k, v in self.target_vals.items()}
        label = inverse_target_vals[prediction]

        return label, prediction_proba


# Inisialisasi encoder dan scaler
scaler = RobustScaler()
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Buat objek model
model = ModelXGB(data='Dataset_A_loan.csv')

# Split data
model.data_split(target_column='loan_status')

# Preprocessing
model.data_preprocessing(scaler, encoder)

# (Opsional) Hyperparameter tuning
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300]
}
model.tune_hyperparameters(param_grid)

# Latih model (jika tidak tuning, gunakan ini)
# model.train()

# Uji model
model.test()

# Prediksi input baru
new_input = pd.DataFrame([{
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
}])

label, probas = model.predict_input(new_input)
print(f"Prediksi: {label}")
print("Probabilitas per kelas:", probas)

# Ekspor semua komponen
model.export_model('model_xgb.pkl')
model.export_scaler('scaler.pkl')
model.export_encoder('encoder.pkl')
model.export_target_vals('target_vals.pkl')
model.export_feature_names('feature_names.pkl')

