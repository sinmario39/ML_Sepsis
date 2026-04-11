import joblib

class MLModelWrapper:

    def __init__(self, sepsis_path, macro_path, preprocessor):
        self.model_sepsis = joblib.load(sepsis_path)
        self.model_macro = joblib.load(macro_path)
        self.preprocessor = preprocessor

    def predict(self, patient_data):
        X = self.preprocessor.transform(patient_data)

        prob_sepsis = self.model_sepsis.predict_proba(X)[0][1]
        macro_pred = self.model_macro.predict(X)[0]

        return prob_sepsis, macro_pred