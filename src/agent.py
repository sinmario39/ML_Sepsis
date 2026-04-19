from rules import compute_all_scores
from decision import make_decision
from explain import generate_explanation


class MedicalAgent:
    """
    Agente intelligente per supporto alla diagnosi medica.

    Integra:
    - Modello ML (sepsi + multiclasse)
    - Sistema di regole (score)
    - Decision engine
    - Spiegazione interpretabile
    """

    def __init__(self, model_wrapper):
        # model_wrapper: oggetto che gestisce predizione della sepsi e predizione multiclasse
        self.model = model_wrapper

    def evaluate(self, patient_data):
        """
        Esegue l'intero processo decisionale su un paziente.

        Input:
            patient_data: dict con feature cliniche

        Output:
            diagnosis: string oppure lista (top-2)
            explanation: stringa spiegazione
        """

        # -------------------------
        # 1. PREDIZIONE ML
        # -------------------------

        # Otteniamo probabilità di sepsi e predizione multiclasse
        prob_sepsis, macro_pred = self.model.predict(patient_data)

        # -------------------------
        # 2. SISTEMA A REGOLE
        # -------------------------

        # Calcolo punteggi per ogni condizione
        scores = compute_all_scores(patient_data)

        # -------------------------
        # 3. DECISION ENGINE
        # -------------------------

        # Combina ML e regole
        diagnosis, decision_info = make_decision(
            prob_sepsis,
            macro_pred,
            scores
        )

        # -------------------------
        # 4. SPIEGAZIONE
        # -------------------------

        explanation = generate_explanation(
            prob_sepsis,
            macro_pred,
            scores,
            patient_data,
            diagnosis,
            decision_info
        )

        # -------------------------
        # 5. OUTPUT FINALE
        # -------------------------

        return {
            "diagnosis": diagnosis,
            "confidence": decision_info["confidence"],
            "reason": decision_info["reason"],
            "explanation": explanation
        }