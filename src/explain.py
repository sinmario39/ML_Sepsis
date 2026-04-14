
def generate_explanation(prob_sepsis, macro_pred, scores, patient_data, decision_output):
    """
    decision_output: può essere stringa o lista (top-2)
    """

    clinical_findings = []

    # -------------------------
    # ESTRAZIONE FEATURES
    # -------------------------

    temp = patient_data.get("Temp")
    hr = patient_data.get("HR")
    resp = patient_data.get("Resp")
    o2 = patient_data.get("O2Sat")
    wbc = patient_data.get("WBC")
    glucose = patient_data.get("Glucose")
    creat = patient_data.get("Creatinine")
    lactate = patient_data.get("Lactate")
    sbp = patient_data.get("SBP")
    dbp = patient_data.get("DBP")
    map =patient_data.get("MAP")
    platelets = patient_data.get("Platelets")
    bun = patient_data.get("BUN")

    # -------------------------
    # COSTRUZIONE EVIDENZE CLINICHE
    # -------------------------

    if temp is not None and (temp > 38 or temp < 36):
        clinical_findings.append("temperatura anomala: ")
        if temp > 38: clinical_findings.append("febbre")
        if temp < 36: clinical_findings.append("ipotermia")

    if hr is not None and hr > 100:
        clinical_findings.append("tachicardia")
    if hr is not None and hr < 60:
        clinical_findings.append("bradicardia")

    if resp is not None and resp > 20:
        clinical_findings.append("frequenza respiratoria elevata")

    if o2 is not None and o2 < 92:
        clinical_findings.append("ipossia")

    if wbc is not None and (wbc > 12000 or wbc < 4000):
        clinical_findings.append("livello di globuli bianchi alterato")

    if platelets is not None and platelets < 150000:
        clinical_findings.append("livello di piastrine basso")

    if glucose is not None and glucose > 125: clinical_findings.append("iperglicemia")
    if glucose is not None and glucose < 55: clinical_findings.append("ipoglicemia")

    if creat is not None and creat > 1.2:
        clinical_findings.append("livelli di creatinina elevati")

    if lactate is not None and lactate > 2:
        clinical_findings.append("livelli di lattato elevati")

    if sbp is not None and sbp > 140:
        clinical_findings.append("pressione sistolica elevata")
    if sbp is not None and sbp < 90:
        clinical_findings.append("pressione sistolica bassa")
    if dbp is not None and dbp > 90:
        clinical_findings.append("pressione diastolica elevata")
    if dbp is not None and dbp < 60:
        clinical_findings.append("pressione diastolica bassa")

    if map is not None and map > 100: clinical_findings.append("pressione media elevata - rischio di ipertensione")
    if resp is not None and map < 65: clinical_findings.append("pressione media bassa - rischio di ipoperfusione")

    if bun is not None and bun > 20:
        clinical_findings.append("quantità elevata di azoto ureico")

    # -------------------------
    # INTERPRETAZIONE DECISIONE
    # -------------------------

    if isinstance(decision_output, list):
        diagnosis_text = f"Possibili diagnosi: {decision_output[0]} o {decision_output[1]}"
        uncertainty_text = "Il sistema ha rilevato incertezza tra più condizioni."
    else:
        diagnosis_text = f"Diagnosi più probabile: {decision_output}"
        uncertainty_text = ""

    # -------------------------
    # COSTRUZIONE SPIEGAZIONE
    # -------------------------

    explanation = diagnosis_text + ".\n\n"

    if clinical_findings:
        explanation += "Evidenze cliniche rilevate:\n"
        for f in clinical_findings:
            explanation += f"- {f}\n"

    explanation += "\n"

    explanation += f"Probabilità stimata di sepsi: {round(prob_sepsis, 2)}.\n"
    explanation += f"La decisione combina modello di machine learning e regole cliniche.\n"

    if macro_pred:
        explanation += f"Il modello multiclasse suggerisce: {macro_pred}.\n"

    if uncertainty_text:
        explanation += "\n" + uncertainty_text

    return explanation