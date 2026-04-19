
def generate_explanation(prob_sepsis, macro_pred, scores, patient_data, decision_output, decision_info):

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
    map_val = patient_data.get("MAP")
    platelets = patient_data.get("Platelets")
    bun = patient_data.get("BUN")

    # -------------------------
    # EVIDENZE CLINICHE
    # -------------------------

    if temp is not None:
        if temp > 38:
            clinical_findings.append(f"febbre (Temp = {temp}°C)")
        elif temp < 36:
            clinical_findings.append(f"ipotermia (Temp = {temp}°C)")

    if hr is not None:
        if hr > 100:
            clinical_findings.append(f"tachicardia (HR = {hr} bpm)")
        elif hr < 60:
            clinical_findings.append(f"bradicardia (HR = {hr} bpm)")

    if resp is not None and resp > 20:
        clinical_findings.append("frequenza respiratoria elevata")

    if o2 is not None and o2 < 92:
        clinical_findings.append("ipossia")

    if wbc is not None and (wbc > 12000 or wbc < 4000):
        clinical_findings.append("globuli bianchi alterati")

    if platelets is not None and platelets < 150000:
        clinical_findings.append("piastrine basse")

    if glucose is not None:
        if glucose > 125:
            clinical_findings.append("iperglicemia")
        elif glucose < 55:
            clinical_findings.append("ipoglicemia")

    if creat is not None and creat > 1.2:
        clinical_findings.append("creatinina elevata")

    if lactate is not None and lactate > 2:
        clinical_findings.append(f"lattato elevato (Lactate = {lactate})")

    if sbp is not None:
        if sbp > 140:
            clinical_findings.append(f"ipertensione sistolica (SBP = {sbp} mmHg)")
        elif sbp < 90:
            clinical_findings.append(f"ipotensione sistolica (SBP = {sbp} mmHg)")

    if dbp is not None:
        if dbp > 90:
            clinical_findings.append(f"ipertensione diastolica (DBP = {dbp} mmHg)")
        elif dbp < 60:
            clinical_findings.append(f"ipotensione diastolica (DBP = {dbp} mmHg)")

    if map_val is not None:
        if map_val > 100:
            clinical_findings.append("pressione media elevata")
        elif map_val < 65:
            clinical_findings.append("possibile ipoperfusione")

    if bun is not None and bun > 20:
        clinical_findings.append("BUN elevato")

    # Regola combinata (Shock Settico)
    if sbp is not None and hr is not None:
        if (temp > 38 or temp < 36) and hr > 100 and resp > 20 and sbp < 90:
            clinical_findings.append("stato di shock settico")



    # -------------------------
    # DECISIONE
    # -------------------------

    if isinstance(decision_output, list):
        diagnosis_text = f"Possibili diagnosi: {decision_output[0]} o {decision_output[1]}"
        uncertainty_text = "Il sistema ha rilevato incertezza tra più condizioni."
    else:
        diagnosis_text = f"Diagnosi più probabile: {decision_output}"
        uncertainty_text = ""

    # -------------------------
    # OUTPUT
    # -------------------------

    explanation = diagnosis_text + ".\n\n"

    if clinical_findings:
        explanation += "Evidenze cliniche rilevate:\n"
        for f in clinical_findings:
            explanation += f"- {f}\n"

    explanation += "\n"
    explanation += f"Probabilità stimata di sepsi: {round(prob_sepsis, 2)}.\n"
    explanation += "La decisione combina modello di machine learning e regole cliniche.\n"

    if macro_pred:
        explanation += f"Il modello multiclasse suggerisce: {macro_pred}.\n"

    if uncertainty_text:
        explanation += "\n" + uncertainty_text

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    explanation += "\nCondizioni più probabili secondo il sistema a regole:\n"
    for k, v in sorted_scores[:2]:
        explanation += f"- {k} (score: {round(v, 2)})\n"

    if decision_info:
        explanation += f"\nMotivazione decisione: {decision_info.get('reason', '')}\n"
        explanation += f"Confidenza: {round(decision_info.get('confidence', 0), 2)}\n"

    return explanation