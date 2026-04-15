import math

# -------------------------
# Utility functions
# -------------------------

def safe_get(data, key):
    # Recupera valore gestendo NaN/mancanti
    val = data.get(key, None)
    if val is None:
        return None
    try:
        if math.isnan(val):
            return None
    except:
        pass
    return val

def add_score(condition, weight):
    return weight if condition else 0

def normalize_score(score, max_score):
    if max_score == 0:
        return 0
    return score / max_score


def age_modifier(data):
    age = safe_get(data, "Age")

    if age is None:
        return 0

    if age > 75:
        return 1
    elif age > 65:
        return 0.5
    else:
        return 0

# -------------------------
# SEPSIS SCORE
# -------------------------

def compute_sepsis_score(data):
    score = 0
    max_score = 17  # somma pesi

    temp = safe_get(data, "Temp")
    hr = safe_get(data, "HR")
    resp = safe_get(data, "Resp")
    wbc = safe_get(data, "WBC")
    creat = safe_get(data, "Creatinine")
    lactate = safe_get(data, "Lactate")
    sbp = safe_get(data, "SBP")
    platelets = safe_get(data, "Platelets")

    score += add_score(temp is not None and (temp > 38 or temp < 36), 2)
    score += add_score(hr is not None and hr > 100, 1)
    score += add_score(resp is not None and resp > 20, 2)
    score += add_score(wbc is not None and (wbc > 12000 or wbc < 4000), 2)
    score += add_score(creat is not None and creat > 1.5, 1)
    score += add_score(lactate is not None and lactate > 2, 3)
    score += add_score(sbp is not None and sbp < 90, 2)
    score += add_score(platelets is not None and platelets < 150000, 1)
    score += age_modifier(data)

    # Regola combinata delle soglie (Shock Settico)
    if temp is not None and hr is not None:
        score += add_score((temp > 38 or temp < 36) and hr > 100 and resp > 20 and sbp < 90, 3)

    return normalize_score(score, max_score)

# -------------------------
# RESPIRATORY SCORE
# -------------------------

def compute_respiratory_score(data):
    score = 0
    max_score = 8

    o2 = safe_get(data, "O2Sat")
    resp = safe_get(data, "Resp")
    hr = safe_get(data, "HR")

    score += add_score(o2 is not None and o2 < 92, 2)
    score += add_score(resp is not None and resp > 22, 2)
    score += add_score(hr is not None and hr > 100, 1)
    score += age_modifier(data)

    return normalize_score(score, max_score)

# -------------------------
# METABOLIC SCORE
# -------------------------

def compute_metabolic_score(data):
    score = 0
    max_score = 8

    glucose = safe_get(data, "Glucose")
    creat = safe_get(data, "Creatinine")
    lactate = safe_get(data, "Lactate")
    bun = safe_get(data, "BUN")

    score += add_score(glucose is not None and (glucose > 125 or glucose < 55), 3)
    score += add_score(creat is not None and creat > 1.2, 2)
    score += add_score(lactate is not None and lactate > 2, 1)
    score += add_score(bun is not None and bun > 20, 1)
    score += age_modifier(data)

    return normalize_score(score, max_score)

# -------------------------
# HEMODYNAMIC SCORE
# -------------------------

def compute_hemodynamic_score(data):
    score = 0
    max_score = 8

    hr = safe_get(data, "HR")
    sbp = safe_get(data, "SBP")
    dbp = safe_get(data, "DBP")
    map_val = safe_get(data, "MAP")

    score += add_score(hr is not None and (hr > 100 or hr < 60), 2)
    score += add_score(sbp is not None and (sbp > 140 or sbp < 90), 1)
    score += add_score(dbp is not None and (dbp > 90 or dbp < 60), 1)
    score += add_score(map_val is not None and (map_val < 65 or map_val > 100), 2)
    if sbp is not None and hr is not None:
        score += add_score(sbp < 90 and hr > 100, 1)
    score += age_modifier(data)

    return normalize_score(score, max_score)

# -------------------------
# STABLE SCORE
# -------------------------

def compute_stable_score(data):
    score = 0
    max_score = 7

    temp = safe_get(data, "Temp")
    hr = safe_get(data, "HR")
    o2 = safe_get(data, "O2Sat")
    glucose = safe_get(data, "Glucose")
    wbc = safe_get(data, "WBC")
    map_val = safe_get(data, "MAP")
    resp = safe_get(data, "Resp")

    score += add_score(temp is not None and 36 <= temp <= 37.5, 1)
    score += add_score(hr is not None and (hr >= 60 or hr <= 100), 1)
    score += add_score(o2 is not None and o2 > 95, 1)
    score += add_score(glucose is not None and (glucose > 70 or glucose < 125), 1)
    score += add_score(wbc is not None and 4000 <= wbc <= 12000, 1)
    score += add_score(map_val is not None and 70 <= map_val < 90, 1)
    score += add_score(resp is not None and resp < 20, 1)

    return normalize_score(score, max_score)

# -------------------------
# INTERFACE
# -------------------------

def compute_all_scores(data):
    return {
        "sepsis": compute_sepsis_score(data),
        "respiratory": compute_respiratory_score(data),
        "metabolic": compute_metabolic_score(data),
        "hemodynamic": compute_hemodynamic_score(data),
        "stable": compute_stable_score(data),
    }