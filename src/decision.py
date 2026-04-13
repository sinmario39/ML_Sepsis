
def normalize(value, min_val=0, max_val=1):
    if max_val - min_val == 0:
        return 0
    return (value - min_val) / (max_val - min_val)

def make_decision(prob_sepsis, macro_pred, scores):
    """
    prob_sepsis: float (0-1)
    macro_pred: string (es. 'respiratory')
    scores: dict con punteggi normalizzati
    """

    # -------------------------
    # FUSIONE SEPSI
    # -------------------------

    sepsis_score = scores.get("sepsis", 0)
    final_sepsis = 0.8 * prob_sepsis + 0.2 * sepsis_score

    # -------------------------
    # PRIORITÀ CLINICHE
    # -------------------------

    # Sepsi
    if final_sepsis > 0.7:
        return "sepsis", {
            "confidence": final_sepsis,
            "reason": "High sepsis risk"
        }

    # Altre classi
    other_scores = {
        k: v for k, v in scores.items() if k != "sepsis" # Prende "scores" e rimuove sepsis
    }
    # Boost della predizione ML multiclasse
    if macro_pred in other_scores:
        other_scores[macro_pred] += 0.1

    # -------------------------
    # SCELTA MIGLIORE
    # -------------------------

    best_class = max(other_scores, key=other_scores.get)
    best_score = other_scores[best_class]

    # Gestione incertezza
    sorted_scores = sorted(other_scores.values(), reverse=True)

    if len(sorted_scores) > 1:
        gap = sorted_scores[0] - sorted_scores[1]
    else:
        gap = 1

    if gap < 0.1:
        return best_class, {
            "confidence": best_score,
            "reason": "Low confidence - multiple possible conditions"
        }

    # Caso normale
    return best_class, {
        "confidence": best_score,
        "reason": "Score-based decision with ML support"
    }
