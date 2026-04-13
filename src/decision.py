
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

    # Ordina le classi per punteggio
    sorted_classes = sorted(other_scores.items(), key=lambda x: x[1], reverse=True)
    best_class, best_score = sorted_classes[0]

    # Seconda migliore (se esiste)
    second_class, second_score = (None, None)
    if len(sorted_classes) > 1:
        second_class, second_score = sorted_classes[1]

    # Gap tra primo e secondo
    gap = best_score - (second_score if second_score is not None else 0)

    # Caso incertezza
    if gap < 0.1 and second_class is not None:
        return [best_class, second_class], {
            "confidence": best_score,
            "reason": "Low confidence - multiple possible conditions"
        }

    # Caso normale
    return best_class, {
        "confidence": best_score,
        "reason": "Score-based decision with ML support"
    }
