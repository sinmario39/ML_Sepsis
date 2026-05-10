
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
    # Pesi adattivi dinamici
    if sepsis_score > 0.7:
        ml_weight = 0.7
        rule_weight = 0.3
    else:
        ml_weight = 0.8
        rule_weight = 0.2

    # Fusione
    final_sepsis = (
            ml_weight * prob_sepsis +
            rule_weight * sepsis_score
    )

    # Rafforzamento clinico
    if sepsis_score > 0.6 and prob_sepsis > 0.5:
        final_sepsis += 0.1

    # Limitazione dei valori superiori ad 1
    final_sepsis = min(final_sepsis, 1.0)

    # -------------------------
    # PRIORITÀ CLINICHE
    # -------------------------

    # Sepsi
    # La Sepsi ha priorità clinica elevata
    if final_sepsis > 0.85:
        return "sepsis", {
            "confidence": final_sepsis,
            "reason": "Critical sepsis indicators detected"
        }
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
    # Rafforziamo la classe suggerita dal modello ML, evidenziando il contributo del modello statistico
    if macro_pred in other_scores:
        other_scores[macro_pred] += 0.1
    else:
        # Penalizzazione in caso di incoerenza tra le predizioni
        for k in other_scores:
            other_scores[k] *= 0.95

    # Penalizzazione dello stato "Stable" in caso di segnali clinici anomali
    if "stable" in other_scores:
        if any(v > 0.5 for k, v in other_scores.items() if k != "stable"):
            other_scores["stable"] *= 0.7

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

    # Caso incertezza (Caso di punteggi vicini)
    if gap < 0.1 and second_class is not None: # Il sistema restituisce due diagnosi probabili simulando l'incertezza medica
        return [best_class, second_class], {
            "confidence": best_score,
            "reason": "Low confidence - multiple possible conditions"
        }

    # Caso normale
    return best_class, {
        "confidence": best_score,
        "reason": "Score-based decision with ML support"
    }
