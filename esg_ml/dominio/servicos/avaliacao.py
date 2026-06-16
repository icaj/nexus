# esg_ml/dominio/servicos/avaliacao.py
# CRISP-DM Fase 5 — Critério formal: acc>=93%, F1-Medium>=88%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

LIMIAR_ACURACIA  = 0.93
LIMIAR_F1_MEDIUM = 0.88

class ModeloInsuficienteError(Exception):
    """Lançada quando modelo não atinge o critério mínimo — retornar à Fase 4."""
    pass

def calcular_metricas(y_test, y_pred, le_target) -> dict:
    classes  = list(le_target.classes_)
    idx_high = classes.index('High')   if 'High'   in classes else 0
    idx_med  = classes.index('Medium') if 'Medium' in classes else 1
    return {
        'accuracy':        round(accuracy_score(y_test, y_pred), 4),
        'precision_macro': round(precision_score(y_test, y_pred, average='macro', zero_division=0), 4),
        'recall_macro':    round(recall_score(y_test, y_pred,    average='macro', zero_division=0), 4),
        'f1_macro':        round(f1_score(y_test, y_pred,        average='macro', zero_division=0), 4),
        'f1_high':         round(f1_score(y_test, y_pred, labels=[idx_high], average='macro', zero_division=0), 4),
        'f1_medium':       round(f1_score(y_test, y_pred, labels=[idx_med],  average='macro', zero_division=0), 4),
    }

def validar_criterio(nome: str, metricas: dict) -> None:
    erros = []
    if metricas['accuracy']  < LIMIAR_ACURACIA:
        erros.append(f"Acurácia {metricas['accuracy']:.2%} < mínimo {LIMIAR_ACURACIA:.0%}")
    if metricas['f1_medium'] < LIMIAR_F1_MEDIUM:
        erros.append(f"F1-Medium {metricas['f1_medium']:.2%} < mínimo {LIMIAR_F1_MEDIUM:.0%}")
    if erros:
        raise ModeloInsuficienteError(
            f"[{nome}] Reprovado — retornar à Fase 4:\n" +
            '\n'.join(f'  • {e}' for e in erros))

def avaliar_modelos(resultado: dict) -> dict:
    le_target = resultado['le_target']
    y_test    = resultado['y_test']
    m_knn = calcular_metricas(y_test, resultado['y_pred_knn'], le_target)
    m_rf  = calcular_metricas(y_test, resultado['y_pred_rf'],  le_target)
    validar_criterio('KNN',          m_knn)
    validar_criterio('RandomForest', m_rf)
    return {'knn': m_knn, 'rf': m_rf}
