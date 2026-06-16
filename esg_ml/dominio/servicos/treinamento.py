# esg_ml/dominio/servicos/treinamento.py
# CRISP-DM Fase 4 — GridSearchCV: KNN (k=3..21) + RF (24 combinações)
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

FEATURES = ['environment_score','social_score','governance_score','industry_enc']
TARGET   = 'total_level'

def preparar_features(df: pd.DataFrame) -> tuple:
    df = df.copy()
    le_ind = LabelEncoder()
    df['industry_enc'] = le_ind.fit_transform(df['industry'])
    X = df[FEATURES].values
    y = df[TARGET].values
    le_target = LabelEncoder()
    y_enc = le_target.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)
    return X_train, X_test, y_train, y_test, le_ind, le_target, df

def treinar_knn(X_train, y_train, cv) -> tuple:
    """Grid Search KNN k=3..21 (ímpares) com StandardScaler."""
    pipe = Pipeline([('scaler',StandardScaler()),('knn',KNeighborsClassifier(metric='euclidean'))])
    grid = GridSearchCV(pipe, {'knn__n_neighbors': list(range(3,22,2))},
                        cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_, grid.cv_results_

def treinar_random_forest(X_train, y_train, cv) -> tuple:
    """Grid Search RF: 2 × 4 × 3 = 24 combinações."""
    pipe = Pipeline([('rf', RandomForestClassifier(random_state=42, n_jobs=-1))])
    param_grid = {
        'rf__n_estimators':     [100, 200],
        'rf__max_depth':        [4, 6, 8, None],
        'rf__min_samples_leaf': [5, 10, 15],
    }
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    importancias = sorted(
        zip(FEATURES, grid.best_estimator_.named_steps['rf'].feature_importances_),
        key=lambda x: -x[1])
    return grid.best_estimator_, grid.best_params_, grid.best_score_, importancias

def executar_treino(df: pd.DataFrame) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_train, X_test, y_train, y_test, le_ind, le_target, df = preparar_features(df)
    knn_model, knn_params, knn_cv, knn_cv_results = treinar_knn(X_train, y_train, cv)
    rf_model,  rf_params,  rf_cv,  importancias   = treinar_random_forest(X_train, y_train, cv)
    return {
        'knn_model': knn_model, 'rf_model': rf_model,
        'knn_params': knn_params, 'rf_params': rf_params,
        'knn_cv_score': knn_cv, 'rf_cv_score': rf_cv,
        'knn_cv_results': knn_cv_results, 'importancias': importancias,
        'X_test': X_test, 'y_test': y_test,
        'y_pred_knn': knn_model.predict(X_test),
        'y_pred_rf':  rf_model.predict(X_test),
        'le_ind': le_ind, 'le_target': le_target,
        'df': df, 'FEATURES': FEATURES,
    }
