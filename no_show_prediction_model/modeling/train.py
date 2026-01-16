from pathlib import Path

import joblib
import pandas as pd
import typer
from loguru import logger

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

from xgboost import XGBClassifier

from no_show_prediction_model.config import PROCESSED_DATA_DIR, MODELS_DIR
from no_show_prediction_model.plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
)

app = typer.Typer()

THRESHOLD = 0.30


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "features_processed_medical_appointments.csv",
    model_path: Path = MODELS_DIR / "xgboost_no_show_final.joblib",
):
    logger.info("Carregando dataset...")
    df = pd.read_csv(input_path)

    target = "nao_compareceu"

    X = df.drop(columns=[target])
    y = df[target].replace({"no": 0, "yes": 1}).astype(int)

    logger.info(f"Distribuição do target:\n{y.value_counts()}")

    # ======================================================
    # Hold-out
    # ======================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # ======================================================
    # Colunas
    # ======================================================
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    num_cols = X_train.select_dtypes(exclude="object").columns.tolist()

    logger.info(f"Variáveis categóricas: {cat_cols}")
    logger.info(f"Variáveis numéricas: {num_cols}")

    # ======================================================
    # Pré-processamento
    # ======================================================
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
            ("num", "passthrough", num_cols),
        ]
    )

    # ======================================================
    # Modelo
    # ======================================================
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=y_train.value_counts()[0] / y_train.value_counts()[1],
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # ======================================================
    # Cross-Validation (somente treino)
    # ======================================================
    logger.info("Rodando Cross-Validation (ROC-AUC)...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )

    logger.success(f"ROC-AUC médio (CV): {cv_scores.mean():.4f}")

    # ======================================================
    # Avaliação em hold-out com threshold
    # ======================================================
    logger.info("Treinando modelo no conjunto de treino...")
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    logger.info(f"Avaliando com threshold = {THRESHOLD}")

    logger.success(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    logger.success(f"Precision: {precision_score(y_test, y_pred):.4f}")
    logger.success(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    logger.success(f"F1-score : {f1_score(y_test, y_pred):.4f}")
    logger.success(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")

    logger.info("Classification Report:")
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    # ======================================================
    # Plots
    # ======================================================
    plot_confusion_matrix(y_test, y_pred, threshold=THRESHOLD)
    plot_roc_curve(y_test, y_prob)
    plot_precision_recall_curve(y_test, y_prob)
    
    # ======================================================
    # Treinamento final (produção)
    # ======================================================
    logger.info("Treinando modelo final com 100% dos dados...")
    pipeline.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    logger.success(f"Modelo salvo em: {model_path}")


if __name__ == "__main__":
    app()
