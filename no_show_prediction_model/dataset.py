from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer

from no_show_prediction_model.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "medical-appointments-no-show-en.csv",
    output_path: Path = PROCESSED_DATA_DIR / "clean_medical_appointments.csv",
):
    logger.info("Carregando base...")
    df = pd.read_csv(input_path)

    logger.info("Iniciando Limpeza...")

    # ============================================================
    # 1. Renomeando colunas
    # ============================================================
    rename_map = {
        'specialty': 'especialidade',
        'appointment_time': 'horario_consulta',
        'gender': 'genero',
        'appointment_date': 'data_consulta',
        'no_show': 'nao_compareceu',
        'no_show_reason': 'motivo_nao_comparecimento',
        'disability': 'deficiencia',
        'date_of_birth': 'data_nascimento',
        'entry_service_date': 'data_entrada_servico',
        'city': 'cidade',
        'icd': 'cid',
        'appointment_month': 'mes_consulta',
        'appointment_year': 'ano_consulta',
        'appointment_shift': 'turno_consulta',
        'age': 'idade',
        'under_12_years_old': 'menor_12_anos',
        'over_60_years_old': 'maior_60_anos',
        'patient_needs_companion': 'paciente_precisa_acompanhante',
        'average_temp_day': 'temperatura_media_dia',
        'average_rain_day': 'chuva_media_dia',
        'max_temp_day': 'temperatura_maxima_dia',
        'max_rain_day': 'chuva_maxima_dia',
        'rainy_day_before': 'dia_chuvoso_anterior',
        'storm_day_before': 'chuva_forte_dia_anterior',
        'rain_intensity': 'intensidade_chuva',
        'heat_intensity': 'intensidade_calor'
    }
    df = df.rename(columns=rename_map)
    logger.info("Renomeação concluída.")

    # ============================================================
    # 2. Remover colunas irrelevantes
    # ============================================================
    cols_remove = [
        "motivo_nao_comparecimento",
        "data_nascimento",
        "cid"
    ]
    df = df.drop(columns=[c for c in cols_remove if c in df.columns])
    logger.info("Colunas removidas.")

    # ============================================================
    # 3. Remover registros com idade vazia/inválida
    # ============================================================
    df["idade"] = df["idade"].replace(["", " ", None], pd.NA)
    df = df.dropna(subset=["idade"])
    logger.info("Registros com idade inválida removidos.")

    # ============================================================
    # 4. Imputação categórica
    # ============================================================
    imputar_desconhecido = [
        "cidade",
        "data_entrada_servico",
        "especialidade"
    ]

    for col in imputar_desconhecido:
        if col in df.columns:
            df[col] = df[col].fillna("desconhecido")

    if "deficiencia" in df.columns:
        df["deficiencia"] = df["deficiencia"].fillna("não informado")

    logger.info("Imputação categórica concluída.")

    # ============================================================
    # 5. Imputação de numéricos com média
    # ============================================================
    imputar_media = [
        "temperatura_media_dia",
        "chuva_media_dia",
        "temperatura_maxima_dia",
        "chuva_maxima_dia"
    ]

    for col in imputar_media:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    logger.info("Imputação numérica concluída.")

    # ============================================================
    # 6. Padronizando textos 
    # ============================================================
    cols_texto = df.select_dtypes(include=["object"]).columns

    for col in cols_texto:
        df[col] = df[col].astype(str).str.strip().str.lower()

    logger.info("Padronização de textos concluída.")

    # ============================================================
    # 7. Removendo gênero inválido "i"
    # ============================================================
    if "genero" in df.columns:
        antes = len(df)
        df = df[~df["genero"].isin(["i", "na"])]

        removidos = antes - len(df)
        logger.info(f"Registros removidos com gênero inválido: {removidos}")

    # ============================================================
    # 8. Convertendo data da consulta
    # ============================================================
    if "data_consulta" in df.columns:
        df["data_consulta"] = pd.to_datetime(df["data_consulta"], format='%d/%m/%Y', errors="coerce")

    # ============================================================
    # 9. Removendo duplicatas
    # ============================================================
    df = df.drop_duplicates()
    logger.info("Duplicatas removidas.")

    # ============================================================
    # 10. Salvando dataset processado
    # ============================================================
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.success(f"Dataset limpo salvo em: {output_path}")


if __name__ == "__main__":
    app()
