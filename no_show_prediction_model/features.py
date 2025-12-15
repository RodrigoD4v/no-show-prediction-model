from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm
import typer

from no_show_prediction_model.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "clean_medical_appointments.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features_processed_medical_appointments.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Carregando a Base...")
    df = pd.read_csv(input_path)
    
    logger.info("Gerando features dataset...")
   
    # ============================================================
    # 1. Criando Features
    # ============================================================
    
    # 1.1 Features temporais básicas
    df["data_consulta"] = pd.to_datetime(df["data_consulta"], dayfirst=True, errors="coerce")
    df['data_primeira_consulta'] = pd.to_datetime(df['data_primeira_consulta'], dayfirst=True, errors='coerce')
    
    df["antiguidade_paciente"] = (df["data_consulta"] - df["data_primeira_consulta"]).dt.days
    df["antiguidade_paciente"] = (
    df["data_consulta"] - df["data_primeira_consulta"]).dt.days.fillna(0)

    df["dia_da_semana"] = df["data_consulta"].dt.weekday
    df["semana_do_ano"] = df["data_consulta"].dt.isocalendar().week.astype(int)
    df["dia_do_ano"] = df["data_consulta"].dt.dayofyear

    # 1.2 Faixa etária
    df['faixa_etaria'] = pd.cut(
    df['idade'],
    bins=[0,12,18,30,45,60,120],
    labels=['0-12','13-18','19-30','31-45','46-60','61+']
)
    
    # 1.3 Meses númericos
    mapa_meses = {
    'jan':1,'feb':2,'mar':3,'april':4,'may':5,'june':6,
    'july':7,'aug':8,'sept':9,'oct':10,'nov':11,'dec':12
    }
    df["mes_num"] = df["mes_consulta"].map(mapa_meses)
    df["mes_sin"] = np.sin(2 * np.pi * df["mes_num"]/12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes_num"]/12)
    
    # Removendo coluns desnecessárias
    df = df.drop([
        "menor_12_anos",
        "maior_60_anos",
        "data_consulta",
        "mes_consulta",
        "mes_num",
        "data_primeira_consulta",
    ], axis=1)
    
    # ============================================================
    # 3. Salvando dataset final de features
    # ============================================================
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.success(f"Features dataset salvo em: {output_path}")
    # -----------------------------------------

if __name__ == "__main__":
    app()
