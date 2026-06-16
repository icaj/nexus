# esg_ml/adaptadores/saida/csv_resultado_adapter.py
# Porta: Notificador → CSV/Excel
import os
from datetime import datetime
from typing import List
import pandas as pd
from esg_ml.dominio.entidades.diagnostico import DiagnosticoESG

class CsvResultadoAdapter:
    def __init__(self, caminho_saida: str = None):
        if caminho_saida:
            self.caminho = caminho_saida
        else:
            pasta = './data/resultados'
            os.makedirs(pasta, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.caminho = os.path.join(pasta, f'diagnostico_esg_{ts}.csv')

    def publicar(self, diagnosticos: List[DiagnosticoESG]) -> None:
        if not diagnosticos:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.caminho)), exist_ok=True)
        df = pd.DataFrame([d.to_dict() for d in diagnosticos])
        if self.caminho.lower().endswith('.xlsx'):
            df.to_excel(self.caminho, index=False)
        else:
            df.to_csv(self.caminho, index=False, encoding='utf-8-sig')
        print(f'\n  Resultados salvos: {os.path.abspath(self.caminho)}')
