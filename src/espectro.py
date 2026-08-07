# -*- coding: utf-8 -*-
"""
espectro.py
Utilidades espectrais compartilhadas entre etapas que consomem a saída da
etapa 03 (Etapas/FFT) para montar uma matriz condição x frequência — hoje
usadas pela etapa 05 (heatmap 2D) e pela etapa 07 (waterfall 3D).

Mantido separado de pipeline_io.py (que é sobre leitura/escrita de
grupos/parquet em geral) porque isso aqui é especificamente sobre a
matemática de "várias condições, mesmo grid de frequência, mesma escala de
cor" — o núcleo comum entre qualquer visualização espectral multi-condição.
"""

import csv
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d


def sanitizar_nome(nome: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(nome)).strip("_") or "canal"


def ler_metadados_condicoes(caminho_csv: Path) -> dict:
    """
    Lê o CSV opcional de metadados por condição (mesmo formato usado pela
    etapa 05, documentado no README):
    condicao,f_vfd_hz,vazao_m3h,reducao_shaft,reducao_cavidade.

    Retorna: dict condicao -> {"f_vfd_hz": float|None, "vazao_m3h": float|None,
                                "reducao_shaft": float|None, "reducao_cavidade": float|None}
    """
    metadados = {}
    with open(caminho_csv, newline="", encoding="utf-8-sig") as f:
        leitor = csv.DictReader(f)
        colunas_esperadas = {"condicao", "f_vfd_hz", "vazao_m3h", "reducao_shaft", "reducao_cavidade"}
        if not colunas_esperadas.issubset(set(leitor.fieldnames or [])):
            faltando = colunas_esperadas - set(leitor.fieldnames or [])
            raise ValueError(f"CSV de metadados sem as colunas esperadas: {faltando}")

        def _float_ou_none(valor):
            valor = (valor or "").strip()
            if not valor:
                return None
            return float(valor)

        for linha in leitor:
            condicao = linha["condicao"].strip()
            if not condicao:
                continue
            metadados[condicao] = {
                "f_vfd_hz": _float_ou_none(linha.get("f_vfd_hz")),
                "vazao_m3h": _float_ou_none(linha.get("vazao_m3h")),
                "reducao_shaft": _float_ou_none(linha.get("reducao_shaft")),
                "reducao_cavidade": _float_ou_none(linha.get("reducao_cavidade")),
            }
    return metadados


def construir_matriz(condicoes_ordenadas: list, espectros: dict, freq_grid: np.ndarray) -> np.ndarray:
    """
    Interpola o espectro de cada condição (freqs, amplitude) — que pode ter
    resolução/comprimento diferente entre condições — no MESMO grid comum de
    frequência, e empilha em uma matriz (condição x freq). Fora do intervalo
    medido de cada condição, preenche com 0 (sem extrapolar).
    """
    linhas = []
    for condicao in condicoes_ordenadas:
        freqs, amplitude = espectros[condicao]
        interpolador = interp1d(freqs, amplitude, bounds_error=False, fill_value=0.0)
        linhas.append(interpolador(freq_grid))
    return np.array(linhas)


def converter_escala(matrix_raw: np.ndarray, modo_escala: str, db_min: float):
    """
    Converte a matriz RAW (já recortada pra UMA faixa: low/mid/high/full) pra
    a escala escolhida — mesmas 6 opções e mesma matemática usadas pela
    etapa 05 (heatmap), documentadas no README. Todos os modos "por
    condição" (pico-canal, abs-condicao, rms-canal, db) calculam a
    referência LINHA A LINHA (uma condição não influencia a escala da
    outra); "abs-global"/"db-global" usam uma única referência pra toda a
    matriz (todas as condições daquela faixa).
    """
    eps = 1e-12
    abs_matrix = np.abs(matrix_raw)

    if modo_escala in ("pico-canal", "abs-condicao"):
        picos = np.max(abs_matrix, axis=1, keepdims=True) + eps
        dados = matrix_raw / picos
        rotulo = ("Amplitude - normalized to per-condition peak" if modo_escala == "pico-canal"
                  else "Amplitude - absolute scale per condition (own peak = 1.0)")
        return dados, 0.0, 1.0, rotulo

    if modo_escala == "rms-canal":
        rms = np.sqrt(np.mean(matrix_raw ** 2, axis=1, keepdims=True)) + eps
        dados = matrix_raw / rms
        return dados, 0.0, float(np.max(dados)) if dados.size else 1.0, "Amplitude - normalized to per-condition RMS"

    if modo_escala == "db":
        picos = np.max(abs_matrix, axis=1, keepdims=True) + eps
        dados = 20 * np.log10((abs_matrix / picos) + eps)
        return dados, db_min, 0.0, "Amplitude (dB, relative to per-condition peak)"

    if modo_escala == "db-global":
        v_max_global = float(np.max(abs_matrix)) if abs_matrix.size else eps
        dados = 20 * np.log10((abs_matrix / v_max_global) + eps)
        return dados, db_min, 0.0, "Amplitude (dB, relative to global peak across all conditions)"

    # abs-global — mesma referência (máxima absoluta, entre TODAS as
    # condições) pra toda a matriz; valores na unidade original.
    v_max_global = float(np.max(abs_matrix)) if abs_matrix.size else eps
    return matrix_raw, 0.0, v_max_global, "Amplitude - absolute, shared scale across all conditions"
