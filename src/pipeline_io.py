# -*- coding: utf-8 -*-
"""
pipeline_io.py
Funções compartilhadas entre as etapas do pipeline para ler/escrever
dados particionados por sensor/condição (em vez de um único arquivo
monolítico). Isso permite:
  - processar um grupo por vez (economia de RAM/disco)
  - retomar o pipeline a partir de qualquer etapa intermediária
  - rodar em modo "rápido" (apenas o primeiro grupo) para testes
"""

from pathlib import Path
from datetime import datetime
import getpass
import re
import pandas as pd


def salvar_grupo(df: pd.DataFrame, sensor: str, condicao: str, saida_dir: Path,
                  downcast_float32: bool = True) -> Path:
    """Salva um grupo (sensor, condicao) em um arquivo .parquet próprio."""
    pasta = saida_dir / str(sensor)
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / f"{condicao}.parquet"

    if downcast_float32:
        cols_float = df.select_dtypes(include=["float64"]).columns
        df[cols_float] = df[cols_float].astype("float32")

    df.to_parquet(caminho, index=False)
    return caminho


def listar_grupos(etapa_dir: Path):
    """
    Lista todos os arquivos .parquet de uma etapa anterior.
    Retorna lista de tuplas (sensor, condicao, caminho), ordenada.
    """
    if not etapa_dir.exists():
        return []

    arquivos = sorted(etapa_dir.rglob("*.parquet"))
    grupos = []
    for arq in arquivos:
        condicao = arq.stem          # nome do arquivo = condicao (ex: T1)
        sensor = arq.parent.name     # nome da pasta pai = sensor (ex: ACL)
        grupos.append((sensor, condicao, arq))
    return grupos


def carregar_grupo(caminho: Path) -> pd.DataFrame:
    """Carrega um único grupo (sensor, condicao) sob demanda."""
    return pd.read_parquet(caminho)


def extrair_numero_condicao(nome):
    """Extrai o número de uma condição no padrão T<N> (ex.: 'T3' -> 3,
    'T10' -> 10). Retorna None se o nome não bater com esse padrão."""
    match = re.search(r"T(\d+)", str(nome), re.IGNORECASE)
    return int(match.group(1)) if match else None


def filtrar_desde_condicao(itens, condicao_minima, indice_condicao=None):
    """
    Suporte a --from CONDICAO (retomar uma etapa a partir de uma condição
    específica, inclusive). Filtra `itens` mantendo só os que têm número de
    condição >= o de `condicao_minima` (extraído via extrair_numero_condicao).

    `itens`: lista de strings (condicao) ou de tuplas onde um dos elementos
    é a condicao (informe `indice_condicao` nesse caso, ex.: 1 para o
    formato (sensor, condicao, caminho) do listar_grupos).

    Comportamento seguro: se `condicao_minima` for None/vazio, retorna a
    lista inteira sem filtrar. Se não for possível extrair um número de
    `condicao_minima` (ou de algum item), esse item NÃO é descartado — é
    mantido, para nunca perder dados silenciosamente por um nome fora do
    padrão T<N>.
    """
    if not condicao_minima:
        return itens

    numero_min = extrair_numero_condicao(condicao_minima)
    if numero_min is None:
        return itens

    resultado = []
    for item in itens:
        condicao = item[indice_condicao] if indice_condicao is not None else item
        numero = extrair_numero_condicao(condicao)
        if numero is None or numero >= numero_min:
            resultado.append(item)
    return resultado


def registrar_log(raiz_path: Path, etapa: str, parametros: dict,
                   pastas_alteradas=None) -> Path:
    """
    Registra, em um .txt cumulativo (uma entrada por execução), os
    parâmetros usados em cada etapa do pipeline. Serve para permitir
    refazer/auditar uma análise depois, sabendo exatamente com que
    configuração cada figura/arquivo foi gerado, quem rodou e o que
    foi tocado no disco.

    O log é sempre em modo "append" (nunca sobrescreve): quando o
    pipeline é usado com análises parciais/retomadas, cada execução
    fica registrada como uma entrada própria, identificada por
    timestamp e usuário — em vez de perder o histórico anterior.

    Arquivo: DadosTratados/Logs/pipeline_log.txt

    Parâmetros
    ----------
    pastas_alteradas : list[Path] | None
        Pastas efetivamente criadas/escritas nesta execução (parquet
        e/ou figuras). Ajuda a auditar rapidamente "o que essa rodada
        mexeu" sem precisar reconstruir isso a partir dos parâmetros.
    """
    pasta_logs = raiz_path / "DadosTratados" / "Logs"
    pasta_logs.mkdir(parents=True, exist_ok=True)
    caminho_log = pasta_logs / "pipeline_log.txt"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    usuario = getpass.getuser()

    linhas = [
        "=" * 60,
        f"[{timestamp}] Etapa: {etapa}",
        f"Usuario: {usuario}",
        "Parametros:",
    ]
    for chave, valor in parametros.items():
        linhas.append(f"  {chave}: {valor}")

    if pastas_alteradas:
        linhas.append("Pastas alteradas:")
        for pasta in sorted(set(str(Path(p).resolve()) for p in pastas_alteradas)):
            linhas.append(f"  - {pasta}")

    linhas.append("=" * 60)
    linhas.append("")

    with open(caminho_log, "a", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")

    return caminho_log
