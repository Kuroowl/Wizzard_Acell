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
import csv
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


def extrair_numero_subgrupo(caminho_relativo) -> int:
    """
    Extrai o número do subgrupo G<N> a partir das partes de um caminho
    relativo à pasta da condição (ex.: 'G2/arquivo.csv' -> 2).

    Um operador pode ter subpastas G1/G2/G3 dentro de T<N>, que
    representam blocos sequenciais no tempo dentro daquela condição.
    Arquivos soltos direto em T<N> (sem subpasta G<N>) contam como
    "grupo 0", para que sempre venham antes de qualquer G1/G2/G3 na
    ordenação sequencial.
    """
    for parte in Path(caminho_relativo).parts:
        match = re.fullmatch(r"G(\d+)", parte, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 0


def ordenar_arquivos_sequencialmente(pasta_base: Path, arquivos) -> list:
    """
    Ordena uma lista de arquivos (Path) encontrados sob `pasta_base` de
    forma determinística e sequencial:
      1º) pelo número do subgrupo (G1, G2, G3... arquivos soltos em
          T<N>, sem subgrupo, contam como grupo 0 e vêm primeiro);
      2º) pelo nome do arquivo, como desempate dentro do mesmo grupo.

    Necessário porque `Path.rglob()` NÃO garante nenhuma ordem
    específica (depende do sistema de arquivos) — sem isso, a
    concatenação sequencial G1 -> G2 -> G3 (modo --all) não seria
    confiável.
    """
    def chave(arq: Path):
        rel = arq.relative_to(pasta_base)
        return (extrair_numero_subgrupo(rel), rel.name.lower())

    return sorted(arquivos, key=chave)


def ler_metadados_calibracao(caminho_csv: Path) -> dict:
    """
    Lê o CSV opcional de calibração por canal (converte o sinal BRUTO — em
    mV, como já é lido pelo pipeline — para unidade física, usando a
    sensibilidade do sensor e o ganho aplicado no condicionador):

        sensor,canal,condicao,sensibilidade_mv_por_unidade,ganho,unidade_saida

    - sensor: ACL, PZT, etc. (obrigatório)
    - canal: nome exato da coluna no arquivo bruto, ex. 'Channel 0' (obrigatório)
    - condicao: T<N> específico, ou vazio para valer em TODAS as condições
      daquele sensor/canal (útil quando sensibilidade/ganho não mudam entre
      condições — só preencha por condição se o ganho variou durante o
      ensaio)
    - sensibilidade_mv_por_unidade: sensibilidade do sensor, em mV por
      unidade física (ex.: 100 para um acelerômetro de 100 mV/g)
    - ganho: ganho aplicado no condicionador para esse canal (ex.: 1, 10, 100)
    - unidade_saida: rótulo da unidade física resultante (ex.: 'g', 'm/s2') —
      usado só para exibição/rótulo de eixo, não entra na conta.

    Retorna: dict (sensor, canal, condicao) -> {"sensibilidade_mv_por_unidade":
    float, "ganho": float, "unidade_saida": str}. Uma linha com condicao vazia
    fica armazenada com chave (sensor, canal, "") — funciona como fallback
    para todas as condições daquele sensor/canal que não tenham uma linha
    mais específica no CSV.
    """
    metadados = {}
    with open(caminho_csv, newline="", encoding="utf-8-sig") as f:
        leitor = csv.DictReader(f)
        colunas_esperadas = {"sensor", "canal", "condicao", "sensibilidade_mv_por_unidade", "ganho", "unidade_saida"}
        if not colunas_esperadas.issubset(set(leitor.fieldnames or [])):
            faltando = colunas_esperadas - set(leitor.fieldnames or [])
            raise ValueError(f"CSV de calibração sem as colunas esperadas: {faltando}")

        for linha in leitor:
            sensor = (linha.get("sensor") or "").strip().upper()
            canal = (linha.get("canal") or "").strip()
            condicao = (linha.get("condicao") or "").strip().upper()
            if not sensor or not canal:
                continue
            try:
                sensibilidade = float((linha.get("sensibilidade_mv_por_unidade") or "").strip())
                ganho = float((linha.get("ganho") or "").strip())
            except ValueError:
                raise ValueError(
                    f"CSV de calibração: sensibilidade/ganho inválidos para {sensor}/{canal}/{condicao or '(todas)'}."
                )
            if sensibilidade <= 0 or ganho <= 0:
                raise ValueError(
                    f"CSV de calibração: sensibilidade e ganho devem ser > 0 ({sensor}/{canal}/{condicao or '(todas)'})."
                )
            metadados[(sensor, canal, condicao)] = {
                "sensibilidade_mv_por_unidade": sensibilidade,
                "ganho": ganho,
                "unidade_saida": (linha.get("unidade_saida") or "").strip() or "un",
            }
    return metadados


def buscar_fator_calibracao(metadados_calibracao: dict, sensor: str, canal: str, condicao: str):
    """
    Busca a entrada de calibração de (sensor, canal, condicao) — primeiro
    tentando um match específico para essa condição, depois caindo para a
    entrada genérica (condicao vazia no CSV, vale para todas).

    Retorna (fator, unidade_saida) onde `fator` converte o sinal bruto (o
    dado já lido do CSV, em mV) para a unidade física: fisico = bruto_mV *
    fator. Retorna (None, None) se não houver calibração cadastrada para
    esse sensor/canal.
    """
    sensor = str(sensor).strip().upper()
    condicao = str(condicao).strip().upper()
    info = metadados_calibracao.get((sensor, canal, condicao)) or metadados_calibracao.get((sensor, canal, ""))
    if info is None:
        return None, None
    # mV_medido = ganho * sensibilidade_mV_por_unidade * valor_fisico
    # => valor_fisico = mV_medido / (ganho * sensibilidade_mV_por_unidade)
    fator = 1.0 / (info["ganho"] * info["sensibilidade_mv_por_unidade"])
    return fator, info["unidade_saida"]


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
