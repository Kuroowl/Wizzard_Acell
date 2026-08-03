import argparse
import os
import re
from pathlib import Path
import pandas as pd


def extrair_numero_ensaio(nome: str) -> int:
    """Extrai o número do ensaio/condição (ex: 'T1' -> 1, 'T12.csv' -> 12)."""
    match = re.search(r"T(\d+)", nome, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 9999


def carregar_arquivo_sinal(caminho_arquivo: Path) -> pd.DataFrame:
    """Carrega um arquivo .csv ou .txt de sinal em um DataFrame pandas."""
    try:
        # Tenta carregar com vírgula ou ponto e vírgula como separador
        try:
            df = pd.read_csv(caminho_arquivo)
            if df.shape[1] == 1:  # Se leu tudo em uma coluna só, tenta ';'
                df = pd.read_csv(caminho_arquivo, sep=";")
        except Exception:
            df = pd.read_csv(caminho_arquivo, sep=";")

        return df
    except Exception as e:
        print(f"   ⚠️ Erro ao ler {caminho_arquivo.name}: {e}")
        return pd.DataFrame()


def mapear_e_carregar_dados(base_dir: Path) -> pd.DataFrame:
    """Varre as subpastas T1..Tn dentro de ACL e PZT e consolida todos os dados."""

    alvo_dir = base_dir / "DadosPuros" / "Acelerometros"

    if not alvo_dir.exists():
        raise FileNotFoundError(
            f"❌ Diretório de entrada não encontrado: '{alvo_dir.resolve()}'\n"
            "   Certifique-se de que a pasta selecionada contém 'DadosPuros/Acelerometros'."
        )

    sensores = ["ACL", "PZT"]
    registros = []

    print(f"🔍 Buscando dados em: {alvo_dir.resolve()}\n")

    for sensor in sensores:
        sensor_path = alvo_dir / sensor

        if not sensor_path.exists():
            print(f"⚠️ Pasta do sensor '{sensor}' não encontrada em: {alvo_dir}. Pulando...")
            continue

        print(f"📂 Processando Sensor: [{sensor}]")

        # Mapeia subpastas T1, T2, ..., Tn (ou arquivos soltos caso existam)
        itens = [f for f in sensor_path.iterdir() if not f.name.startswith(".")]

        # Separa pastas de ensaio (T1, T2...)
        pastas_ensaio = [d for d in itens if d.is_dir()]

        # Ordena as pastas numericamente pelo número T (T1, T2, ..., T10)
        pastas_ensaio.sort(key=lambda d: extrair_numero_ensaio(d.name))

        if not pastas_ensaio:
            print(f"   ⚠️ Nenhuma pasta de condição (T1, T2...) encontrada em {sensor_path}")
            continue

        for pasta_t in pastas_ensaio:
            num_ensaio = extrair_numero_ensaio(pasta_t.name)
            tag_condicao = pasta_t.name  # Ex: "T1", "T2"

            # Busca todos os arquivos .csv dentro da pasta Tn (recursivo)
            arquivos_csv = sorted(
                [f for f in pasta_t.rglob("*.csv") if not f.name.startswith(".")]
            )

            if not arquivos_csv:
                # Tenta buscar arquivos de texto (.txt) caso não haja .csv
                arquivos_csv = sorted(
                    [f for f in pasta_t.rglob("*.txt") if not f.name.startswith(".")]
                )

            for arq in arquivos_csv:
                print(f"   ├── 📄 Lendo: Condição [{tag_condicao}] | Arquivo: {arq.name}")

                df_sinal = carregar_arquivo_sinal(arq)

                if not df_sinal.empty:
                    # Adiciona colunas de metadados cruciais
                    df_sinal["sensor"] = sensor
                    df_sinal["condicao"] = tag_condicao
                    df_sinal["ordem_ensaio"] = num_ensaio
                    df_sinal["arquivo_origem"] = arq.name

                    registros.append(df_sinal)

        print()  # Quebra de linha entre sensores

    if not registros:
        raise ValueError(
            "❌ Nenhum dado foi carregado! Verifique se há arquivos .csv/.txt "
            "dentro das pastas T1, T2... em DadosPuros/Acelerometros/ACL e PZT."
        )

    # Consolida tudo