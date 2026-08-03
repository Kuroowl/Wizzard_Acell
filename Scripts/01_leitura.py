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
        try:
            df = pd.read_csv(caminho_arquivo)
            if df.shape[1] == 1:
                df = pd.read_csv(caminho_arquivo, sep=";")
        except Exception:
            df = pd.read_csv(caminho_arquivo, sep=";")

        return df
    except Exception as e:
        print(f"   ⚠️ Erro ao ler {caminho_arquivo.name}: {e}")
        return pd.DataFrame()


def mapear_e_carregar_dados(base_dir: Path) -> pd.DataFrame:
    """Varre as subpastas T1..Tn dentro de ACL e PZT e consolida todos os dados."""

    # Verifica se existe DadosPuros/Acelerometros ou usa a própria pasta informada
    alvo_dir = base_dir / "DadosPuros" / "Acelerometros"
    if not alvo_dir.exists():
        alvo_dir = base_dir

    print(f"🔍 Procurando dados em: {alvo_dir.resolve()}\n")

    sensores = ["ACL", "PZT"]
    registros = []

    for sensor in sensores:
        sensor_path = alvo_dir / sensor

        # Se a subpasta ACL/PZT não existir na raiz, tenta buscar pastas T1, T2 diretas
        if not sensor_path.exists():
            sensor_path = alvo_dir

        print(f"📂 Buscando para o Sensor: [{sensor}] em {sensor_path.resolve()}")

        # Procura pastas no formato T1, T2, T3...
        pastas_ensaio = [
            d for d in sensor_path.glob("*") if d.is_dir() and extrair_numero_ensaio(d.name) != 9999
        ]
        
        # Ordena numericamente (T1, T2, ..., T10)
        pastas_ensaio.sort(key=lambda d: extrair_numero_ensaio(d.name))

        if not pastas_ensaio:
            print(f"   ⚠️ Nenhuma pasta de condição (T1, T2...) encontrada para {sensor}.")
            continue

        for pasta_t in pastas_ensaio:
            tag_condicao = pasta_t.name
            num_ensaio = extrair_numero_ensaio(tag_condicao)

            # Busca arquivos de dados (.csv, .txt, .dat)
            arquivos = sorted(list(pasta_t.glob("*.csv")) + list(pasta_t.glob("*.txt")) + list(pasta_t.glob("*.dat")))

            for arq in arquivos:
                print(f"   ├── 📄 Lendo [{sensor}] | Condição [{tag_condicao}] | Arquivo: {arq.name}")

                df_sinal = carregar_arquivo_sinal(arq)

                if not df_sinal.empty:
                    df_sinal["sensor"] = sensor
                    df_sinal["condicao"] = tag_condicao
                    df_sinal["ordem_ensaio"] = num_ensaio
                    df_sinal["arquivo_origem"] = arq.name

                    registros.append(df_sinal)

        print()

    if not registros:
        print("\n❌ NENHUM ARQUIVO ENCONTRADO!")
        print(" Certifique-se de que a pasta selecionada contém subpastas no formato T1, T2, T3 com arquivos .csv ou .txt dentro.")
        raise ValueError("Nenhum dado válido foi localizado na pasta selecionada.")

    df_consolidado = pd.concat(registros, ignore_index=True)
    return df_consolidado


def main():
    parser = argparse.ArgumentParser(
        description="01_leitura: Mapeamento e consolidação dos ensaios T1, T2, T3..."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Caminho da pasta raiz selecionada no main",
    )
    args = parser.parse_args()
    raiz_path = Path(args.data_dir)

    try:
        # 1. Carrega os dados brutos
        df_dados = mapear_e_carregar_dados(raiz_path)

        # 2. Garante a criação do diretório DadosTratados
        output_dir = raiz_path / "DadosTratados"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 3. Salva em Pickle (.pkl)
        output_path = output_dir / "DadosTratados.pkl"
        df_dados.to_pickle(output_path)

        print("=" * 60)
        print("✅ Leitura concluída e salva com sucesso!")
        print("📊 Resumo do carregamento:")
        print(f"   - Total de amostras carregadas: {len(df_dados):,}")
        print(f"   - Sensores identificados: {df_dados['sensor'].unique().tolist()}")
        print(f"   - Condições mapeadas: {sorted(df_dados['condicao'].unique().tolist())}")
        print(f"💾 Arquivo gerado em: {output_path.resolve()}")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Erro na etapa 01_leitura: {e}")
        exit(1)


if __name__ == "__main__":
    main()