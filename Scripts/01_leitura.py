import argparse
import os
import re
from pathlib import Path
import pandas as pd


def extrair_numero_ensaio(nome_arquivo: str) -> int:
    """Extrai a ordem numérica do ensaio/condição (ex: 'T1.csv' -> 1, 'T12.txt' -> 12)."""
    match = re.search(r"[Tt](\d+)", nome_arquivo)
    if match:
        return int(match.group(1))
    return 9999


def carregar_arquivo_sinal(caminho_arquivo: Path) -> pd.DataFrame:
    """Carrega o arquivo bruto em um DataFrame.

    Ajuste os argumentos do read_csv conforme seu formato (ex: sep=';', decimal=',').
    """
    try:
        df = pd.read_csv(caminho_arquivo)
        return df
    except Exception as e:
        print(f"⚠️ Erro ao ler {caminho_arquivo.name}: {e}")
        return pd.DataFrame()


def mapear_e_carregar_dados(base_dir: Path) -> pd.DataFrame:
    """Varre diretamente as pastas ACL e PZT e lê os arquivos T1, T2, T3..."""
    alvo_dir = base_dir / "DadosPuros" / "Acelerometros"

    if not alvo_dir.exists():
        raise FileNotFoundError(
            f"❌ Diretório não encontrado: '{alvo_dir}'"
        )

    sensores = ["ACL", "PZT"]
    registros = []

    print(f"🔍 Buscando arquivos diretamente em: {alvo_dir}")

    for sensor in sensores:
        sensor_path = alvo_dir / sensor
        if not sensor_path.exists():
            print(f"⚠️ Pasta '{sensor}' não encontrada em {alvo_dir}. Pulando...")
            continue

        # Filtra os arquivos válidos na pasta do sensor
        arquivos = [
            f for f in sensor_path.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ]

        # Ordena numericamente pelo T (T1, T2, ..., T10)
        arquivos.sort(key=lambda f: extrair_numero_ensaio(f.name))

        for arquivo in arquivos:
            num_ensaio = extrair_numero_ensaio(arquivo.name)
            tag_condicao = f"T{num_ensaio}" if num_ensaio != 9999 else arquivo.stem

            print(f"  -> Lendo: Sensor [{sensor}] | Condição/Ensaio [{tag_condicao}] ({arquivo.name})")

            df_sinal = carregar_arquivo_sinal(arquivo)

            if not df_sinal.empty:
                # Adiciona metadados
                df_sinal["sensor"] = sensor
                df_sinal["condicao"] = tag_condicao  # T1, T2... representam a condição
                df_sinal["ordem_ensaio"] = num_ensaio
                df_sinal["arquivo_origem"] = arquivo.name

                registros.append(df_sinal)

    if not registros:
        raise ValueError("❌ Nenhum dado foi carregado. Verifique os arquivos nas pastas ACL e PZT.")

    return pd.concat(registros, ignore_index=True)


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
        df_dados = mapear_e_carregar_dados(raiz_path)

        # Pasta de saída intermediária
        output_dir = Path("outputs") / "intermediarios"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "01_dados_brutos.parquet"
        df_dados.to_parquet(output_path, index=False)

        print("\n✅ Leitura concluída com sucesso!")
        print("📊 Resumo do carregamento:")
        print(f"   - Total de registros/amostras: {len(df_dados)}")
        print(f"   - Sensores identificados: {df_dados['sensor'].unique().tolist()}")
        print(f"   - Condições/Ensaios mapeados: {sorted(df_dados['condicao'].unique().tolist())}")
        print(f"💾 Salvo em: {output_path.resolve()}\n")

    except Exception as e:
        print(f"\n❌ Erro na etapa 01_leitura: {e}")
        exit(1)


if __name__ == "__main__":
    main()