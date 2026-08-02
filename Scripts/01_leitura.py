import argparse
import os
import re
from pathlib import Path
import pandas as pd


def extrair_numero_ensaio(nome_pasta_ou_arquivo: str) -> int:
    """Extrai a ordem numérica da pasta/ensaio (ex: 'T1' -> 1, 'T12' -> 12).
    
    Caso não encontre o padrão T+número, retorna 9999 para ordenação.
    """
    match = re.search(r"[Tt](\d+)", nome_pasta_ou_arquivo)
    if match:
        return int(match.group(1))
    return 9999


def carregar_arquivo_sinal(caminho_arquivo: Path) -> pd.DataFrame:
    """Carrega um arquivo CSV/TXT bruto em um DataFrame.

    Ajuste os parâmetros do read_csv conforme a formatação do seu arquivo
    (ex: sep=';', decimal=',', header=None, etc.).
    """
    try:
        # Se os seus CSVs usarem outro separador ou vírgula decimal, ajuste aqui:
        df = pd.read_csv(caminho_arquivo)
        return df
    except Exception as e:
        print(f"  ⚠️ Erro ao ler {caminho_arquivo.name}: {e}")
        return pd.DataFrame()


def mapear_e_carregar_dados(base_dir: Path) -> pd.DataFrame:
    """Varre a estrutura DadosPuros/Acelerometros/[ACL|PZT]/[T1..Tn]/*.csv

    e consolida tudo em um único DataFrame Pandas.
    """
    alvo_dir = base_dir / "DadosPuros" / "Acelerometros"

    if not alvo_dir.exists():
        raise FileNotFoundError(
            f"❌ Diretório não encontrado: '{alvo_dir}'"
        )

    sensores = ["ACL", "PZT"]
    registros = []

    print(f"🔍 Buscando dados recursivamente em: {alvo_dir}\n")

    for sensor in sensores:
        sensor_path = alvo_dir / sensor
        if not sensor_path.exists():
            print(f"⚠️ Pasta do sensor '{sensor}' não encontrada. Pulando...")
            continue

        # Lista as pastas de condições (T1, T2, ..., Tn)
        pastas_condicoes = [
            p for p in sensor_path.iterdir() if p.is_dir()
        ]

        # Ordena as pastas numericamente pelo T (T1 -> 1, T2 -> 2, T10 -> 10)
        pastas_condicoes.sort(key=lambda p: extrair_numero_ensaio(p.name))

        for pasta_t in pastas_condicoes:
            num_ensaio = extrair_numero_ensaio(pasta_t.name)
            tag_condicao = f"T{num_ensaio}" if num_ensaio != 9999 else pasta_t.name

            # Busca todos os arquivos CSV/TXT dentro da pasta Tn (recursivamente ou direto)
            arquivos_csv = [
                f for f in pasta_t.rglob("*") 
                if f.is_file() and f.suffix.lower() in [".csv", ".txt", ".dat"] and not f.name.startswith(".")
            ]

            if not arquivos_csv:
                print(f"  ⚠️ Nenhum arquivo suportado encontrado na pasta: {pasta_t}")
                continue

            print(f"📂 Sensor [{sensor}] | Condição [{tag_condicao}] -> Encontrados {len(arquivos_csv)} arquivo(s)")

            for arquivo in arquivos_csv:
                print(f"   └── Lendo: {arquivo.name}")
                df_sinal = carregar_arquivo_sinal(arquivo)

                if not df_sinal.empty:
                    # Adiciona metadados em cada linha
                    df_sinal["sensor"] = sensor
                    df_sinal["condicao"] = tag_condicao  # T1, T2, Tn...
                    df_sinal["ordem_ensaio"] = num_ensaio
                    df_sinal["arquivo_origem"] = arquivo.name

                    registros.append(df_sinal)

    if not registros:
        raise ValueError("❌ Nenhum dado foi carregado. Verifique a estrutura das pastas.")

    return pd.concat(registros, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="01_leitura: Mapeamento recursivo e consolidação das pastas T1..Tn"
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

        # Criar diretório de saídas intermediárias
        output_dir = Path("outputs") / "intermediarios"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "01_dados_brutos.parquet"
        
        # Salva a tabela consolidada
        df_dados.to_parquet(output_path, index=False)

        print("\n" + "="*50)
        print("✅ Leitura concluída com sucesso!")
        print("📊 Resumo do carregamento:")
        print(f"   - Total de amostras/linhas salvas: {len(df_dados):,}")
        print(f"   - Sensores identificados: {df_dados['sensor'].unique().tolist()}")
        print(f"   - Condições (Pastas T): {sorted(df_dados['condicao'].unique().tolist())}")
        print(f"💾 Salvo em: {output_path.resolve()}")
        print("="*50 + "\n")

    except Exception as e:
        print(f"\n❌ Erro na etapa 01_leitura: {e}")
        exit(1)


if __name__ == "__main__":
    main()