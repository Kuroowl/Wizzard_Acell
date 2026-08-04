import argparse
from pathlib import Path
import pandas as pd
import re

def extrair_numero_ensaio(nome: str) -> int:
    match = re.search(r"T(\d+)", nome, re.IGNORECASE)
    return int(match.group(1)) if match else 9999

def carregar_arquivo_sinal(caminho_arquivo: Path) -> pd.DataFrame:
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
    alvo_dir = base_dir / "DadosPuros" / "Acelerometros"
    if not alvo_dir.exists():
        alvo_dir = base_dir

    print(f"🔍 Procurando dados em: {alvo_dir.resolve()}\n")
    registros = []

    pastas_ensaio = [
        d for d in alvo_dir.rglob("*") 
        if d.is_dir() and extrair_numero_ensaio(d.name) != 9999
    ]
    pastas_ensaio.sort(key=lambda d: extrair_numero_ensaio(d.name))

    for pasta_t in pastas_ensaio:
        tag_condicao = pasta_t.name
        num_ensaio = extrair_numero_ensaio(tag_condicao)
        
        partes_caminho = [p.upper() for p in pasta_t.parts]
        sensor = "ACL" if "ACL" in partes_caminho else ("PZT" if "PZT" in partes_caminho else "Geral")

        arquivos = (
            list(pasta_t.rglob("*.csv")) + 
            list(pasta_t.rglob("*.txt")) + 
            list(pasta_t.rglob("*.dat"))
        )
        arquivos = [f for f in arquivos if not f.name.startswith(".")]

        for arq in arquivos:
            print(f"   ├── 📄 Lendo: [{sensor}] | [{tag_condicao}] | Arquivo: {arq.name}")
            df_sinal = carregar_arquivo_sinal(arq)

            if not df_sinal.empty:
                df_sinal["sensor"] = sensor
                df_sinal["condicao"] = tag_condicao
                df_sinal["ordem_ensaio"] = num_ensaio
                df_sinal["arquivo_origem"] = arq.name
                registros.append(df_sinal)

    if not registros:
        raise ValueError(f"❌ Nenhum arquivo de sinal encontrado em {alvo_dir}")

    return pd.concat(registros, ignore_index=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    
    raiz_path = Path(args.data_dir)
    nome_projeto = raiz_path.name  # Exemplo: TesteBomba

    try:
        df_dados = mapear_e_carregar_dados(raiz_path)

        output_dir = raiz_path / "DadosTratados"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Saída padronizada: TesteBomba_leitura.pkl
        output_path = output_dir / f"{nome_projeto}_leitura.pkl"
        df_dados.to_pickle(output_path)

        print("\n" + "=" * 60)
        print(f"✅ Etapa 01 (Leitura) Concluída!")
        print(f"💾 Arquivo gerado: {output_path.resolve()}")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Erro na etapa 01_leitura: {e}")
        exit(1)

if __name__ == "__main__":
    main()