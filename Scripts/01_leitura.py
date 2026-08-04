import argparse
import importlib.util
from pathlib import Path
import pandas as pd
import re


def _carregar_pipeline_io():
    """
    Carrega src/pipeline_io.py diretamente pelo caminho do arquivo,
    sem passar pelo mecanismo de pacotes (import src.xxx). Isso evita
    problemas de sys.path/__init__.py/imports circulares que podem
    surgir dependendo de como o script é executado (direto, via
    run_pipeline.py, com outro "src" no ambiente, etc.).
    """
    caminho = Path(__file__).resolve().parent.parent / "src" / "pipeline_io.py"
    spec = importlib.util.spec_from_file_location("pipeline_io", caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


_pipeline_io = _carregar_pipeline_io()
salvar_grupo = _pipeline_io.salvar_grupo
registrar_log = _pipeline_io.registrar_log


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


def mapear_e_processar_dados(base_dir: Path, output_dir: Path, quick: bool = False):
    """
    Percorre as pastas de ensaio e, PARA CADA GRUPO (sensor, condicao),
    concatena apenas os arquivos daquele grupo e salva imediatamente em
    disco (.parquet). Isso evita manter o dataset inteiro em memória e
    evita gerar um único arquivo gigante.
    """
    alvo_dir = base_dir / "DadosPuros" / "Acelerometros"
    if not alvo_dir.exists():
        alvo_dir = base_dir

    print(f"🔍 Procurando dados em: {alvo_dir.resolve()}\n")

    pastas_ensaio = [
        d for d in alvo_dir.rglob("*")
        if d.is_dir() and extrair_numero_ensaio(d.name) != 9999
    ]
    pastas_ensaio.sort(key=lambda d: extrair_numero_ensaio(d.name))

    if quick:
        pastas_ensaio = pastas_ensaio[:1]
        print("⚡ Modo rápido (--quick): processando apenas a primeira pasta de ensaio.\n")

    grupos_gerados = []

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

        registros_grupo = []
        for arq in arquivos:
            print(f"   ├── 📄 Lendo: [{sensor}] | [{tag_condicao}] | Arquivo: {arq.name}")
            df_sinal = carregar_arquivo_sinal(arq)

            if not df_sinal.empty:
                df_sinal["sensor"] = sensor
                df_sinal["condicao"] = tag_condicao
                df_sinal["ordem_ensaio"] = num_ensaio
                df_sinal["arquivo_origem"] = arq.name
                registros_grupo.append(df_sinal)

        if not registros_grupo:
            print(f"   ⚠️ Nenhum arquivo válido em {tag_condicao}, pulando grupo.")
            continue

        # Concatena e salva SÓ este grupo, depois libera a memória
        df_grupo = pd.concat(registros_grupo, ignore_index=True)
        caminho_salvo = salvar_grupo(df_grupo, sensor, tag_condicao, output_dir)
        tamanho_mb = caminho_salvo.stat().st_size / (1024 * 1024)
        print(f"   └── 💾 Grupo salvo: {caminho_salvo} ({tamanho_mb:.1f} MB)\n")

        grupos_gerados.append((sensor, tag_condicao, caminho_salvo))
        del df_grupo, registros_grupo  # libera memória explicitamente

    if not grupos_gerados:
        raise ValueError(f"❌ Nenhum arquivo de sinal encontrado em {alvo_dir}")

    return grupos_gerados


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--quick", action="store_true",
                         help="Processa apenas a primeira pasta de ensaio encontrada (teste rápido).")
    args = parser.parse_args()

    raiz_path = Path(args.data_dir)

    # Saída passa a ser um DIRETÓRIO particionado por sensor/condicao,
    # em vez de um único arquivo .pkl. Agrupado em "Etapas/" junto com
    # as saídas de parquet das demais etapas do pipeline.
    output_dir = raiz_path / "DadosTratados" / "Etapas" / "Leitura"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        grupos = mapear_e_processar_dados(raiz_path, output_dir, quick=args.quick)

        caminho_log = registrar_log(raiz_path, "01_leitura", {
            "data_dir": raiz_path.resolve(),
            "quick": args.quick,
            "grupos_gerados": len(grupos),
        }, pastas_alteradas=[output_dir])

        print("\n" + "=" * 60)
        print(f"✅ Etapa 01 (Leitura) Concluída!")
        print(f"💾 {len(grupos)} grupo(s) salvos em: {output_dir.resolve()}")
        print(f"📝 Log de parâmetros: {caminho_log.resolve()}")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Erro na etapa 01_leitura: {e}")
        exit(1)


if __name__ == "__main__":
    main()