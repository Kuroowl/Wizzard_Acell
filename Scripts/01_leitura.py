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
extrair_numero_condicao = _pipeline_io.extrair_numero_condicao
extrair_numero_subgrupo = _pipeline_io.extrair_numero_subgrupo
ordenar_arquivos_sequencialmente = _pipeline_io.ordenar_arquivos_sequencialmente


def extrair_numero_ensaio(nome: str) -> int:
    match = re.search(r"T(\d+)", nome, re.IGNORECASE)
    return int(match.group(1)) if match else 9999


def parse_grupo_alvo(valor: str) -> int:
    """Converte o valor passado em --grupo (ex.: 'G1', 'g1', '1') para o
    número inteiro do subgrupo. Levanta ValueError se o formato não bater
    com G<N>/N, para falhar cedo com uma mensagem clara."""
    match = re.fullmatch(r"G?(\d+)", valor.strip(), re.IGNORECASE)
    if not match:
        raise ValueError(f"Valor inválido para --grupo: '{valor}' (use algo como G1, G2, G3...).")
    return int(match.group(1))


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


def mapear_e_processar_dados(base_dir: Path, output_dir: Path, quick: bool = False,
                              from_condicao: str = None, ler_todos: bool = False,
                              grupo_alvo: int = None):
    """
    Percorre as pastas de ensaio e, PARA CADA GRUPO (sensor, condicao),
    concatena apenas os arquivos daquele grupo e salva imediatamente em
    disco (.parquet). Isso evita manter o dataset inteiro em memória e
    evita gerar um único arquivo gigante.

    Dentro de cada pasta T<N> pode haver subpastas de subgrupo
    (G1/G2/G3, ...), representando blocos sequenciais no tempo dentro
    daquela condição. Existem três modos de leitura, mutuamente
    exclusivos:

      - padrão (ler_todos=False, grupo_alvo=None): lê apenas o
        PRIMEIRO arquivo encontrado em T<N>, na ordem sequencial
        G1 -> G2 -> G3 (e, dentro de cada G, por nome de arquivo). Se
        não houver subgrupos, é o primeiro arquivo (por nome) direto
        em T<N>.
      - ler_todos=True (flag --all): concatena TODOS os arquivos de
        TODOS os subgrupos de T<N>, na ordem sequencial G1 -> G2 -> G3
        (e, dentro de cada G, por nome de arquivo), formando um único
        "mega arquivo" por condição.
      - grupo_alvo=N (flag --grupo GN): concatena apenas os arquivos
        do subgrupo GN daquela condição (arquivo 1, 2, 3... n, na
        ordem por nome), ignorando os demais subgrupos. Condições sem
        o subgrupo GN são puladas com aviso.

    Em nenhum dos modos a informação de subgrupo é persistida no
    parquet de saída — ela só existe durante a leitura, para decidir
    quais arquivos entram e em que ordem. As etapas seguintes do
    pipeline não sabem (nem precisam saber) de onde cada linha veio.
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

    if from_condicao:
        numero_min = extrair_numero_condicao(from_condicao)
        if numero_min is not None:
            pastas_ensaio = [p for p in pastas_ensaio if extrair_numero_ensaio(p.name) >= numero_min]
        print(f"⏩ Retomando a partir de {from_condicao} (--from): {len(pastas_ensaio)} pasta(s) a processar.\n")

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
        # rglob() não garante nenhuma ordem específica (depende do
        # sistema de arquivos) — ordena sempre de forma determinística
        # e sequencial (G1 -> G2 -> G3, depois por nome de arquivo),
        # tanto para escolher "o primeiro arquivo" quanto para montar
        # o mega arquivo no modo --all.
        arquivos = ordenar_arquivos_sequencialmente(pasta_t, arquivos)

        if not arquivos:
            print(f"   ⚠️ Nenhum arquivo válido em {tag_condicao}, pulando grupo.")
            continue

        if grupo_alvo is not None:
            arquivos_do_grupo = [
                a for a in arquivos
                if extrair_numero_subgrupo(a.relative_to(pasta_t)) == grupo_alvo
            ]
            if not arquivos_do_grupo:
                print(f"   ⚠️ {tag_condicao} não tem subgrupo G{grupo_alvo}, pulando grupo.")
                continue
            print(f"   ℹ️  --grupo G{grupo_alvo}: {len(arquivos_do_grupo)} arquivo(s) em "
                  f"{tag_condicao}/G{grupo_alvo} serão concatenados em ordem.")
            arquivos = arquivos_do_grupo
        elif ler_todos:
            if len(arquivos) > 1:
                print(f"   ℹ️  --all: {len(arquivos)} arquivo(s) em {tag_condicao} "
                      f"serão concatenados em ordem sequencial (subgrupos G1/G2/G3...).")
        else:
            if len(arquivos) > 1:
                print(f"   ℹ️  {len(arquivos)} arquivo(s) encontrados em {tag_condicao}; "
                      f"lendo apenas o primeiro em ordem sequencial: {arquivos[0].name} "
                      f"(use --all para concatenar todos os subgrupos, ou --grupo GN para um "
                      f"subgrupo específico).")
            arquivos = arquivos[:1]

        registros_grupo = []
        for arq in arquivos:
            subgrupo = extrair_numero_subgrupo(arq.relative_to(pasta_t))
            rotulo_subgrupo = f"G{subgrupo}" if subgrupo else "-"
            print(f"   ├── 📄 Lendo: [{sensor}] | [{tag_condicao}] | Subgrupo: {rotulo_subgrupo} | Arquivo: {arq.name}")
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
    parser.add_argument("--from", dest="from_condicao", type=str, default=None,
                         help="Retoma a partir desta condição, inclusive (ex.: --from T3 processa T3, "
                              "T4, T5... e pula T1/T2). Extrai o número do padrão T<N> no nome da pasta.")
    grupo_modo = parser.add_mutually_exclusive_group()
    grupo_modo.add_argument("--all", dest="ler_todos", action="store_true",
                             help="Quando uma condição T<N> tem subgrupos G1/G2/G3 (blocos sequenciais "
                                  "no tempo), concatena TODOS os arquivos de TODOS os subgrupos, em ordem "
                                  "sequencial G1->G2->G3, formando um único mega arquivo por condição.")
    grupo_modo.add_argument("--grupo", dest="grupo_alvo_raw", type=str, default=None,
                             help="Concatena apenas os arquivos de um subgrupo específico (ex.: --grupo G1), "
                                  "ignorando os demais subgrupos daquela condição. Condições sem esse "
                                  "subgrupo são puladas com aviso. Mutuamente exclusivo com --all.")
    args = parser.parse_args()

    grupo_alvo = None
    if args.grupo_alvo_raw is not None:
        try:
            grupo_alvo = parse_grupo_alvo(args.grupo_alvo_raw)
        except ValueError as e:
            print(f"❌ {e}")
            exit(1)

    raiz_path = Path(args.data_dir)

    # Saída passa a ser um DIRETÓRIO particionado por sensor/condicao,
    # em vez de um único arquivo .pkl. Agrupado em "Etapas/" junto com
    # as saídas de parquet das demais etapas do pipeline.
    output_dir = raiz_path / "DadosTratados" / "Etapas" / "Leitura"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        grupos = mapear_e_processar_dados(raiz_path, output_dir, quick=args.quick,
                                           from_condicao=args.from_condicao, ler_todos=args.ler_todos,
                                           grupo_alvo=grupo_alvo)

        caminho_log = registrar_log(raiz_path, "01_leitura", {
            "data_dir": raiz_path.resolve(),
            "quick": args.quick,
            "from_condicao": args.from_condicao,
            "all": args.ler_todos,
            "grupo": args.grupo_alvo_raw,
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