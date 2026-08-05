from __future__ import annotations
import argparse
import importlib.util
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def _carregar_modulo(nome: str, arquivo: str):
    """Carrega um módulo de src/ diretamente pelo caminho (evita import via pacote 'src', ver 01_leitura.py)."""
    caminho = Path(__file__).resolve().parent.parent / "src" / arquivo
    spec = importlib.util.spec_from_file_location(nome, caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


_pipeline_io = _carregar_modulo("pipeline_io", "pipeline_io.py")
listar_grupos = _pipeline_io.listar_grupos
carregar_grupo = _pipeline_io.carregar_grupo
salvar_grupo = _pipeline_io.salvar_grupo
registrar_log = _pipeline_io.registrar_log

_estilo = _carregar_modulo("estilo_grafico", "estilo_grafico.py")
My_axis = _estilo.My_axis

# Mesmas cores vivas da etapa 04, fora do esquema de cor do heatmap (viridis/inferno/etc.)
COR_PICOS = {"low": "#2979FF", "mid": "#00C853", "high": "#FF1744"}

# Texto explicativo por modo de --escala, registrado no log (mesmo espírito
# do METODO_ESPECTRAL da etapa 03: método + funções + parâmetros usados).
METODO_ESCALA_DESCRICAO = {
    "db-global": (
        "Amplitude (dB) = 20*log10(|A| / A_max_global + eps), onde A_max_global "
        "é a maior amplitude absoluta entre TODAS as condições daquela faixa "
        "(np.max sobre a matriz inteira). Só a condição com o pico global bate "
        "0 dB; as demais ficam abaixo, preservando a intensidade ABSOLUTA "
        "relativa entre condições. Piso do gráfico: --db-min."
    ),
    "abs-global": (
        "Sem normalizar: dados plotados na unidade original (saída da etapa "
        "03/FFT), com uma única referência de cor pra toda a figura: "
        "vmax = maior amplitude absoluta entre TODAS as condições daquela "
        "faixa (np.max)."
    ),
    "abs-condicao": (
        "Cada condição (linha do mapa) dividida pelo próprio pico absoluto: "
        "A / max(|A|) por linha (np.max(axis=1)). Mesmo cálculo do "
        "'pico-canal', lido aqui como 'escala absoluta com referência própria "
        "por condição' (única forma de dar a cada linha seu próprio teto de "
        "cor dentro de uma imagem só)."
    ),
    "pico-canal": (
        "Normalização relativa: cada condição dividida pelo próprio pico "
        "absoluto (np.max(axis=1) por linha), resultando em valores de 0 a 1 "
        "(sem unidade física)."
    ),
    "rms-canal": (
        "Cada condição dividida pelo próprio RMS: A / sqrt(mean(A**2)) por "
        "linha (np.sqrt, np.mean(axis=1)), realçando o sinal acima do nível "
        "médio de energia daquela condição."
    ),
    "db": (
        "Amplitude (dB) = 20*log10(|A| / max(|A|) + eps), com max(|A|) "
        "calculado POR LINHA (por condição, não global) — todas as condições "
        "batem 0 dB no próprio pico; não preserva intensidade absoluta entre "
        "condições (ver 'db-global' para isso). Piso do gráfico: --db-min."
    ),
}


def _bordas_a_partir_de_centros(centros) -> np.ndarray:
    """
    Converte posições centrais (ex.: f_vfd_hz de cada condição) em bordas
    (len+1), pra usar com pcolormesh — respeita o espaçamento REAL entre
    condições, ao contrário do imshow (que assume espaçamento uniforme).
    """
    centros = np.asarray(centros, dtype=float)
    if len(centros) == 1:
        return np.array([centros[0] - 0.5, centros[0] + 0.5])
    bordas = np.empty(len(centros) + 1)
    bordas[1:-1] = (centros[:-1] + centros[1:]) / 2
    bordas[0] = centros[0] - (centros[1] - centros[0]) / 2
    bordas[-1] = centros[-1] + (centros[-1] - centros[-2]) / 2
    return bordas


def sanitizar_nome(nome: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(nome)).strip("_") or "canal"


# ==============================================================================
# 🛠️ 1. METADADOS OPCIONAIS POR CONDIÇÃO (CSV)
# ==============================================================================
def ler_metadados_condicoes(caminho_csv: Path) -> dict:
    """
    Lê o CSV opcional de metadados por condição (formato documentado no
    README): condicao,f_vfd_hz,vazao_m3h,reducao_shaft,reducao_cavidade.

    Campos vazios viram None. reducao_shaft/reducao_cavidade são constantes
    do equipamento (mesmo valor esperado em todas as linhas); aqui só
    guardamos o que veio, a decisão de qual valor usar fica em obter_reducoes().

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


def obter_reducoes(metadados: dict, condicoes: list[str]):
    """Pega reducao_shaft/reducao_cavidade da 1ª condição que tiver o valor preenchido
    (são constantes do equipamento, não da condição individual)."""
    reducao_shaft = reducao_cavidade = None
    for condicao in condicoes:
        info = metadados.get(condicao, {})
        if reducao_shaft is None and info.get("reducao_shaft") is not None:
            reducao_shaft = info["reducao_shaft"]
        if reducao_cavidade is None and info.get("reducao_cavidade") is not None:
            reducao_cavidade = info["reducao_cavidade"]
    return reducao_shaft, reducao_cavidade


# ==============================================================================
# 🛠️ 2. CONSTRUÇÃO DA MATRIZ ESPECTRAL (condição x frequência)
# ==============================================================================
def construir_matriz(condicoes_ordenadas: list[str], espectros: dict, freq_grid: np.ndarray) -> np.ndarray:
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
    Converte a matriz RAW (já recortada pra UMA faixa: low/mid/high) pra a
    escala de cor escolhida. Todos os modos "por condição" (pico-canal,
    abs-condicao, rms-canal, db) calculam a referência LINHA A LINHA (uma
    condição não influencia a escala da outra); "abs-global" usa uma única
    referência pra toda a matriz (todas as condições daquela faixa).
    """
    eps = 1e-12
    abs_matrix = np.abs(matrix_raw)

    if modo_escala in ("pico-canal", "abs-condicao"):
        # Mesma operação matemática nos dois nomes (é a única forma de dar a
        # cada condição seu próprio teto de cor dentro de uma imagem só) —
        # "pico-canal" é lido como normalização relativa (0-1, sem unidade);
        # "abs-condicao" é a mesma escala lida como "absoluta, mas com
        # referência própria de cada condição" (Opção B do heatmap).
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
        # Amplitude (dB) = 20*log10(A(f,condicao) / A_max_global) — só a
        # condição que contém o pico global bate 0 dB; as outras ficam
        # abaixo, preservando a comparação de intensidade ABSOLUTA entre
        # condições (ao contrário do "db" acima, que normaliza cada
        # condição pelo próprio pico e por isso todas batem 0 dB).
        v_max_global = float(np.max(abs_matrix)) if abs_matrix.size else eps
        dados = 20 * np.log10((abs_matrix / v_max_global) + eps)
        return dados, db_min, 0.0, "Amplitude (dB, relative to global peak across all conditions)"

    # abs-global — Opção A: mesma referência (máxima absoluta,
    # entre TODAS as condições) pra toda a matriz; valores plotados em
    # unidade original (sem dividir nada).
    v_max_global = float(np.max(abs_matrix)) if abs_matrix.size else eps
    return matrix_raw, 0.0, v_max_global, "Amplitude - absolute, shared scale across all conditions"


# ==============================================================================
# 🚀 3. EXECUÇÃO PRINCIPAL
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--quick", action="store_true",
                         help="Processa apenas o primeiro sensor encontrado, para teste rápido.")
    parser.add_argument("--metadados-condicoes", type=str, default=None,
                         help="Caminho de um CSV opcional (condicao,f_vfd_hz,vazao_m3h,reducao_shaft,"
                              "reducao_cavidade) — ver formato no README. Se omitido, o eixo Y usa o "
                              "rótulo categórico da condição (T1..Tn).")
    parser.add_argument("--escala", choices=["db-global", "abs-global", "abs-condicao", "pico-canal", "rms-canal", "db"],
                         default="db-global",
                         help="Escala de cor do heatmap (padrão: db-global). Ver README para o que cada "
                              "modo significa.")
    parser.add_argument("--db-min", type=float, default=-40.0,
                         help="Piso (dB) usado quando --escala db (padrão: -40.0).")
    parser.add_argument("--cmap", type=str, default="viridis",
                         help="Colormap do matplotlib (padrão: viridis).")
    parser.add_argument("--freq-max", type=float, default=None,
                         help="Frequência máxima (Hz) do heatmap. Padrão: a menor frequência máxima "
                              "entre as condições daquele sensor/canal (evita extrapolar).")
    parser.add_argument("--freq-resolucao", type=float, default=0.5,
                         help="Resolução (Hz) do grid comum de frequência usado para interpolar as "
                              "condições (padrão: 0.5).")
    parser.add_argument("--f1", type=float, default=15.0,
                         help="Limite low/mid (Hz), só para desenhar linhas de referência verticais (padrão: 15.0).")
    parser.add_argument("--f2", type=float, default=400.0,
                         help="Limite mid/high (Hz), só para desenhar linhas de referência verticais (padrão: 400.0).")
    parser.add_argument("--sem-picos", action="store_true",
                         help="Não sobrepõe os picos (Etapas/Picos) como marcadores no heatmap.")
    args = parser.parse_args()

    raiz_path = Path(args.data_dir)

    input_dir = raiz_path / "DadosTratados" / "Etapas" / "FFT"
    grupos = listar_grupos(input_dir)

    if not grupos:
        print(f"❌ Nenhum grupo encontrado em: {input_dir.resolve()}")
        print(" Certifique-se de executar a etapa 03_fft antes desta.")
        exit(1)

    metadados = {}
    caminho_metadados_condicoes = None
    if args.metadados_condicoes:
        caminho_metadados_condicoes = Path(args.metadados_condicoes)
    else:
        # Busca automática: se não foi passado --metadados-condicoes, procura
        # um "condicoes.csv" direto na pasta base (--data_dir). Evita ter que
        # digitar o caminho toda vez se o arquivo já mora ao lado dos dados.
        candidato = raiz_path / "condicoes.csv"
        if candidato.exists():
            caminho_metadados_condicoes = candidato
            print(f"📋 Encontrado condicoes.csv na pasta base, usando automaticamente: {candidato.resolve()}")

    if caminho_metadados_condicoes:
        try:
            metadados = ler_metadados_condicoes(caminho_metadados_condicoes)
            print(f"📋 Metadados de condição carregados: {caminho_metadados_condicoes.resolve()} ({len(metadados)} condição(ões))")
        except Exception as e:
            print(f"⚠️ Não foi possível ler {caminho_metadados_condicoes} ({e}). Seguindo sem metadados.")
            metadados = {}

    # Agrupa por sensor, preservando a ordem de aparição das condições
    sensores = {}
    for sensor, condicao, caminho_parquet in grupos:
        sensores.setdefault(sensor, {})[condicao] = caminho_parquet

    lista_sensores = list(sensores.keys())
    if args.quick:
        lista_sensores = lista_sensores[:1]
        print("⚡ Modo rápido (--quick): processando apenas o primeiro sensor.\n")

    pasta_figuras_raiz = raiz_path / "DadosTratados" / "Figuras"
    output_dir = raiz_path / "DadosTratados" / "Etapas" / "Heatmap"
    pasta_picos_raiz = raiz_path / "DadosTratados" / "Etapas" / "Picos"

    pastas_alteradas = {output_dir}
    sensores_ok, sensores_com_erro = 0, 0

    print(f"⚙️ Gerando mapa espectral por condição para {len(lista_sensores)} tipo(s) de sensor...\n")

    for sensor in lista_sensores:
        condicoes_disponiveis = sorted(sensores[sensor].keys())
        print(f"\n📖 Sensor: [{sensor}]  ←  {len(condicoes_disponiveis)} condição(ões): {condicoes_disponiveis}")

        # --- decide o modo do eixo Y para ESTE sensor (tudo ou nada, ver README) ---
        usar_eixo_continuo = False
        condicoes_sem_f_vfd = []
        if metadados:
            for c in condicoes_disponiveis:
                if metadados.get(c, {}).get("f_vfd_hz") is None:
                    condicoes_sem_f_vfd.append(c)
            usar_eixo_continuo = len(condicoes_sem_f_vfd) == 0

            if metadados and not usar_eixo_continuo:
                print(f"   ℹ️ Faltam f_vfd_hz para {condicoes_sem_f_vfd}; usando eixo categórico "
                      f"para o sensor [{sensor}] (sem linhas teóricas).")

        if usar_eixo_continuo:
            condicoes_ordenadas = sorted(condicoes_disponiveis, key=lambda c: metadados[c]["f_vfd_hz"])
            valores_y = [metadados[c]["f_vfd_hz"] for c in condicoes_ordenadas]
            rotulo_eixo_y = "VFD Frequency (Hz)"
        else:
            condicoes_ordenadas = condicoes_disponiveis  # já ordenado (sorted acima)
            valores_y = list(range(len(condicoes_ordenadas)))
            rotulo_eixo_y = "Condition"

        # --- carrega os espectros de todas as condições desse sensor ---
        espectros_por_canal = {}  # canal -> {condicao: (freqs, amplitude)}
        for condicao in condicoes_ordenadas:
            try:
                df_fft = carregar_grupo(sensores[sensor][condicao])
            except Exception as e:
                print(f"   ⚠️ Erro ao carregar {sensor}/{condicao}: {e}. Condição pulada.")
                continue

            if "freq_hz" not in df_fft.columns:
                print(f"   ⚠️ {sensor}/{condicao}: sem coluna 'freq_hz'. Condição pulada.")
                continue

            freqs = df_fft["freq_hz"].to_numpy()
            for col in df_fft.columns:
                if not str(col).endswith("_amplitude"):
                    continue
                canal = str(col)[:-len("_amplitude")]
                espectros_por_canal.setdefault(canal, {})[condicao] = (freqs, df_fft[col].to_numpy())

        if not espectros_por_canal:
            print(f"   ⚠️ Nenhum canal com dado válido para o sensor [{sensor}]. Pulando sensor.")
            sensores_com_erro += 1
            continue

        pasta_figuras = pasta_figuras_raiz / str(sensor) / "Heatmap"
        pasta_figuras.mkdir(parents=True, exist_ok=True)
        pastas_alteradas.add(pasta_figuras)

        reducao_shaft, reducao_cavidade = obter_reducoes(metadados, condicoes_ordenadas) if usar_eixo_continuo else (None, None)

        houve_erro_no_sensor = False

        for canal, espectros in espectros_por_canal.items():
            condicoes_com_dado = [c for c in condicoes_ordenadas if c in espectros]
            if len(condicoes_com_dado) < 2:
                print(f"   ⚠️ Canal {canal}: menos de 2 condições com dado válido, pulando (heatmap precisa de várias condições).")
                houve_erro_no_sensor = True
                continue

            valores_y_canal = [valores_y[condicoes_ordenadas.index(c)] for c in condicoes_com_dado]

            freq_max_dados = min(freqs.max() for freqs, _ in espectros.values())
            freq_max = args.freq_max if args.freq_max is not None else freq_max_dados
            freq_grid = np.arange(0.0, freq_max, args.freq_resolucao)

            matrix_raw = construir_matriz(condicoes_com_dado, espectros, freq_grid)

            # --- salva a matriz consolidada COMPLETA (RAW, não escalada, banda inteira) para reuso futuro ---
            df_heatmap = pd.DataFrame(
                matrix_raw,
                index=pd.Index(condicoes_com_dado, name="condicao"),
                columns=freq_grid,
            ).reset_index().melt(id_vars="condicao", var_name="freq_hz", value_name="amplitude")
            df_heatmap["y_valor"] = df_heatmap["condicao"].map(dict(zip(condicoes_com_dado, valores_y_canal)))
            nome_canal_arquivo = sanitizar_nome(canal)
            salvar_grupo(df_heatmap, sensor, nome_canal_arquivo, output_dir)
            print(f"      💾 Matriz salva: Etapas/Heatmap/{sensor}/{nome_canal_arquivo}.parquet")

            reducao_shaft, reducao_cavidade = obter_reducoes(metadados, condicoes_com_dado) if usar_eixo_continuo else (None, None)

            # --- 3 figuras por canal: low/mid/high, cada uma com sua PRÓPRIA escala de cor
            # (calculada só com os dados daquela faixa — assim uma faixa muito mais forte
            # não "afoga" visualmente as outras duas, ver discussão no README) ---
            faixas = [
                (0.0, args.f1, "low"),
                (args.f1, args.f2, "mid"),
                (args.f2, freq_max, "high"),
            ]

            for f_min, f_max, rotulo_faixa in faixas:
                mascara_freq = (freq_grid >= f_min) & (freq_grid <= f_max)
                if not mascara_freq.any():
                    print(f"      ⚠️ Canal {canal} | faixa {rotulo_faixa}: sem dado nessa faixa, pulando figura.")
                    continue

                freq_grid_faixa = freq_grid[mascara_freq]
                matrix_plot, v_min, v_max, label_cbar = converter_escala(
                    matrix_raw[:, mascara_freq], args.escala, args.db_min
                )

                fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
                bordas_x = _bordas_a_partir_de_centros(freq_grid_faixa)
                bordas_y = _bordas_a_partir_de_centros(valores_y_canal)
                im = ax.pcolormesh(
                    bordas_x, bordas_y, matrix_plot, cmap=args.cmap, vmin=v_min, vmax=v_max,
                    shading="flat",
                )
                cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.14, aspect=50)
                cbar.set_label(label_cbar, fontsize=11, labelpad=6)

                if not usar_eixo_continuo:
                    ax.set_yticks(valores_y_canal)
                    ax.set_yticklabels(condicoes_com_dado)

                # --- linhas teóricas (só com eixo contínuo + reduções conhecidas) ---
                if usar_eixo_continuo and reducao_shaft and reducao_cavidade:
                    y_cont = np.linspace(min(valores_y_canal), max(valores_y_canal), 200)
                    ax.plot(y_cont, y_cont, c="black", alpha=0.6, linewidth=2, label="1X Motor / VFD")
                    ax.plot(y_cont / reducao_shaft, y_cont, c="red", alpha=0.6, linewidth=2, label="1X Pump Shaft")
                    f_cav = y_cont / reducao_cavidade
                    ax.plot(f_cav, y_cont, c="red", alpha=0.7, linestyle="--", linewidth=2, label="1X Cavity")
                    ax.plot(2 * f_cav, y_cont, c="red", alpha=0.7, linestyle="-.", linewidth=2, label="2X Cavity")
                    ax.plot(4 * f_cav, y_cont, c="red", alpha=0.7, linestyle=":", linewidth=2, label="4X Cavity")

                # --- overlay dos picos dessa faixa (Etapas/Picos), se disponível ---
                if not args.sem_picos:
                    freqs_overlay, y_overlay = [], []
                    for condicao, y_valor in zip(condicoes_com_dado, valores_y_canal):
                        caminho_picos = pasta_picos_raiz / str(sensor) / f"{condicao}.parquet"
                        if not caminho_picos.exists():
                            continue
                        try:
                            df_picos = pd.read_parquet(caminho_picos)
                        except Exception:
                            continue
                        mascara = (df_picos["canal"] == canal) & (df_picos["escopo"] == rotulo_faixa)
                        for f_p in df_picos.loc[mascara, "freq_hz"]:
                            freqs_overlay.append(f_p)
                            y_overlay.append(y_valor)

                    if freqs_overlay:
                        ax.scatter(freqs_overlay, y_overlay, facecolors="none", edgecolors=COR_PICOS[rotulo_faixa],
                                   marker="o", s=40, linewidths=1.2, alpha=0.9, zorder=5,
                                   label=f"Peaks ({rotulo_faixa})")

                My_axis(
                    ax, font=13,
                    xlim=[freq_grid_faixa[0], freq_grid_faixa[-1]],
                    ylim=[min(valores_y_canal), max(valores_y_canal)] if len(valores_y_canal) > 1 else [-0.5, 0.5],
                    setaxis=[f"Spectral Map ({rotulo_faixa}) - {sensor} | {canal} | {f_min:.0f}-{f_max:.0f} Hz\n",
                             "Frequency (Hz)", rotulo_eixo_y],
                    legbox=[0.98, 0.98, 1, 10],
                )

                nome_figura = f"heatmap_{nome_canal_arquivo}_{rotulo_faixa}_{f_min:.0f}-{f_max:.0f}hz.png"
                caminho_figura = pasta_figuras / nome_figura
                plt.tight_layout()
                plt.savefig(caminho_figura, dpi=150)
                plt.close(fig)

                print(f"      🖼️ Figura salva: Figuras/{sensor}/Heatmap/{nome_figura}")

        sensores_ok += 1
        if houve_erro_no_sensor:
            sensores_com_erro += 1

    caminho_log = registrar_log(raiz_path, "05_heatmap", {
        "data_dir": raiz_path.resolve(),
        "metadados_condicoes": str(caminho_metadados_condicoes.resolve()) if caminho_metadados_condicoes else None,
        "escala": args.escala,
        "metodo_escala": METODO_ESCALA_DESCRICAO.get(args.escala, ""),
        "db_min": args.db_min if args.escala in ("db", "db-global") else None,
        "cmap": args.cmap,
        "freq_max": args.freq_max,
        "freq_resolucao_hz": args.freq_resolucao,
        "f1_hz": args.f1,
        "f2_hz": args.f2,
        "sobrepor_picos": not args.sem_picos,
        "quick": args.quick,
        "tipos_de_sensor_processados": sensores_ok,
        "tipos_de_sensor_com_aviso": sensores_com_erro,
    }, pastas_alteradas=pastas_alteradas)

    print("\n" + "=" * 65)
    print(f"✅ Etapa 05 (Heatmap) Concluída!")
    print(f"   Tipos de sensor processados: {sensores_ok} | Tipos de sensor com algum aviso: {sensores_com_erro}")
    print(f"💾 Matrizes salvas em: {output_dir.resolve()}")
    print(f"📝 Log de parâmetros: {caminho_log.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
