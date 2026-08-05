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
    """Converte a matriz RAW pra 'norm' (0-1 por linha) ou 'db' (relativo ao pico de cada linha)."""
    eps = 1e-12
    if modo_escala == "norm":
        picos = np.max(matrix_raw, axis=1, keepdims=True) + eps
        dados = matrix_raw / picos
        return dados, 0.0, 1.0, "FFT Amplitude - Normalized (per condition)"
    if modo_escala == "db":
        picos = np.max(matrix_raw, axis=1, keepdims=True) + eps
        dados = 20 * np.log10((matrix_raw / picos) + eps)
        return dados, db_min, 0.0, "FFT Amplitude (dB, relative to peak per condition)"
    return matrix_raw, float(np.min(matrix_raw)), float(np.max(matrix_raw)), "FFT Amplitude - Raw"


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
    parser.add_argument("--escala", choices=["norm", "db", "raw"], default="norm",
                         help="Escala de cor do heatmap (padrão: norm).")
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
    if args.metadados_condicoes:
        caminho_csv = Path(args.metadados_condicoes)
        try:
            metadados = ler_metadados_condicoes(caminho_csv)
            print(f"📋 Metadados de condição carregados: {caminho_csv.resolve()} ({len(metadados)} condição(ões))")
        except Exception as e:
            print(f"⚠️ Não foi possível ler --metadados-condicoes ({e}). Seguindo sem metadados.")
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

    print(f"⚙️ Gerando mapa espectral por condição para {len(lista_sensores)} sensor(es)...\n")

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
            matrix_plot, v_min, v_max, label_cbar = converter_escala(matrix_raw, args.escala, args.db_min)

            # --- salva a matriz consolidada (RAW, não escalada) para reuso futuro ---
            df_heatmap = pd.DataFrame(
                matrix_raw,
                index=pd.Index(condicoes_com_dado, name="condicao"),
                columns=freq_grid,
            ).reset_index().melt(id_vars="condicao", var_name="freq_hz", value_name="amplitude")
            df_heatmap["y_valor"] = df_heatmap["condicao"].map(dict(zip(condicoes_com_dado, valores_y_canal)))
            nome_canal_arquivo = sanitizar_nome(canal)
            salvar_grupo(df_heatmap, sensor, nome_canal_arquivo, output_dir)

            # --- figura ---
            fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
            im = ax.imshow(
                matrix_plot, aspect="auto", origin="lower", cmap=args.cmap,
                extent=[freq_grid[0], freq_grid[-1], min(valores_y_canal), max(valores_y_canal)]
                if len(valores_y_canal) > 1 else [freq_grid[0], freq_grid[-1], -0.5, 0.5],
                vmin=v_min, vmax=v_max, interpolation="nearest",
            )
            cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.14, aspect=50)
            cbar.set_label(label_cbar, fontsize=11, labelpad=6)

            if not usar_eixo_continuo:
                ax.set_yticks(valores_y_canal)
                ax.set_yticklabels(condicoes_com_dado)

            # --- linhas de referência low/mid boundary (sempre) ---
            ax.axvline(args.f1, color="white", linestyle=":", linewidth=1.2, alpha=0.7)
            ax.axvline(args.f2, color="white", linestyle=":", linewidth=1.2, alpha=0.7)

            # --- linhas teóricas (só com eixo contínuo + reduções conhecidas) ---
            if usar_eixo_continuo and reducao_shaft and reducao_cavidade:
                y_cont = np.linspace(min(valores_y_canal), max(valores_y_canal), 200)
                ax.plot(y_cont, y_cont, c="black", alpha=0.6, linewidth=2, label="1X Motor / VFD")
                ax.plot(y_cont / reducao_shaft, y_cont, c="red", alpha=0.6, linewidth=2, label="1X Pump Shaft")
                f_cav = y_cont / reducao_cavidade
                ax.plot(f_cav, y_cont, c="red", alpha=0.7, linestyle="--", linewidth=2, label="1X Cavity")
                ax.plot(2 * f_cav, y_cont, c="red", alpha=0.7, linestyle="-.", linewidth=2, label="2X Cavity")
                ax.plot(4 * f_cav, y_cont, c="red", alpha=0.7, linestyle=":", linewidth=2, label="4X Cavity")

            # --- overlay dos picos (Etapas/Picos), se disponível ---
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
                    mascara = (df_picos["canal"] == canal) & (df_picos["escopo"].isin(["low", "mid", "high"]))
                    for f_p in df_picos.loc[mascara, "freq_hz"]:
                        freqs_overlay.append(f_p)
                        y_overlay.append(y_valor)

                if freqs_overlay:
                    ax.scatter(freqs_overlay, y_overlay, facecolors="none", edgecolors="red",
                               marker="o", s=40, linewidths=1.2, alpha=0.85, zorder=5,
                               label="Peaks (low/mid/high)")
                else:
                    print(f"      ℹ️ Canal {canal}: nenhum pico encontrado em Etapas/Picos para sobrepor "
                          f"(rode a etapa 04 antes, se quiser essa camada).")

            My_axis(
                ax, font=13,
                xlim=[freq_grid[0], freq_grid[-1]],
                ylim=[min(valores_y_canal), max(valores_y_canal)] if len(valores_y_canal) > 1 else [-0.5, 0.5],
                setaxis=[f"Spectral Map - {sensor} | {canal}\n", "Frequency (Hz)", rotulo_eixo_y],
                legbox=[0.98, 0.98, 1, 10],
            )

            nome_figura = f"heatmap_{nome_canal_arquivo}.png"
            caminho_figura = pasta_figuras / nome_figura
            plt.tight_layout()
            plt.savefig(caminho_figura, dpi=150)
            plt.close(fig)

            print(f"      🖼️ Figura salva: Figuras/{sensor}/Heatmap/{nome_figura}")
            print(f"      💾 Matriz salva: Etapas/Heatmap/{sensor}/{nome_canal_arquivo}.parquet")

        sensores_ok += 1
        if houve_erro_no_sensor:
            sensores_com_erro += 1

    caminho_log = registrar_log(raiz_path, "05_heatmap", {
        "data_dir": raiz_path.resolve(),
        "metadados_condicoes": str(Path(args.metadados_condicoes).resolve()) if args.metadados_condicoes else None,
        "escala": args.escala,
        "db_min": args.db_min if args.escala == "db" else None,
        "cmap": args.cmap,
        "freq_max": args.freq_max,
        "freq_resolucao_hz": args.freq_resolucao,
        "f1_hz": args.f1,
        "f2_hz": args.f2,
        "sobrepor_picos": not args.sem_picos,
        "quick": args.quick,
        "sensores_processados": sensores_ok,
        "sensores_com_aviso": sensores_com_erro,
    }, pastas_alteradas=pastas_alteradas)

    print("\n" + "=" * 65)
    print(f"✅ Etapa 05 (Heatmap) Concluída!")
    print(f"   Sensores processados: {sensores_ok} | Sensores com algum aviso: {sensores_com_erro}")
    print(f"💾 Matrizes salvas em: {output_dir.resolve()}")
    print(f"📝 Log de parâmetros: {caminho_log.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
