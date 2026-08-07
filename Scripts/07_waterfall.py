from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — necessário para registrar projection="3d"
from matplotlib.collections import PolyCollection


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
registrar_log = _pipeline_io.registrar_log
filtrar_desde_condicao = _pipeline_io.filtrar_desde_condicao

_espectro = _carregar_modulo("espectro", "espectro.py")
sanitizar_nome = _espectro.sanitizar_nome
ler_metadados_condicoes = _espectro.ler_metadados_condicoes
construir_matriz = _espectro.construir_matriz
converter_escala = _espectro.converter_escala


# ==============================================================================
# 🛠️ PLOTAGEM 3D (uma linha "cascata" por condição)
# ==============================================================================
def plotar_waterfall(matrix_plot: np.ndarray, freq_grid_faixa: np.ndarray, valores_y: list,
                      rotulos_y_ticks: list, titulo: str, rotulo_eixo_y: str, label_z: str,
                      cmap_nome: str, elev: float, azim: float, caminho_figura: Path):
    """
    Uma linha 3D por condição (cascata), preenchimento translúcido abaixo de
    cada linha pra reforçar a leitura de "camadas" empilhadas — mesmo
    princípio visual do protótipo em src/waterfall.py, agora alimentado
    pelos dados já calibrados/consolidados da etapa 03 (Etapas/FFT), em vez
    de recalcular a FFT a partir do sinal bruto.
    """
    fig = plt.figure(figsize=(12, 9), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    colormap = plt.get_cmap(cmap_nome)
    v_min_cor, v_max_cor = float(np.min(matrix_plot)), float(np.max(matrix_plot))
    if v_max_cor <= v_min_cor:
        v_max_cor = v_min_cor + 1e-9
    norm_cor = plt.Normalize(v_min_cor, v_max_cor)

    for i, y_val in enumerate(valores_y):
        z_val = matrix_plot[i, :]
        cor = colormap(norm_cor(np.max(z_val)))
        ax.plot(freq_grid_faixa, [y_val] * len(freq_grid_faixa), z_val, color=cor, linewidth=1.5, zorder=len(valores_y) - i)

        # Preenchimento translúcido abaixo da linha (efeito "cascata sólida").
        # plt.fill_between()/gca() não serve aqui: com o Axes3D já ativo,
        # 'fill_between' vira o método 3D (assinatura diferente) e quebra.
        # A forma robusta é montar o polígono (x, z) manualmente e projetá-lo
        # no plano da condição via add_collection3d(zs=y_val, zdir='y').
        base = float(np.min(matrix_plot))
        verts = [(freq_grid_faixa[0], base)] + list(zip(freq_grid_faixa, z_val)) + [(freq_grid_faixa[-1], base)]
        poligono = PolyCollection([verts], facecolors=[cor], alpha=0.15, edgecolors="none")
        ax.add_collection3d(poligono, zs=y_val, zdir="y")

    ax.set_xlabel("\nFrequency (Hz)", fontsize=12, labelpad=10)
    ax.set_ylabel(f"\n{rotulo_eixo_y}", fontsize=12, labelpad=10)
    ax.set_zlabel(f"\n{label_z}", fontsize=11, labelpad=10)
    ax.set_title(titulo, fontsize=13, fontweight="bold", pad=0)

    ax.set_xlim(freq_grid_faixa[0], freq_grid_faixa[-1])
    ax.set_ylim(min(valores_y), max(valores_y))
    ax.set_zlim(np.min(matrix_plot), np.max(matrix_plot))
    ax.set_yticks(valores_y)
    ax.set_yticklabels(rotulos_y_ticks)

    ax.view_init(elev=elev, azim=azim)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("gray")
    ax.yaxis.pane.set_edgecolor("gray")
    ax.zaxis.pane.set_edgecolor("gray")

    plt.savefig(caminho_figura, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# 🚀 EXECUÇÃO PRINCIPAL
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--quick", action="store_true",
                         help="Processa apenas o primeiro sensor encontrado, para teste rápido.")
    parser.add_argument("--from", dest="from_condicao", type=str, default=None,
                         help="Inclui na comparação só as condições a partir desta, inclusive (ex.: "
                              "--from T3 usa T3, T4, T5... e ignora T1/T2). Extrai o número do padrão "
                              "T<N> no nome da condição; nomes fora desse padrão nunca são descartados.")
    parser.add_argument("--metadados-condicoes", type=str, default=None,
                         help="Caminho de um CSV opcional (condicao,f_vfd_hz,vazao_m3h,reducao_shaft,"
                              "reducao_cavidade) — mesmo formato/arquivo usado pela etapa 05. Se omitido, "
                              "procura automaticamente um 'condicoes.csv' na raiz de --data_dir. Sem "
                              "metadados, o eixo Y usa o rótulo categórico da condição (T1..Tn).")
    parser.add_argument("--escala", choices=["db-global", "abs-global", "abs-condicao", "pico-canal", "rms-canal", "db"],
                         default="db-global",
                         help="Escala do eixo Z (mesmas 6 opções e mesma matemática da etapa 05 — ver "
                              "README). Padrão: db-global.")
    parser.add_argument("--db-min", type=float, default=-40.0,
                         help="Piso (dB) usado quando --escala db ou db-global (padrão: -40.0).")
    parser.add_argument("--cmap", type=str, default="jet",
                         help="Colormap do matplotlib usado para colorir cada linha/camada pela sua "
                              "própria amplitude (padrão: jet — visual clássico de waterfall/cascata).")
    parser.add_argument("--freq-max", type=float, default=None,
                         help="Frequência máxima (Hz) do waterfall. Padrão: a menor frequência máxima "
                              "entre as condições daquele sensor/canal (evita extrapolar).")
    parser.add_argument("--freq-resolucao", type=float, default=0.5,
                         help="Resolução (Hz) do grid comum de frequência usado para interpolar as "
                              "condições (padrão: 0.5).")
    parser.add_argument("--elev", type=float, default=25.0,
                         help="Elevação (graus) da câmera 3D (padrão: 25.0).")
    parser.add_argument("--azim", type=float, default=-60.0,
                         help="Azimute (graus) da câmera 3D (padrão: -60.0).")
    args = parser.parse_args()

    raiz_path = Path(args.data_dir)

    # Lê direto de Etapas/FFT (saída da etapa 03) — NÃO depende da etapa 05
    # (heatmap) nem de nenhuma etapa depois dela. As duas (05 e 07) são
    # irmãs: consomem o mesmo insumo, cada uma pode rodar antes, depois ou
    # ao mesmo tempo que a outra, e alterar uma não exige reprocessar a
    # outra — só depende de já ter rodado 01→02→03 antes.
    input_dir = raiz_path / "DadosTratados" / "Etapas" / "FFT"
    grupos = listar_grupos(input_dir)

    if not grupos:
        print(f"❌ Nenhum grupo encontrado em: {input_dir.resolve()}")
        print(" Certifique-se de executar a etapa 03_fft antes desta (não precisa rodar 04/05/06).")
        exit(1)

    metadados = {}
    caminho_metadados_condicoes = None
    if args.metadados_condicoes:
        caminho_metadados_condicoes = Path(args.metadados_condicoes)
    else:
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

    sensores = {}
    for sensor, condicao, caminho_parquet in grupos:
        sensores.setdefault(sensor, {})[condicao] = caminho_parquet

    lista_sensores = list(sensores.keys())
    if args.quick:
        lista_sensores = lista_sensores[:1]
        print("⚡ Modo rápido (--quick): processando apenas o primeiro sensor.\n")

    pasta_figuras_raiz = raiz_path / "DadosTratados" / "Figuras"
    pastas_alteradas = set()
    sensores_ok, sensores_com_erro = 0, 0

    print(f"⚙️ Gerando waterfall 3D para {len(lista_sensores)} tipo(s) de sensor...\n")

    for sensor in lista_sensores:
        condicoes_disponiveis = sorted(sensores[sensor].keys())
        condicoes_disponiveis = filtrar_desde_condicao(condicoes_disponiveis, args.from_condicao)
        if args.from_condicao and not condicoes_disponiveis:
            print(f"   ℹ️ Sensor [{sensor}]: nenhuma condição >= {args.from_condicao}, pulando sensor.")
            continue
        print(f"\n📖 Sensor: [{sensor}]  ←  {len(condicoes_disponiveis)} condição(ões): {condicoes_disponiveis}")

        # --- mesmo critério "tudo ou nada" da etapa 05 pro eixo Y contínuo ---
        usar_eixo_continuo = False
        condicoes_sem_f_vfd = []
        if metadados:
            for c in condicoes_disponiveis:
                if metadados.get(c, {}).get("f_vfd_hz") is None:
                    condicoes_sem_f_vfd.append(c)
            usar_eixo_continuo = len(condicoes_sem_f_vfd) == 0
            if metadados and not usar_eixo_continuo:
                print(f"   ℹ️ Faltam f_vfd_hz para {condicoes_sem_f_vfd}; usando eixo categórico "
                      f"para o sensor [{sensor}].")

        if usar_eixo_continuo:
            condicoes_ordenadas = sorted(condicoes_disponiveis, key=lambda c: metadados[c]["f_vfd_hz"])
            valores_reais = [metadados[c]["f_vfd_hz"] for c in condicoes_ordenadas]
            rotulo_eixo_y = "VFD Frequency (Hz)"
        else:
            condicoes_ordenadas = condicoes_disponiveis
            valores_reais = None
            rotulo_eixo_y = "Condition"

        # Posições do eixo Y sempre em bandas de altura igual (0,1,2...) —
        # mesma decisão de design da etapa 05: os valores reais (f_vfd_hz)
        # são pontos discretos escolhidos no ensaio, não uma variável
        # amostrada continuamente; usá-los como posição distorceria o
        # espaçamento. Eles só entram como RÓTULO do tick.
        valores_y = list(range(len(condicoes_ordenadas)))
        rotulos_y_ticks = [f"{v:g}" for v in valores_reais] if usar_eixo_continuo else condicoes_ordenadas

        espectros_por_canal = {}
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

        pasta_figuras = pasta_figuras_raiz / str(sensor) / "Waterfall"
        pasta_figuras.mkdir(parents=True, exist_ok=True)
        pastas_alteradas.add(pasta_figuras)

        houve_erro_no_sensor = False

        for canal, espectros in espectros_por_canal.items():
            condicoes_com_dado = [c for c in condicoes_ordenadas if c in espectros]
            if len(condicoes_com_dado) < 2:
                print(f"   ⚠️ Canal {canal}: menos de 2 condições com dado válido, pulando "
                      f"(waterfall precisa de várias condições).")
                houve_erro_no_sensor = True
                continue

            valores_y_canal = [valores_y[condicoes_ordenadas.index(c)] for c in condicoes_com_dado]
            rotulos_y_canal = [rotulos_y_ticks[condicoes_ordenadas.index(c)] for c in condicoes_com_dado]

            freq_max_dados = min(freqs.max() for freqs, _ in espectros.values())
            freq_max = args.freq_max if args.freq_max is not None else freq_max_dados
            freq_grid = np.arange(0.0, freq_max, args.freq_resolucao)

            matrix_raw = construir_matriz(condicoes_com_dado, espectros, freq_grid)
            nome_canal_arquivo = sanitizar_nome(canal)

            matrix_plot, _v_min, _v_max, label_z = converter_escala(matrix_raw, args.escala, args.db_min)

            titulo = f"Waterfall - {sensor} | {canal} | 0-{freq_max:.0f} Hz"
            nome_figura = f"waterfall_{nome_canal_arquivo}_0-{freq_max:.0f}hz.png"
            caminho_figura = pasta_figuras / nome_figura

            plotar_waterfall(
                matrix_plot, freq_grid, valores_y_canal, rotulos_y_canal,
                titulo, rotulo_eixo_y, label_z, args.cmap, args.elev, args.azim, caminho_figura,
            )
            print(f"      🖼️ Figura salva: Figuras/{sensor}/Waterfall/{nome_figura}")

        sensores_ok += 1
        if houve_erro_no_sensor:
            sensores_com_erro += 1

    caminho_log = registrar_log(raiz_path, "07_waterfall", {
        "data_dir": raiz_path.resolve(),
        "metadados_condicoes": str(caminho_metadados_condicoes.resolve()) if caminho_metadados_condicoes else None,
        "escala": args.escala,
        "db_min": args.db_min if args.escala in ("db", "db-global") else None,
        "cmap": args.cmap,
        "freq_max": args.freq_max,
        "freq_resolucao_hz": args.freq_resolucao,
        "elev": args.elev,
        "azim": args.azim,
        "quick": args.quick,
        "from_condicao": args.from_condicao,
        "tipos_de_sensor_processados": sensores_ok,
        "tipos_de_sensor_com_aviso": sensores_com_erro,
    }, pastas_alteradas=pastas_alteradas)

    print("\n" + "=" * 65)
    print(f"✅ Etapa 07 (Waterfall) Concluída!")
    print(f"   Tipos de sensor processados: {sensores_ok} | Tipos de sensor com algum aviso: {sensores_com_erro}")
    print(f"💾 Figuras salvas em: {pasta_figuras_raiz.resolve()}")
    print(f"📝 Log de parâmetros: {caminho_log.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
