from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
filtrar_desde_condicao = _pipeline_io.filtrar_desde_condicao

_estilo = _carregar_modulo("estilo_grafico", "estilo_grafico.py")
My_axis = _estilo.My_axis

_espectro = _carregar_modulo("espectro", "espectro.py")
sanitizar_nome = _espectro.sanitizar_nome
ler_metadados_condicoes = _espectro.ler_metadados_condicoes
construir_matriz = _espectro.construir_matriz

COR_BANDA = {"broadband_tempo": "#212121", "low": "#2979FF", "mid": "#00C853", "high": "#FF1744"}
MARCADOR_BANDA = {"broadband_tempo": "o", "low": "s", "mid": "^", "high": "D"}

METODO_DESCRICAO = (
    "RMS de banda larga (broadband): sqrt(mean(sinal**2)) calculado DIRETO no "
    "domínio do tempo, sobre o sinal já calibrado/tratado da etapa 02 (Etapas/"
    "Preprocessamento) — não depende de FFT, janela ou nperseg. É o RMS 'oficial' "
    "de severidade de vibração (base de normas tipo ISO 10816/20816). "
    "RMS por banda (low/mid/high) e PSD (densidade espectral): DERIVADOS do "
    "espectro que a etapa 03 já calculou (scipy.signal.welch, scaling='spectrum', "
    "salvo em Etapas/FFT como amplitude=sqrt(Pxx)) — SEM recalcular Welch. Pelo "
    "teorema de Parseval, a soma de amplitude**2 nos bins de uma faixa é o RMS**2 "
    "daquela faixa; dividindo Pxx (=amplitude**2) pela resolução em Hz (df) dá a "
    "densidade espectral de potência (PSD, em unidade**2/Hz) normalizada, "
    "comparável entre ensaios com configurações de Welch (nperseg/janela) "
    "diferentes — ao contrário do espectro de amplitude cru da etapa 03. RMS por "
    "banda e PSD são portanto duas leituras da mesma informação: PSD é a "
    "'densidade', RMS-por-banda é a PSD integrada naquela faixa."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--quick", action="store_true",
                         help="Processa apenas o primeiro sensor encontrado, para teste rápido.")
    parser.add_argument("--from", dest="from_condicao", type=str, default=None,
                         help="Inclui no agregado só as condições a partir desta, inclusive (ex.: "
                              "--from T3 usa T3, T4, T5... e ignora T1/T2). Extrai o número do padrão "
                              "T<N> no nome da condição; nomes fora desse padrão nunca são descartados.")
    parser.add_argument("--metadados-condicoes", type=str, default=None,
                         help="Caminho de um CSV opcional (condicao,f_vfd_hz,vazao_m3h,reducao_shaft,"
                              "reducao_cavidade) — mesmo formato/arquivo usado pelas etapas 05/07. Se "
                              "omitido, procura automaticamente um 'condicoes.csv' na raiz de --data_dir. "
                              "Sem metadados, o eixo X do gráfico de tendência usa o rótulo categórico "
                              "da condição (T1..Tn).")
    parser.add_argument("--f1", type=float, default=15.0,
                         help="Limite entre a faixa LOW e MID, em Hz (padrão: 15.0). Mesma convenção "
                              "de faixas do resto do pipeline.")
    parser.add_argument("--f2", type=float, default=400.0,
                         help="Limite entre a faixa MID e HIGH, em Hz (padrão: 400.0).")
    parser.add_argument("--freq-resolucao", type=float, default=0.5,
                         help="Resolução (Hz) do grid comum de frequência usado para interpolar as "
                              "condições no gráfico de PSD sobreposta (padrão: 0.5). Só afeta o gráfico "
                              "de PSD — o RMS por banda usa sempre a resolução nativa de cada condição, "
                              "sem interpolar.")
    parser.add_argument("--cmap", type=str, default="tab10",
                         help="Colormap qualitativo do matplotlib usado para colorir uma linha por "
                              "condição no gráfico de PSD sobreposta (padrão: tab10).")
    args = parser.parse_args()

    raiz_path = Path(args.data_dir)

    # Lê de Etapas/FFT (etapa 03, RMS por banda + PSD) e Etapas/Preprocessamento
    # (etapa 02, RMS de banda larga no tempo) — não depende de 04/05/06/07/08.
    input_dir_fft = raiz_path / "DadosTratados" / "Etapas" / "FFT"
    input_dir_prep = raiz_path / "DadosTratados" / "Etapas" / "Preprocessamento"
    grupos = listar_grupos(input_dir_fft)

    if not grupos:
        print(f"❌ Nenhum grupo encontrado em: {input_dir_fft.resolve()}")
        print(" Certifique-se de executar as etapas 02_preprocessamento e 03_fft antes desta "
              "(não precisa rodar 04/05/06/07/08).")
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
    output_dir_rms = raiz_path / "DadosTratados" / "Etapas" / "RMS"
    output_dir_psd = raiz_path / "DadosTratados" / "Etapas" / "PSD"
    pastas_alteradas = {output_dir_rms, output_dir_psd}
    sensores_ok, sensores_com_erro = 0, 0

    print(f"⚙️ Calculando RMS/PSD para {len(lista_sensores)} tipo(s) de sensor...")
    print(f"   Método: {METODO_DESCRICAO}\n")

    for sensor in lista_sensores:
        condicoes_disponiveis = sorted(sensores[sensor].keys())
        condicoes_disponiveis = filtrar_desde_condicao(condicoes_disponiveis, args.from_condicao)
        if args.from_condicao and not condicoes_disponiveis:
            print(f"   ℹ️ Sensor [{sensor}]: nenhuma condição >= {args.from_condicao}, pulando sensor.")
            continue
        print(f"\n📖 Sensor: [{sensor}]  ←  {len(condicoes_disponiveis)} condição(ões): {condicoes_disponiveis}")

        # --- mesmo critério "tudo ou nada" pro eixo X contínuo (ver etapas 05/07) ---
        usar_eixo_continuo = False
        if metadados:
            condicoes_sem_f_vfd = [c for c in condicoes_disponiveis if metadados.get(c, {}).get("f_vfd_hz") is None]
            usar_eixo_continuo = len(condicoes_sem_f_vfd) == 0
            if not usar_eixo_continuo:
                print(f"   ℹ️ Faltam f_vfd_hz para {condicoes_sem_f_vfd}; usando eixo categórico "
                      f"para o sensor [{sensor}].")

        if usar_eixo_continuo:
            condicoes_ordenadas = sorted(condicoes_disponiveis, key=lambda c: metadados[c]["f_vfd_hz"])
            valores_x = [metadados[c]["f_vfd_hz"] for c in condicoes_ordenadas]
            rotulo_eixo_x = "VFD Frequency (Hz)"
        else:
            condicoes_ordenadas = condicoes_disponiveis
            valores_x = list(range(len(condicoes_ordenadas)))
            rotulo_eixo_x = "Condition"

        # --- carrega FFT (amplitude, escala 'spectrum') e Preprocessamento (sinal no tempo) por condição ---
        amplitude_por_canal = {}   # canal -> {condicao: (freqs, amplitude)}
        sinal_por_canal = {}       # canal -> {condicao: sinal_tratado}

        for condicao in condicoes_ordenadas:
            try:
                df_fft = carregar_grupo(sensores[sensor][condicao])
            except Exception as e:
                print(f"   ⚠️ Erro ao carregar FFT de {sensor}/{condicao}: {e}. Condição pulada.")
                continue
            if "freq_hz" not in df_fft.columns:
                print(f"   ⚠️ {sensor}/{condicao}: sem coluna 'freq_hz' em Etapas/FFT. Condição pulada.")
                continue
            freqs = df_fft["freq_hz"].to_numpy()
            for col in df_fft.columns:
                if not str(col).endswith("_amplitude"):
                    continue
                canal = str(col)[:-len("_amplitude")]
                amplitude_por_canal.setdefault(canal, {})[condicao] = (freqs, df_fft[col].to_numpy())

            caminho_prep = input_dir_prep / str(sensor) / f"{condicao}.parquet"
            if not caminho_prep.exists():
                print(f"   ⚠️ {sensor}/{condicao}: sem Etapas/Preprocessamento correspondente "
                      f"({caminho_prep.name}) — RMS de banda larga (tempo) não será calculado "
                      f"para esta condição.")
                continue
            df_prep = pd.read_parquet(caminho_prep)
            for col in df_prep.columns:
                if not str(col).endswith("_tratado"):
                    continue
                canal = str(col)[:-len("_tratado")]
                sinal_por_canal.setdefault(canal, {})[condicao] = df_prep[col].to_numpy()

        if not amplitude_por_canal:
            print(f"   ⚠️ Nenhum canal com dado válido para o sensor [{sensor}]. Pulando sensor.")
            sensores_com_erro += 1
            continue

        pasta_figuras_rms = pasta_figuras_raiz / str(sensor) / "RMS"
        pasta_figuras_psd = pasta_figuras_raiz / str(sensor) / "PSD"
        pasta_figuras_rms.mkdir(parents=True, exist_ok=True)
        pasta_figuras_psd.mkdir(parents=True, exist_ok=True)
        pastas_alteradas.add(pasta_figuras_rms)
        pastas_alteradas.add(pasta_figuras_psd)

        houve_erro_no_sensor = False

        for canal, espectros in amplitude_por_canal.items():
            condicoes_com_dado = [c for c in condicoes_ordenadas if c in espectros]
            if not condicoes_com_dado:
                continue
            nome_canal_arquivo = sanitizar_nome(canal)

            # ================= RMS: broadband (tempo) + por banda (espectro) =================
            linhas_rms = []
            for condicao in condicoes_com_dado:
                freqs, amplitude = espectros[condicao]
                nyquist = float(freqs.max()) if freqs.size else 0.0
                faixas = [(0.0, args.f1, "low"), (args.f1, args.f2, "mid"), (args.f2, nyquist, "high")]

                linha = {"condicao": condicao}
                for f_min, f_max, rotulo in faixas:
                    mascara = (freqs >= f_min) & (freqs <= f_max)
                    linha[f"rms_{rotulo}"] = float(np.sqrt(np.sum(amplitude[mascara] ** 2))) if mascara.any() else np.nan
                linha["rms_full_espectro"] = float(np.sqrt(np.sum(amplitude ** 2)))

                sinal_tempo = sinal_por_canal.get(canal, {}).get(condicao)
                linha["rms_broadband_tempo"] = float(np.sqrt(np.mean(sinal_tempo ** 2))) if sinal_tempo is not None else np.nan

                linhas_rms.append(linha)

            df_rms = pd.DataFrame(linhas_rms)
            salvar_grupo(df_rms, sensor, nome_canal_arquivo, output_dir_rms)
            print(f"      💾 RMS salvo: Etapas/RMS/{sensor}/{nome_canal_arquivo}.parquet")

            fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
            eixo_x_canal = [valores_x[condicoes_ordenadas.index(c)] for c in condicoes_com_dado]
            for coluna, rotulo_legenda in [("rms_broadband_tempo", "Broadband (tempo)"),
                                            ("rms_low", "Low"), ("rms_mid", "Mid"), ("rms_high", "High")]:
                escopo = coluna.replace("rms_", "")
                y = df_rms[coluna].to_numpy()
                if np.all(np.isnan(y)):
                    continue
                ax.plot(eixo_x_canal, y, marker=MARCADOR_BANDA.get(escopo, "o"),
                        color=COR_BANDA.get(escopo, "#607D8B"), linewidth=1.8, markersize=6,
                        label=rotulo_legenda)

            if usar_eixo_continuo:
                ax.set_xticks(eixo_x_canal)
            else:
                ax.set_xticks(eixo_x_canal)
                ax.set_xticklabels(condicoes_com_dado)

            y_max = np.nanmax(df_rms[["rms_broadband_tempo", "rms_low", "rms_mid", "rms_high"]].to_numpy())
            My_axis(
                ax, font=12,
                xlim=[min(eixo_x_canal) - 0.5, max(eixo_x_canal) + 0.5] if not usar_eixo_continuo
                     else [min(eixo_x_canal), max(eixo_x_canal)],
                ylim=[0, y_max * 1.15 if np.isfinite(y_max) and y_max > 0 else 1.0],
                legbox=[0.98, 0.98, 1, 10],
                setaxis=[f"RMS Trend - {sensor} | {canal}\n", rotulo_eixo_x, "RMS"],
            )
            ax.grid(True, axis="y", which="major", linestyle="--", alpha=0.5, color="gray", zorder=0)
            ax.set_axisbelow(True)

            nome_figura_rms = f"rms_trend_{nome_canal_arquivo}.png"
            plt.savefig(pasta_figuras_rms / nome_figura_rms, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"      🖼️ Figura salva: Figuras/{sensor}/RMS/{nome_figura_rms}")

            # ================= PSD sobreposta (todas as condições) =================
            freq_max_dados = min(freqs.max() for freqs, _ in espectros.values())
            freq_grid = np.arange(0.0, freq_max_dados, args.freq_resolucao)
            matrix_amplitude = construir_matriz(condicoes_com_dado, espectros, freq_grid)
            df_hz = freq_grid[1] - freq_grid[0] if len(freq_grid) > 1 else 1.0
            matrix_psd = (matrix_amplitude ** 2) / df_hz

            df_psd = pd.DataFrame({"freq_hz": freq_grid})
            for i, condicao in enumerate(condicoes_com_dado):
                df_psd[f"{condicao}_psd"] = matrix_psd[i, :]
            salvar_grupo(df_psd, sensor, nome_canal_arquivo, output_dir_psd)
            print(f"      💾 PSD salva: Etapas/PSD/{sensor}/{nome_canal_arquivo}.parquet")

            # Duas figuras por padrão: 'full' (espectro inteiro) e 'low'
            # (0-f1) — a faixa low costuma concentrar as tônicas de VFD/
            # hidráulicas, e no gráfico 'full' ela fica espremida perto do
            # eixo Y (escala linear em X cobrindo até dezenas de kHz).
            colormap = plt.get_cmap(args.cmap)
            figuras_psd = [(0.0, freq_max_dados, "full")]
            if args.f1 < freq_max_dados:
                figuras_psd.append((0.0, args.f1, "low"))

            for f_min, f_max, rotulo_faixa in figuras_psd:
                mascara_freq = (freq_grid >= f_min) & (freq_grid <= f_max)
                if not mascara_freq.any():
                    continue
                freq_grid_faixa = freq_grid[mascara_freq]
                matrix_psd_faixa = matrix_psd[:, mascara_freq]

                fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
                for i, condicao in enumerate(condicoes_com_dado):
                    ax.semilogy(freq_grid_faixa, np.maximum(matrix_psd_faixa[i, :], 1e-30),
                                color=colormap(i % colormap.N), linewidth=1.2, label=condicao)

                My_axis(
                    ax, font=12,
                    xlim=[f_min, f_max],
                    ylim=[max(float(np.min(matrix_psd_faixa[matrix_psd_faixa > 0])) * 0.5, 1e-30),
                          float(np.max(matrix_psd_faixa)) * 2],
                    legbox=[0.98, 0.98, 2, 9],
                    logy=True,
                    setaxis=[f"PSD ({rotulo_faixa}) - {sensor} | {canal} | {f_min:.0f}-{f_max:.0f} Hz\n",
                             "Frequency (Hz)", "PSD (unit²/Hz)"],
                )
                ax.grid(True, which="major", linestyle="--", alpha=0.4, color="gray", zorder=0)
                ax.set_axisbelow(True)

                nome_figura_psd = f"psd_overlay_{nome_canal_arquivo}_{rotulo_faixa}_{f_min:.0f}-{f_max:.0f}hz.png"
                plt.savefig(pasta_figuras_psd / nome_figura_psd, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"      🖼️ Figura salva: Figuras/{sensor}/PSD/{nome_figura_psd}")

        sensores_ok += 1
        if houve_erro_no_sensor:
            sensores_com_erro += 1

    caminho_log = registrar_log(raiz_path, "09_rms_psd", {
        "data_dir": raiz_path.resolve(),
        "metodo": METODO_DESCRICAO,
        "metadados_condicoes": str(caminho_metadados_condicoes.resolve()) if caminho_metadados_condicoes else None,
        "f1_hz": args.f1,
        "f2_hz": args.f2,
        "freq_resolucao_hz_grid_psd": args.freq_resolucao,
        "cmap": args.cmap,
        "quick": args.quick,
        "from_condicao": args.from_condicao,
        "tipos_de_sensor_processados": sensores_ok,
        "tipos_de_sensor_com_aviso": sensores_com_erro,
    }, pastas_alteradas=pastas_alteradas)

    print("\n" + "=" * 65)
    print(f"✅ Etapa 09 (RMS/PSD) Concluída!")
    print(f"   Tipos de sensor processados: {sensores_ok} | Tipos de sensor com algum aviso: {sensores_com_erro}")
    print(f"💾 Figuras salvas em: {pasta_figuras_raiz.resolve()}")
    print(f"📝 Log de parâmetros: {caminho_log.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
