from __future__ import annotations
import argparse
import importlib.util
import csv
import re
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
filtrar_desde_condicao = _pipeline_io.filtrar_desde_condicao
carregar_grupo = _pipeline_io.carregar_grupo
salvar_grupo = _pipeline_io.salvar_grupo
registrar_log = _pipeline_io.registrar_log

_estilo = _carregar_modulo("estilo_grafico", "estilo_grafico.py")
My_axis = _estilo.My_axis

# Mesmas cores vivas das etapas 04/05
COR_PICOS = {"low": "#2979FF", "mid": "#00C853", "high": "#FF1744"}

# Texto explicativo por modo de --escala, registrado no log (mesmo espírito
# do METODO_ESPECTRAL da etapa 03). Aqui cada "linha" do mapa é um
# sensor/canal, não uma condição de ensaio.
METODO_ESCALA_DESCRICAO = {
    "db-global": (
        "Amplitude (dB) = 20*log10(|A| / A_max_global + eps), onde A_max_global "
        "é a maior amplitude absoluta entre TODOS os sensores/canais daquela "
        "faixa (np.max sobre a matriz inteira). Só o sensor com o pico global "
        "bate 0 dB; os demais ficam abaixo, preservando a intensidade "
        "ABSOLUTA relativa entre sensores. Piso do gráfico: --db-min."
    ),
    "abs-global": (
        "Sem normalizar: dados plotados na unidade original (saída da etapa "
        "03/FFT), com uma única referência de cor pra toda a figura: "
        "vmax = maior amplitude absoluta entre TODOS os sensores/canais "
        "daquela faixa (np.max)."
    ),
    "abs-condicao": (
        "Cada sensor/canal (linha do mapa) dividido pelo próprio pico "
        "absoluto: A / max(|A|) por linha (np.max(axis=1)). Mesmo cálculo do "
        "'pico-canal', lido aqui como 'escala absoluta com referência própria "
        "por sensor'."
    ),
    "pico-canal": (
        "Normalização relativa: cada sensor/canal dividido pelo próprio pico "
        "absoluto (np.max(axis=1) por linha), resultando em valores de 0 a 1 "
        "(sem unidade física)."
    ),
    "rms-canal": (
        "Cada sensor/canal dividido pelo próprio RMS: A / sqrt(mean(A**2)) "
        "por linha (np.sqrt, np.mean(axis=1)), realçando o sinal acima do "
        "nível médio de energia daquele sensor."
    ),
    "db": (
        "Amplitude (dB) = 20*log10(|A| / max(|A|) + eps), com max(|A|) "
        "calculado POR LINHA (por sensor, não global) — todos os sensores "
        "batem 0 dB no próprio pico; não preserva intensidade absoluta entre "
        "sensores (ver 'db-global' para isso). Piso do gráfico: --db-min."
    ),
}


def _bordas_a_partir_de_centros(centros) -> np.ndarray:
    """
    Converte uma lista de posições centrais (ex.: posição física de cada
    canal, ou frequência de cada bin) em bordas (len(centros)+1), pra usar
    com pcolormesh. Ao contrário do imshow (que assume espaçamento uniforme
    entre linhas/colunas), isso respeita o espaçamento REAL entre pontos —
    essencial no eixo Y quando ele representa posição física (metros) ou
    f_vfd (Hz), que raramente são igualmente espaçados entre condições/canais.
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


def _chave_ordenacao_canal(nome_canal: str):
    """Ordena canais numericamente quando possível (Channel 0, 1, 2... 10),
    em vez de alfabeticamente (que colocaria 'Channel 10' antes de 'Channel 2')."""
    numeros = re.findall(r"\d+", str(nome_canal))
    return (int(numeros[0]) if numeros else 10**9, str(nome_canal))


# ==============================================================================
# 🛠️ 1. METADADOS OPCIONAIS POR CANAL/SENSOR (CSV)
# ==============================================================================
def ler_metadados_canais(caminho_csv: Path) -> dict:
    """
    Lê o CSV opcional de metadados por canal (posição física ao longo da
    tubulação, ver README): sensor,canal,posicao_m,rotulo.

    Retorna: dict (sensor, canal) -> {"posicao_m": float|None, "rotulo": str|None}
    """
    metadados = {}
    with open(caminho_csv, newline="", encoding="utf-8-sig") as f:
        leitor = csv.DictReader(f)
        colunas_esperadas = {"sensor", "canal", "posicao_m", "rotulo"}
        if not colunas_esperadas.issubset(set(leitor.fieldnames or [])):
            faltando = colunas_esperadas - set(leitor.fieldnames or [])
            raise ValueError(f"CSV de metadados sem as colunas esperadas: {faltando}")

        for linha in leitor:
            sensor = (linha.get("sensor") or "").strip().upper()
            canal = (linha.get("canal") or "").strip()
            if not sensor or not canal:
                continue
            posicao_raw = (linha.get("posicao_m") or "").strip()
            rotulo = (linha.get("rotulo") or "").strip() or None
            metadados[(sensor, canal)] = {
                "posicao_m": float(posicao_raw) if posicao_raw else None,
                "rotulo": rotulo,
            }
    return metadados


# ==============================================================================
# 🚀 2. EXECUÇÃO PRINCIPAL
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--quick", action="store_true",
                         help="Processa apenas o primeiro grupo (sensor, condição) encontrado, para teste rápido.")
    parser.add_argument("--from", dest="from_condicao", type=str, default=None,
                         help="Retoma a etapa a partir desta condição, inclusive (ex.: --from T3 processa "
                              "T3, T4, T5... e pula T1/T2). Extrai o número do padrão T<N> no nome da "
                              "condição; nomes fora desse padrão nunca são descartados.")
    parser.add_argument("--metadados-canais", type=str, default=None,
                         help="Caminho de um CSV opcional (sensor,canal,posicao_m,rotulo) — ver formato "
                              "no README. Se omitido, o eixo Y usa o nome cru do canal (Channel 0, 1, 2...).")
    parser.add_argument("--escala", choices=["db-global", "abs-global", "abs-condicao", "pico-canal", "rms-canal", "db"],
                         default="db-global",
                         help="Escala de cor do mapa (padrão: db-global). Aqui 'condição/canal' se refere "
                              "à referência por CANAL (linha do mapa), não por condição de ensaio.")
    parser.add_argument("--db-min", type=float, default=-40.0,
                         help="Piso (dB) usado quando --escala db (padrão: -40.0).")
    parser.add_argument("--cmap", type=str, default="viridis",
                         help="Colormap do matplotlib (padrão: viridis).")
    parser.add_argument("--freq-max", type=float, default=None,
                         help="Frequência máxima (Hz) da faixa 'high'. Padrão: automático (menor Nyquist "
                              "entre os canais do grupo).")
    parser.add_argument("--freq-resolucao", type=float, default=0.5,
                         help="Não usado para interpolar aqui (todos os canais de um grupo já compartilham "
                              "o mesmo grid de frequência, saída da etapa 03) — mantido só por simetria com a etapa 05.")
    parser.add_argument("--f1", type=float, default=15.0,
                         help="Limite low/mid (Hz), mesma convenção das etapas 03/04/05 (padrão: 15.0).")
    parser.add_argument("--f2", type=float, default=400.0,
                         help="Limite mid/high (Hz), mesma convenção das etapas 03/04/05 (padrão: 400.0).")
    parser.add_argument("--sem-picos", action="store_true",
                         help="Não sobrepõe os picos (Etapas/Picos) como marcadores no mapa.")
    args = parser.parse_args()

    raiz_path = Path(args.data_dir)

    input_dir = raiz_path / "DadosTratados" / "Etapas" / "FFT"
    grupos = listar_grupos(input_dir)  # [(sensor, condicao, caminho_parquet), ...]

    if not grupos:
        print(f"❌ Nenhum grupo encontrado em: {input_dir.resolve()}")
        print(" Certifique-se de executar a etapa 03_fft antes desta.")
        exit(1)

    if args.quick:
        grupos = grupos[:1]
        print("⚡ Modo rápido (--quick): processando apenas o primeiro grupo (sensor, condição).\n")

    grupos = filtrar_desde_condicao(grupos, args.from_condicao, indice_condicao=1)
    if args.from_condicao:
        print(f"⏩ Retomando a partir de {args.from_condicao} (--from): {len(grupos)} grupo(s) a processar.\n")
        if not grupos:
            print(f"❌ Nenhum grupo com condição >= {args.from_condicao} encontrado. Nada a fazer.")
            exit(1)

    metadados_canais = {}
    caminho_metadados_canais = None
    if args.metadados_canais:
        caminho_metadados_canais = Path(args.metadados_canais)
    else:
        # Busca automática: se não foi passado --metadados-canais, procura um
        # "canais.csv" direto na pasta base (--data_dir).
        candidato = raiz_path / "canais.csv"
        if candidato.exists():
            caminho_metadados_canais = candidato
            print(f"📋 Encontrado canais.csv na pasta base, usando automaticamente: {candidato.resolve()}")

    if caminho_metadados_canais:
        try:
            metadados_canais = ler_metadados_canais(caminho_metadados_canais)
            print(f"📋 Metadados de canal carregados: {caminho_metadados_canais.resolve()} ({len(metadados_canais)} canal(is))")
        except Exception as e:
            print(f"⚠️ Não foi possível ler {caminho_metadados_canais} ({e}). Seguindo sem metadados.")
            metadados_canais = {}

    pasta_figuras_raiz = raiz_path / "DadosTratados" / "Figuras"
    output_dir = raiz_path / "DadosTratados" / "Etapas" / "MapaEspacial"
    pasta_picos_raiz = raiz_path / "DadosTratados" / "Etapas" / "Picos"

    pastas_alteradas = {output_dir}
    grupos_ok, grupos_com_erro = 0, 0

    print(f"⚙️ Gerando mapa espacial (canal x frequência) para {len(grupos)} grupo(s) sensor/condição...\n")

    for sensor, condicao, caminho_parquet in grupos:
        print(f"\n📖 Grupo: [{sensor}] | [{condicao}]  ←  {caminho_parquet.name}")

        try:
            df_fft = carregar_grupo(caminho_parquet)
        except Exception as e:
            print(f"   ⚠️ Erro ao carregar grupo: {e}. Pulando.")
            grupos_com_erro += 1
            continue

        if "freq_hz" not in df_fft.columns:
            print(f"   ⚠️ Coluna 'freq_hz' não encontrada (rode a etapa 03 primeiro). Pulando.")
            grupos_com_erro += 1
            continue

        freqs = df_fft["freq_hz"].to_numpy()
        colunas_canais = [c for c in df_fft.columns if str(c).endswith("_amplitude")]
        canais = [str(c)[:-len("_amplitude")] for c in colunas_canais]

        if len(canais) < 2:
            print(f"   ⚠️ Só {len(canais)} canal(is) neste grupo; mapa espacial precisa de pelo menos 2. Pulando.")
            grupos_com_erro += 1
            continue

        # --- decide o eixo Y: posição física (contínuo) só se TODOS os canais
        # deste sensor tiverem posicao_m preenchida; senão, categórico (tudo
        # ou nada, mesma lógica da etapa 05 com f_vfd_hz) ---
        usar_eixo_continuo = False
        canais_sem_posicao = []
        if metadados_canais:
            for c in canais:
                if metadados_canais.get((str(sensor).upper(), c), {}).get("posicao_m") is None:
                    canais_sem_posicao.append(c)
            usar_eixo_continuo = len(canais_sem_posicao) == 0
            if metadados_canais and not usar_eixo_continuo:
                print(f"   ℹ️ Faltam posicao_m para {canais_sem_posicao}; usando eixo categórico "
                      f"(sem posição física) para [{sensor}]/[{condicao}].")

        if usar_eixo_continuo:
            canais_ordenados = sorted(canais, key=lambda c: metadados_canais[(str(sensor).upper(), c)]["posicao_m"])
            valores_reais = [metadados_canais[(str(sensor).upper(), c)]["posicao_m"] for c in canais_ordenados]
            rotulos_y = [metadados_canais[(str(sensor).upper(), c)]["rotulo"] or c for c in canais_ordenados]
            rotulo_eixo_y = "Position along pipeline (m)"
        else:
            canais_ordenados = sorted(canais, key=_chave_ordenacao_canal)
            valores_reais = None
            rotulos_y = [
                (metadados_canais.get((str(sensor).upper(), c), {}).get("rotulo") or c)
                for c in canais_ordenados
            ]
            rotulo_eixo_y = "Sensor / Channel"

        # Eixo Y SEMPRE com bandas de altura igual (0,1,2...), mesmo no modo
        # contínuo — posicao_m é um conjunto de pontos discretos (onde os
        # sensores foram instalados), não uma variável amostrada
        # continuamente; usar a posição real distorce a altura das bandas e
        # empurra os canais extremos pra beira do gráfico (ver README).
        valores_y = list(range(len(canais_ordenados)))

        # --- matriz (canal x freq) — todos os canais de um grupo já compartilham
        # o MESMO grid de frequência (saída da etapa 03), então não precisa
        # interpolar nada aqui, só empilhar as colunas na ordem certa ---
        matrix_raw = np.array([df_fft[f"{c}_amplitude"].to_numpy() for c in canais_ordenados])

        # --- salva a matriz consolidada (RAW, banda inteira) para reuso futuro ---
        df_mapa = pd.DataFrame(
            matrix_raw, index=pd.Index(canais_ordenados, name="canal"), columns=freqs,
        ).reset_index().melt(id_vars="canal", var_name="freq_hz", value_name="amplitude")
        df_mapa["y_valor"] = df_mapa["canal"].map(dict(zip(canais_ordenados, valores_y)))
        if usar_eixo_continuo:
            df_mapa["posicao_m"] = df_mapa["canal"].map(dict(zip(canais_ordenados, valores_reais)))
        salvar_grupo(df_mapa, sensor, condicao, output_dir)
        print(f"      💾 Matriz salva: Etapas/MapaEspacial/{sensor}/{condicao}.parquet")

        pasta_figuras = pasta_figuras_raiz / str(sensor) / str(condicao) / "MapaEspacial"
        pasta_figuras.mkdir(parents=True, exist_ok=True)
        pastas_alteradas.add(pasta_figuras)

        # carrega picos desse grupo uma vez só (reaproveita nas 3 faixas)
        df_picos_grupo = None
        if not args.sem_picos:
            caminho_picos = pasta_picos_raiz / str(sensor) / f"{condicao}.parquet"
            if caminho_picos.exists():
                try:
                    df_picos_grupo = pd.read_parquet(caminho_picos)
                except Exception:
                    df_picos_grupo = None

        nyquist = freqs.max() if freqs.size else 0.0
        faixas = [
            (0.0, args.f1, "low"),
            (args.f1, args.f2, "mid"),
            (args.f2, args.freq_max if args.freq_max is not None else nyquist, "high"),
        ]

        for f_min, f_max, rotulo_faixa in faixas:
            mascara_freq = (freqs >= f_min) & (freqs <= f_max)
            if not mascara_freq.any():
                print(f"      ⚠️ Faixa {rotulo_faixa}: sem dado nessa faixa, pulando figura.")
                continue

            freqs_faixa = freqs[mascara_freq]

            # reaproveita converter_escala do heatmap por condição (mesma lógica,
            # só que aqui cada "linha" é um canal/sensor em vez de uma condição)
            eps = 1e-12
            sub = matrix_raw[:, mascara_freq]
            abs_sub = np.abs(sub)
            if args.escala in ("pico-canal", "abs-condicao"):
                picos = np.max(abs_sub, axis=1, keepdims=True) + eps
                matrix_plot = sub / picos
                v_min, v_max = 0.0, 1.0
                label_cbar = ("Amplitude - normalized to per-sensor peak" if args.escala == "pico-canal"
                              else "Amplitude - absolute scale per sensor (own peak = 1.0)")
            elif args.escala == "rms-canal":
                rms = np.sqrt(np.mean(sub ** 2, axis=1, keepdims=True)) + eps
                matrix_plot = sub / rms
                v_min, v_max = 0.0, float(np.max(matrix_plot)) if matrix_plot.size else 1.0
                label_cbar = "Amplitude - normalized to per-sensor RMS"
            elif args.escala == "db":
                picos = np.max(abs_sub, axis=1, keepdims=True) + eps
                matrix_plot = 20 * np.log10((abs_sub / picos) + eps)
                v_min, v_max = args.db_min, 0.0
                label_cbar = "Amplitude (dB, relative to per-sensor peak)"
            elif args.escala == "db-global":
                # Amplitude (dB) = 20*log10(A / A_max_global) — só o sensor
                # que contém o pico global bate 0 dB; os outros ficam abaixo,
                # preservando a comparação de intensidade ABSOLUTA entre sensores.
                v_max_global = float(np.max(abs_sub)) if abs_sub.size else eps
                matrix_plot = 20 * np.log10((abs_sub / v_max_global) + eps)
                v_min, v_max = args.db_min, 0.0
                label_cbar = "Amplitude (dB, relative to global peak across all sensors)"
            elif args.escala == "abs-global":
                matrix_plot = sub
                v_min = 0.0
                v_max = float(np.max(abs_sub)) if abs_sub.size else eps
                label_cbar = "Amplitude - absolute, shared scale across all sensors"
            else:
                matrix_plot = sub
                v_min = 0.0
                v_max = float(np.max(abs_sub)) if abs_sub.size else eps
                label_cbar = "Amplitude - absolute, shared scale across all sensors"

            fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
            bordas_x = _bordas_a_partir_de_centros(freqs_faixa)
            bordas_y = _bordas_a_partir_de_centros(valores_y)
            im = ax.pcolormesh(
                bordas_x, bordas_y, matrix_plot, cmap=args.cmap, vmin=v_min, vmax=v_max,
                shading="flat",
            )
            cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.14, aspect=50)
            cbar.set_label(label_cbar, fontsize=11, labelpad=6)

            ax.set_yticks(valores_y)
            ax.set_yticklabels(rotulos_y)

            # --- divisórias entre bandas nos limites (major tick) + linha
            # sutil no centro de cada banda, lembrando que é um ponto
            # discreto (a banda inteira é só espaçamento visual) ---
            for borda in bordas_y:
                ax.axhline(borda, color="white", alpha=0.5, linewidth=1.0, zorder=3)
            for centro in valores_y:
                ax.axhline(centro, color="white", alpha=0.22, linewidth=0.6, zorder=3)

            # --- overlay dos picos dessa faixa (Etapas/Picos), se disponível ---
            if df_picos_grupo is not None:
                freqs_overlay, y_overlay = [], []
                for canal, y_valor in zip(canais_ordenados, valores_y):
                    mascara = (df_picos_grupo["canal"] == canal) & (df_picos_grupo["escopo"] == rotulo_faixa)
                    for f_p in df_picos_grupo.loc[mascara, "freq_hz"]:
                        freqs_overlay.append(f_p)
                        y_overlay.append(y_valor)
                if freqs_overlay:
                    ax.scatter(freqs_overlay, y_overlay, facecolors="none", edgecolors=COR_PICOS[rotulo_faixa],
                               marker="o", s=40, linewidths=1.2, alpha=0.9, zorder=5,
                               label=f"Peaks ({rotulo_faixa})")

            My_axis(
                ax, font=13,
                xlim=[freqs_faixa[0], freqs_faixa[-1]],
                ylim=[bordas_y[0], bordas_y[-1]],
                setaxis=[f"Spatial Map ({rotulo_faixa}) - {sensor} | {condicao} | {f_min:.0f}-{f_max:.0f} Hz\n",
                         "Frequency (Hz)", rotulo_eixo_y],
                legbox=[0.98, 0.98, 1, 10],
            )

            nome_figura = f"mapa_espacial_{rotulo_faixa}_{f_min:.0f}-{f_max:.0f}hz.png"
            caminho_figura = pasta_figuras / nome_figura
            plt.tight_layout()
            plt.savefig(caminho_figura, dpi=150)
            plt.close(fig)

            print(f"      🖼️ Figura salva: Figuras/{sensor}/{condicao}/MapaEspacial/{nome_figura}")

        grupos_ok += 1

    caminho_log = registrar_log(raiz_path, "06_mapa_espacial", {
        "data_dir": raiz_path.resolve(),
        "metadados_canais": str(caminho_metadados_canais.resolve()) if caminho_metadados_canais else None,
        "escala": args.escala,
        "metodo_escala": METODO_ESCALA_DESCRICAO.get(args.escala, ""),
        "db_min": args.db_min if args.escala in ("db", "db-global") else None,
        "cmap": args.cmap,
        "freq_max": args.freq_max,
        "f1_hz": args.f1,
        "f2_hz": args.f2,
        "sobrepor_picos": not args.sem_picos,
        "quick": args.quick,
        "from_condicao": args.from_condicao,
        "grupos_processados": grupos_ok,
        "grupos_com_aviso": grupos_com_erro,
    }, pastas_alteradas=pastas_alteradas)

    print("\n" + "=" * 65)
    print(f"✅ Etapa 06 (Mapa Espacial) Concluída!")
    print(f"   Grupos processados: {grupos_ok} | Grupos com algum aviso: {grupos_com_erro}")
    print(f"💾 Matrizes salvas em: {output_dir.resolve()}")
    print(f"📝 Log de parâmetros: {caminho_log.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
