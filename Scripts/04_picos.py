from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


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


# Cores vivas por faixa, escolhidas pra ficar fora do esquema de cor do
# heatmap (viridis) e serem consistentes entre a etapa 04 e o overlay da 05.
COR_PICOS = {"low": "#2979FF", "mid": "#00C853", "high": "#FF1744"}


# ==============================================================================
# 🛠️ 1. MÉTODO DE IDENTIFICAÇÃO DOS PICOS
# ==============================================================================
METODO_DESCRICAO = (
    "scipy.signal.find_peaks aplicado SEPARADAMENTE em cada faixa de frequencia "
    "(low/mid/high, mesmos limites f1/f2 da etapa 03) sobre o espectro de "
    "amplitude (saida da etapa 03), com distancia minima entre picos vizinhos "
    "(em Hz, ajustavel por tipo de sensor) para evitar pegar varios pontos do "
    "mesmo lobulo espectral; dentro de cada faixa, mantidos os N de maior "
    "amplitude por canal (N e a distancia minima sao configuraveis via CLI). "
    "Buscar por faixa (em vez de no espectro inteiro) evita que uma faixa com "
    "energia muito mais forte 'esconda' os picos de outra faixa mais fraca."
)


def sanitizar_nome(nome: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(nome)).strip("_") or "canal"


def identificar_picos(freqs: np.ndarray, amplitude: np.ndarray, n_picos: int,
                       min_distancia_hz: float):
    """
    Identifica até `n_picos` picos de maior amplitude do espectro.

    Método: scipy.signal.find_peaks, com distância mínima entre picos
    convertida de Hz para nº de amostras a partir da resolução do
    espectro (df = freqs[1]-freqs[0]). Isso evita identificar múltiplos
    pontos vizinhos do mesmo lóbulo como picos "diferentes" — o ajuste
    fino dessa distância é o que muda de sensor para sensor.

    Entre os picos encontrados por find_peaks, ficam os N de maior
    amplitude (não necessariamente os N mais à esquerda no espectro).
    Se find_peaks não encontrar nenhum pico "de verdade" (ex.: espectro
    curto ou praticamente plano), cai para os N pontos de maior
    amplitude do espectro inteiro, como fallback.

    Retorna (freqs_pico, amplitudes_pico), ordenados por frequência
    crescente.
    """
    if freqs.size < 3 or n_picos <= 0:
        return np.array([]), np.array([])

    df_hz = freqs[1] - freqs[0]
    distancia_amostras = max(1, int(round(min_distancia_hz / df_hz))) if df_hz > 0 else 1

    indices, _ = find_peaks(amplitude, distance=distancia_amostras)

    if indices.size == 0:
        indices = np.argsort(amplitude)[::-1][:n_picos]
    else:
        ordem = np.argsort(amplitude[indices])[::-1]
        indices = indices[ordem[:n_picos]]

    indices = np.sort(indices)
    return freqs[indices], amplitude[indices]


# ==============================================================================
# 🚀 2. EXECUÇÃO PRINCIPAL
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--quick", action="store_true",
                         help="Processa (e plota) apenas o primeiro grupo sensor/condicao, para teste rápido.")
    parser.add_argument("--from", dest="from_condicao", type=str, default=None,
                         help="Retoma a etapa a partir desta condição, inclusive (ex.: --from T3 processa "
                              "T3, T4, T5... e pula T1/T2). Extrai o número do padrão T<N> no nome da "
                              "condição; nomes fora desse padrão nunca são descartados.")
    parser.add_argument("--n-picos", type=int, default=5,
                         help="Número de picos a identificar POR FAIXA (low/mid/high) e por canal "
                              "(padrão: 5). A busca global (espectro inteiro, sem plot) usa sempre "
                              "o dobro desse valor.")
    parser.add_argument("--min-dist-acl", type=float, default=2.0,
                         help="Distância mínima (Hz) entre picos vizinhos para sensores ACL (padrão: 2.0).")
    parser.add_argument("--min-dist-pzt", type=float, default=5.0,
                         help="Distância mínima (Hz) entre picos vizinhos para sensores PZT (padrão: 5.0).")
    parser.add_argument("--min-dist", type=float, default=2.0,
                         help="Distância mínima (Hz) de fallback para sensores fora do mapeamento ACL/PZT (padrão: 2.0).")
    parser.add_argument("--f1", type=float, default=15.0,
                         help="Limite entre a faixa LOW e MID, em Hz (padrão: 15.0). Define as faixas "
                              "em que os picos são buscados, mesma convenção da etapa 03.")
    parser.add_argument("--f2", type=float, default=400.0,
                         help="Limite entre a faixa MID e HIGH, em Hz (padrão: 400.0).")
    args = parser.parse_args()

    min_dist_por_sensor = {"ACL": args.min_dist_acl, "PZT": args.min_dist_pzt}

    def obter_min_dist(sensor: str) -> float:
        return min_dist_por_sensor.get(str(sensor).upper(), args.min_dist)

    raiz_path = Path(args.data_dir)

    input_dir = raiz_path / "DadosTratados" / "Etapas" / "FFT"
    grupos = listar_grupos(input_dir)

    if not grupos:
        print(f"❌ Nenhum grupo encontrado em: {input_dir.resolve()}")
        print(" Certifique-se de executar a etapa 03_fft antes desta.")
        exit(1)

    if args.quick:
        grupos = grupos[:1]
        print("⚡ Modo rápido (--quick): processando apenas o primeiro grupo.\n")

    grupos = filtrar_desde_condicao(grupos, args.from_condicao, indice_condicao=1)
    if args.from_condicao:
        print(f"⏩ Retomando a partir de {args.from_condicao} (--from): {len(grupos)} grupo(s) a processar.\n")
        if not grupos:
            print(f"❌ Nenhum grupo com condição >= {args.from_condicao} encontrado. Nada a fazer.")
            exit(1)

    pasta_figuras_raiz = raiz_path / "DadosTratados" / "Figuras"
    output_dir = raiz_path / "DadosTratados" / "Etapas" / "Picos"

    grupos_ok, grupos_com_erro = 0, 0
    pastas_alteradas = {output_dir}

    print(f"⚙️ Identificando picos (N={args.n_picos}/faixa, {args.n_picos * 2} global) em "
          f"{len(grupos)} grupo(s) sensor/condição...")
    print(f"   Método: {METODO_DESCRICAO}\n")

    for sensor, condicao, caminho_parquet in grupos:
        min_dist_hz = obter_min_dist(sensor)

        print(f"\n📖 Grupo: [{sensor}] | [{condicao}]  ←  {caminho_parquet.name}  "
              f"(distância mínima entre picos={min_dist_hz:.2f} Hz)")

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
        colunas_amplitude = [c for c in df_fft.columns if str(c).endswith("_amplitude")]

        if not colunas_amplitude:
            print(f"   ⚠️ Nenhuma coluna '_amplitude' encontrada neste grupo. Pulando.")
            grupos_com_erro += 1
            continue

        nyquist = freqs.max() if freqs.size else 0.0
        faixas = [
            (0.0, args.f1, "low"),
            (args.f1, args.f2, "mid"),
            (args.f2, nyquist, "high"),
        ]

        pasta_figuras = pasta_figuras_raiz / str(sensor) / str(condicao) / "Picos"
        pasta_figuras.mkdir(parents=True, exist_ok=True)
        pastas_alteradas.add(pasta_figuras)

        linhas_picos = []
        houve_erro_no_grupo = False

        for col_canal in colunas_amplitude:
            nome_canal_legivel = str(col_canal)[:-len("_amplitude")]
            nome_canal_arquivo = sanitizar_nome(nome_canal_legivel)
            amplitude = df_fft[col_canal].to_numpy()

            if amplitude.size < 3 or not np.all(np.isfinite(amplitude)):
                print(f"      ⚠️ Canal {nome_canal_legivel}: espectro vazio/curto/inválido, pulando.")
                houve_erro_no_grupo = True
                continue

            algum_pico_no_canal = False

            # --- 1) Busca POR FAIXA (low/mid/high), independente entre si ---
            # Cada faixa é buscada isoladamente para que uma faixa com energia
            # muito mais forte não "esconda" os picos (menores em escala
            # absoluta, mas ainda relevantes) de outra faixa. Gera 1 figura
            # por faixa, com os picos daquela faixa marcados em vermelho.
            for f_min, f_max, rotulo in faixas:
                mascara = (freqs >= f_min) & (freqs <= f_max)
                if not mascara.any():
                    continue

                freqs_banda = freqs[mascara]
                amp_banda = amplitude[mascara]
                freqs_pico, amp_pico = identificar_picos(freqs_banda, amp_banda, args.n_picos, min_dist_hz)

                if freqs_pico.size == 0:
                    print(f"      ⚠️ Canal {nome_canal_legivel} | faixa {rotulo}: nenhum pico identificado.")
                else:
                    algum_pico_no_canal = True
                    for ordem, (f_p, a_p) in enumerate(zip(freqs_pico, amp_pico), start=1):
                        linhas_picos.append({
                            "canal": nome_canal_legivel,
                            "escopo": rotulo,
                            "ordem_pico": ordem,
                            "freq_hz": float(f_p),
                            "amplitude": float(a_p),
                        })

                fig, ax1 = plt.subplots(figsize=(10, 5))
                cor_linha = 'green' if str(sensor).upper() == 'ACL' else 'black'
                ax1.plot(freqs_banda, amp_banda, c=cor_linha, alpha=0.9, linewidth=1.0)

                if freqs_pico.size > 0:
                    ax1.scatter(
                        freqs_pico, amp_pico,
                        facecolors='none', edgecolors=COR_PICOS[rotulo], marker='o', s=90,
                        linewidths=1.8, zorder=5, label=f"Peaks (N={args.n_picos})"
                    )

                y_max = amp_banda.max() * 1.2 if amp_banda.size else 1e-9
                y_max = max(y_max, 1e-9)

                My_axis(
                    ax1,
                    font=12,
                    xlim=[f_min, f_max],
                    ylim=[0, y_max],
                    setaxis=[
                        f"FFT + Peaks ({rotulo}) - {sensor} | {condicao} | {nome_canal_legivel} | {f_min:.0f}-{f_max:.0f} Hz\n",
                        "Frequency (Hz)",
                        "Amplitude"
                    ]
                )

                nome_figura = f"peaks_{nome_canal_arquivo}_{rotulo}_{f_min:.0f}-{f_max:.0f}hz.png"
                caminho_figura = pasta_figuras / nome_figura

                plt.tight_layout()
                plt.savefig(caminho_figura, dpi=150)
                plt.close(fig)

                print(f"      🖼️ Figura salva: Figuras/{sensor}/{condicao}/Picos/{nome_figura}")

            # --- 2) Busca GLOBAL (0 até Nyquist), só registro, sem figura ---
            # Sempre o dobro do N pedido por faixa: como picos por faixa e
            # picos globais respondem perguntas diferentes (destaque local vs.
            # destaque no espectro inteiro), mantemos os dois registrados.
            n_picos_global = args.n_picos * 2
            freqs_pico_global, amp_pico_global = identificar_picos(
                freqs, amplitude, n_picos_global, min_dist_hz
            )

            if freqs_pico_global.size == 0:
                print(f"      ⚠️ Canal {nome_canal_legivel} | global: nenhum pico identificado.")
            else:
                algum_pico_no_canal = True
                for ordem, (f_p, a_p) in enumerate(zip(freqs_pico_global, amp_pico_global), start=1):
                    linhas_picos.append({
                        "canal": nome_canal_legivel,
                        "escopo": "global",
                        "ordem_pico": ordem,
                        "freq_hz": float(f_p),
                        "amplitude": float(a_p),
                    })

            if not algum_pico_no_canal:
                houve_erro_no_grupo = True

        if linhas_picos:
            df_picos = pd.DataFrame(linhas_picos)
            salvar_grupo(df_picos, sensor, condicao, output_dir)
            print(f"      💾 {len(linhas_picos)} peak(s) registrado(s) em "
                  f"Etapas/Picos/{sensor}/{condicao}.parquet")

        grupos_ok += 1
        if houve_erro_no_grupo:
            grupos_com_erro += 1

    caminho_log = registrar_log(raiz_path, "04_picos", {
        "data_dir": raiz_path.resolve(),
        "metodo": METODO_DESCRICAO,
        "n_picos_por_faixa": args.n_picos,
        "n_picos_global": args.n_picos * 2,
        "min_dist_acl_hz": args.min_dist_acl,
        "min_dist_pzt_hz": args.min_dist_pzt,
        "min_dist_fallback_hz": args.min_dist,
        "faixa_low_hz": f"0-{args.f1:.0f}",
        "faixa_mid_hz": f"{args.f1:.0f}-{args.f2:.0f}",
        "faixa_high_hz": f"{args.f2:.0f}-Nyquist",
        "quick": args.quick,
        "from_condicao": args.from_condicao,
        "grupos_processados": grupos_ok,
        "grupos_com_aviso": grupos_com_erro,
    }, pastas_alteradas=pastas_alteradas)

    print("\n" + "=" * 65)
    print(f"✅ Etapa 04 (Picos) Concluída!")
    print(f"   Grupos processados: {grupos_ok} | Grupos com algum aviso: {grupos_com_erro}")
    print(f"💾 Picos (freq/amplitude/escopo por canal) salvos em: {output_dir.resolve()}")
    print(f"📝 Log de parâmetros: {caminho_log.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
