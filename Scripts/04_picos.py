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
carregar_grupo = _pipeline_io.carregar_grupo
salvar_grupo = _pipeline_io.salvar_grupo
registrar_log = _pipeline_io.registrar_log

_estilo = _carregar_modulo("estilo_grafico", "estilo_grafico.py")
My_axis = _estilo.My_axis


# ==============================================================================
# 🛠️ 1. MÉTODO DE IDENTIFICAÇÃO DOS PICOS
# ==============================================================================
METODO_DESCRICAO = (
    "scipy.signal.find_peaks sobre o espectro de amplitude (saida da etapa 03), "
    "com distancia minima entre picos vizinhos (em Hz, ajustavel por tipo de "
    "sensor) para evitar pegar varios pontos do mesmo lobulo espectral; entre "
    "os picos validos, mantidos os N de maior amplitude por canal (N e a "
    "distancia minima sao configuraveis via CLI)."
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


def classificar_faixa(freq: float, f1: float, f2: float) -> str:
    if freq <= f1:
        return "baixa"
    if freq <= f2:
        return "media"
    return "alta"


# ==============================================================================
# 🚀 2. EXECUÇÃO PRINCIPAL
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--quick", action="store_true",
                         help="Processa (e plota) apenas o primeiro grupo sensor/condicao, para teste rápido.")
    parser.add_argument("--n-picos", type=int, default=5,
                         help="Número de picos a identificar por canal (padrão: 5).")
    parser.add_argument("--min-dist-acl", type=float, default=2.0,
                         help="Distância mínima (Hz) entre picos vizinhos para sensores ACL (padrão: 2.0).")
    parser.add_argument("--min-dist-pzt", type=float, default=5.0,
                         help="Distância mínima (Hz) entre picos vizinhos para sensores PZT (padrão: 5.0).")
    parser.add_argument("--min-dist", type=float, default=2.0,
                         help="Distância mínima (Hz) de fallback para sensores fora do mapeamento ACL/PZT (padrão: 2.0).")
    parser.add_argument("--f1", type=float, default=15.0,
                         help="Limite entre a faixa BAIXA e MÉDIA, em Hz (padrão: 15.0). Só usado para "
                              "classificar/plotar os picos por faixa, mesma convenção da etapa 03.")
    parser.add_argument("--f2", type=float, default=400.0,
                         help="Limite entre a faixa MÉDIA e ALTA, em Hz (padrão: 400.0).")
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

    pasta_figuras_raiz = raiz_path / "DadosTratados" / "Figuras"
    output_dir = raiz_path / "DadosTratados" / "Etapas" / "Picos"

    grupos_ok, grupos_com_erro = 0, 0
    pastas_alteradas = {output_dir}

    print(f"⚙️ Identificando picos (N={args.n_picos}) em {len(grupos)} grupo(s) sensor/condição...")
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
            (0.0, args.f1, "baixa"),
            (args.f1, args.f2, "media"),
            (args.f2, nyquist, "alta"),
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

            freqs_pico, amp_pico = identificar_picos(freqs, amplitude, args.n_picos, min_dist_hz)

            if freqs_pico.size == 0:
                print(f"      ⚠️ Canal {nome_canal_legivel}: nenhum pico identificado.")
                houve_erro_no_grupo = True
                continue

            for ordem, (f_p, a_p) in enumerate(zip(freqs_pico, amp_pico), start=1):
                linhas_picos.append({
                    "canal": nome_canal_legivel,
                    "ordem_pico": ordem,
                    "freq_hz": float(f_p),
                    "amplitude": float(a_p),
                    "faixa": classificar_faixa(f_p, args.f1, args.f2),
                })

            # Mesmas figuras (por faixa) da etapa 03, com marcador vermelho em cada pico.
            for f_min, f_max, rotulo in faixas:
                mascara = (freqs >= f_min) & (freqs <= f_max)
                if not mascara.any():
                    continue

                fig, ax1 = plt.subplots(figsize=(10, 5))
                cor_linha = 'green' if str(sensor).upper() == 'ACL' else 'black'
                ax1.plot(freqs[mascara], amplitude[mascara], c=cor_linha, alpha=0.9, linewidth=1.0)

                mascara_picos_faixa = (freqs_pico >= f_min) & (freqs_pico <= f_max)
                if mascara_picos_faixa.any():
                    ax1.scatter(
                        freqs_pico[mascara_picos_faixa], amp_pico[mascara_picos_faixa],
                        facecolors='none', edgecolors='red', marker='o', s=90,
                        linewidths=1.8, zorder=5, label=f"Picos (N={args.n_picos})"
                    )

                y_max = amplitude[mascara].max() * 1.2
                y_max = max(y_max, 1e-9)

                My_axis(
                    ax1,
                    font=12,
                    xlim=[f_min, f_max],
                    ylim=[0, y_max],
                    setaxis=[
                        f"FFT + Picos ({rotulo}) - {sensor} | {condicao} | {nome_canal_legivel} | {f_min:.0f}-{f_max:.0f} Hz\n",
                        "Frequency (Hz)",
                        "Amplitude"
                    ]
                )

                nome_figura = f"picos_{nome_canal_arquivo}_{rotulo}_{f_min:.0f}-{f_max:.0f}hz.png"
                caminho_figura = pasta_figuras / nome_figura

                plt.tight_layout()
                plt.savefig(caminho_figura, dpi=150)
                plt.close(fig)

                print(f"      🖼️ Figura salva: Figuras/{sensor}/{condicao}/Picos/{nome_figura}")

        if linhas_picos:
            df_picos = pd.DataFrame(linhas_picos)
            salvar_grupo(df_picos, sensor, condicao, output_dir)
            print(f"      💾 {len(linhas_picos)} pico(s) registrado(s) em "
                  f"Etapas/Picos/{sensor}/{condicao}.parquet")

        grupos_ok += 1
        if houve_erro_no_grupo:
            grupos_com_erro += 1

    caminho_log = registrar_log(raiz_path, "04_picos", {
        "data_dir": raiz_path.resolve(),
        "metodo": METODO_DESCRICAO,
        "n_picos": args.n_picos,
        "min_dist_acl_hz": args.min_dist_acl,
        "min_dist_pzt_hz": args.min_dist_pzt,
        "min_dist_fallback_hz": args.min_dist,
        "faixa_baixa_hz": f"0-{args.f1:.0f}",
        "faixa_media_hz": f"{args.f1:.0f}-{args.f2:.0f}",
        "faixa_alta_hz": f"{args.f2:.0f}-Nyquist",
        "quick": args.quick,
        "grupos_processados": grupos_ok,
        "grupos_com_aviso": grupos_com_erro,
    }, pastas_alteradas=pastas_alteradas)

    print("\n" + "=" * 65)
    print(f"✅ Etapa 04 (Picos) Concluída!")
    print(f"   Grupos processados: {grupos_ok} | Grupos com algum aviso: {grupos_com_erro}")
    print(f"💾 Picos (freq/amplitude/faixa por canal) salvos em: {output_dir.resolve()}")
    print(f"📝 Log de parâmetros: {caminho_log.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
