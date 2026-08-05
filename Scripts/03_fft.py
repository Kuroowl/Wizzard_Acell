from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch


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
# 🛠️ 1. CÁLCULO DA FFT — scipy.signal.welch (recomendação da gerência)
# ==============================================================================
METODO_ESPECTRAL = (
    "scipy.signal.welch (método de Welch: divide o sinal em segmentos "
    "sobrepostos, aplica janela em cada um, calcula a FFT de cada segmento "
    "e faz a média — reduz a variância do espectro em relação a uma FFT "
    "única sobre o sinal inteiro)."
)


def calcular_fft(sinal: np.ndarray, fs: float, nperseg: int, noverlap: int, janela: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula o espectro de amplitude de banda completa (0 até Nyquist = fs/2)
    via scipy.signal.welch, com scaling='spectrum' (Welch retorna a média do
    quadrado por segmento; aqui tiramos a raiz para voltar à mesma unidade de
    amplitude usada no resto do pipeline — picos, heatmap etc.).

    Os recortes por faixa (low/mid/high) são feitos DEPOIS, apenas fatiando o
    resultado — não se filtra o sinal no tempo nem se recalcula o Welch por
    faixa, então nenhum evento na borda de uma faixa é perdido.
    """
    sinal = sinal - np.mean(sinal)  # garante média zero (defensivo; já vem tratado da etapa 02)

    nperseg_efetivo = min(nperseg, len(sinal))
    noverlap_efetivo = min(noverlap, nperseg_efetivo - 1) if nperseg_efetivo > 1 else 0

    freqs, pxx = welch(
        sinal, fs=fs, window=janela, nperseg=nperseg_efetivo, noverlap=noverlap_efetivo,
        scaling="spectrum", detrend="constant", return_onesided=True,
    )
    amplitude = np.sqrt(np.maximum(pxx, 0.0))
    return freqs, amplitude, nperseg_efetivo, noverlap_efetivo


# ==============================================================================
# 🚀 2. EXECUÇÃO PRINCIPAL
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--quick", action="store_true",
                         help="Processa (e plota) apenas o primeiro grupo sensor/condicao, para teste rápido.")
    parser.add_argument("--fs-acl", type=float, default=30000.0,
                         help="Taxa de amostragem (Hz) dos sensores ACL (padrão: 30000.0).")
    parser.add_argument("--fs-pzt", type=float, default=12500.0,
                         help="Taxa de amostragem (Hz) dos sensores PZT (padrão: 12500.0).")
    parser.add_argument("--fs", type=float, default=None,
                         help="Taxa de amostragem (Hz) de fallback para sensores fora do mapeamento ACL/PZT.")
    parser.add_argument("--f1", type=float, default=15.0,
                         help="Limite entre a faixa LOW e MID, em Hz (padrão: 15.0).")
    parser.add_argument("--f2", type=float, default=400.0,
                         help="Limite entre a faixa MID e HIGH, em Hz (padrão: 400.0).")
    parser.add_argument("--nperseg", type=int, default=8192,
                         help="Tamanho do segmento (nperseg) do scipy.signal.welch, em amostras (padrão: 8192). "
                              "Se o sinal for menor, usa o tamanho do sinal inteiro (sem erro).")
    parser.add_argument("--noverlap", type=int, default=None,
                         help="Sobreposição entre segmentos (noverlap) do welch, em amostras "
                              "(padrão: metade do nperseg efetivo, ou seja, 50%%).")
    parser.add_argument("--janela", type=str, default="hann",
                         help="Janela usada pelo welch (qualquer nome aceito por scipy.signal.get_window; padrão: hann).")
    parser.add_argument("--salvar-figuras", action="store_true",
                         help="Gera as figuras de FFT por faixa (Figuras/{sensor}/{condicao}/FFTs/). "
                              "Desligado por padrão: a etapa 04 (picos) já gera o mesmo gráfico "
                              "com os picos marcados, então manter as duas seria redundante.")
    args = parser.parse_args()

    fs_por_sensor = {"ACL": args.fs_acl, "PZT": args.fs_pzt}

    def obter_fs(sensor: str) -> float:
        fs_sensor = fs_por_sensor.get(str(sensor).upper())
        if fs_sensor is not None:
            return fs_sensor
        if args.fs is not None:
            return args.fs
        print(f"   ⚠️ Sensor '{sensor}' sem fs mapeado e sem --fs de fallback; usando 1000.0 Hz.")
        return 1000.0

    raiz_path = Path(args.data_dir)

    input_dir = raiz_path / "DadosTratados" / "Etapas" / "Preprocessamento"
    grupos = listar_grupos(input_dir)

    if not grupos:
        print(f"❌ Nenhum grupo encontrado em: {input_dir.resolve()}")
        print(" Certifique-se de executar a etapa 02_preprocessamento antes desta.")
        exit(1)

    if args.quick:
        grupos = grupos[:1]
        print("⚡ Modo rápido (--quick): processando apenas o primeiro grupo.\n")

    pasta_figuras_raiz = raiz_path / "DadosTratados" / "Figuras"
    output_dir = raiz_path / "DadosTratados" / "Etapas" / "FFT"

    grupos_ok, grupos_com_erro = 0, 0
    pastas_alteradas = {output_dir}
    noverlap_configurado = args.noverlap if args.noverlap is not None else args.nperseg // 2

    print(f"⚙️ Calculando FFT (Welch) para {len(grupos)} grupo(s) sensor/condição...")
    print(f"   Método: {METODO_ESPECTRAL}")
    print(f"   janela={args.janela} | nperseg={args.nperseg} | noverlap={noverlap_configurado}\n")

    for sensor, condicao, caminho_parquet in grupos:
        fs = obter_fs(sensor)
        nyquist = fs / 2.0

        # Faixas: low [0, f1], mid (f1, f2], high (f2, Nyquist]
        faixas = [
            (0.0, args.f1, "low"),
            (args.f1, args.f2, "mid"),
            (args.f2, nyquist, "high"),
        ]

        print(f"\n📖 Grupo: [{sensor}] | [{condicao}]  ←  {caminho_parquet.name}  "
              f"(fs={fs:.1f} Hz, Nyquist={nyquist:.1f} Hz)")

        try:
            df_grupo = carregar_grupo(caminho_parquet)
        except Exception as e:
            print(f"   ⚠️ Erro ao carregar grupo: {e}. Pulando.")
            grupos_com_erro += 1
            continue

        # Só processa colunas que a etapa 02 gerou como sinal limpo (sufixo "_tratado")
        colunas_tratadas = [c for c in df_grupo.columns if str(c).endswith("_tratado")]

        if not colunas_tratadas:
            print(f"   ⚠️ Nenhuma coluna '_tratado' encontrada neste grupo (rode a etapa 02 primeiro). Pulando.")
            grupos_com_erro += 1
            continue

        # Padrão: Figuras/{sensor}/{condicao}/FFTs/ (mesmo nível que a pasta
        # TimeSerie/ gerada na etapa 02). Só é criada se --salvar-figuras
        # for passado; por padrão a etapa 04 (picos) cobre esse gráfico.
        pasta_figuras = None
        if args.salvar_figuras:
            pasta_figuras = pasta_figuras_raiz / str(sensor) / str(condicao) / "FFTs"
            pasta_figuras.mkdir(parents=True, exist_ok=True)
            pastas_alteradas.add(pasta_figuras)

        espectros_grupo = {}  # nome_canal -> (freqs, amplitude), para salvar tudo junto no final
        houve_erro_no_grupo = False

        for col_canal in colunas_tratadas:
            sufixo = "_tratado"
            nome_canal_legivel = str(col_canal)[:-len(sufixo)] if str(col_canal).endswith(sufixo) else str(col_canal)
            nome_canal_arquivo = "".join(
                ch.lower() if ch.isalnum() else "_" for ch in nome_canal_legivel
            ).strip("_") or "canal"

            sinal = df_grupo[col_canal].to_numpy()

            if sinal.size < 8 or not np.all(np.isfinite(sinal)):
                print(f"      ⚠️ Canal {nome_canal_legivel}: sinal vazio/curto/inválido, pulando FFT.")
                houve_erro_no_grupo = True
                continue

            try:
                freqs, amplitude, nperseg_efetivo, noverlap_efetivo = calcular_fft(
                    sinal, fs=fs, nperseg=args.nperseg, noverlap=noverlap_configurado, janela=args.janela
                )
            except Exception as e:
                print(f"      ⚠️ Erro ao calcular FFT do canal {nome_canal_legivel}: {e}. Pulado.")
                houve_erro_no_grupo = True
                continue

            espectros_grupo[nome_canal_legivel] = (freqs, amplitude)

            if pasta_figuras is not None:
                for f_min, f_max, rotulo in faixas:
                    mascara = (freqs >= f_min) & (freqs <= f_max)
                    if not mascara.any():
                        continue

                    fig, ax1 = plt.subplots(figsize=(10, 5))
                    cor_linha = 'green' if str(sensor).upper() == 'ACL' else 'black'
                    ax1.plot(freqs[mascara], amplitude[mascara], c=cor_linha, alpha=0.9, linewidth=1.0)

                    y_max = amplitude[mascara].max() * 1.2
                    y_max = max(y_max, 1e-9)

                    My_axis(
                        ax1,
                        font=12,
                        xlim=[f_min, f_max],
                        ylim=[0, y_max],
                        setaxis=[
                            f"FFT ({rotulo}) - {sensor} | {condicao} | {nome_canal_legivel} | {f_min:.0f}-{f_max:.0f} Hz\n",
                            "Frequency (Hz)",
                            "Amplitude"
                        ]
                    )

                    nome_figura = f"fft_{nome_canal_arquivo}_{rotulo}_{f_min:.0f}-{f_max:.0f}hz.png"
                    caminho_figura = pasta_figuras / nome_figura

                    plt.tight_layout()
                    plt.savefig(caminho_figura, dpi=150)
                    plt.close(fig)

                    print(f"      🖼️ Figura salva: Figuras/{sensor}/{condicao}/FFTs/{nome_figura}")

        if espectros_grupo:
            # Salva todos os canais do grupo num único parquet: freq_hz + uma coluna de amplitude por canal
            freqs_ref = next(iter(espectros_grupo.values()))[0]
            df_fft = pd.DataFrame({"freq_hz": freqs_ref})
            for nome_canal, (freqs, amplitude) in espectros_grupo.items():
                df_fft[f"{nome_canal}_amplitude"] = amplitude
            salvar_grupo(df_fft, sensor, condicao, output_dir)

        grupos_ok += 1
        if houve_erro_no_grupo:
            grupos_com_erro += 1

    caminho_log = registrar_log(raiz_path, "03_fft", {
        "data_dir": raiz_path.resolve(),
        "metodo_espectral": METODO_ESPECTRAL,
        "janela": args.janela,
        "nperseg": args.nperseg,
        "noverlap": noverlap_configurado,
        "scaling": "spectrum (amplitude = sqrt(Pxx), mesma unidade da FFT usada nas etapas 04/05)",
        "detrend": "constant",
        "fs_acl_hz": args.fs_acl,
        "fs_pzt_hz": args.fs_pzt,
        "fs_fallback_hz": args.fs,
        "faixa_low_hz": f"0-{args.f1:.0f}",
        "faixa_mid_hz": f"{args.f1:.0f}-{args.f2:.0f}",
        "faixa_high_hz": f"{args.f2:.0f}-Nyquist",
        "salvar_figuras": args.salvar_figuras,
        "quick": args.quick,
        "grupos_processados": grupos_ok,
        "grupos_com_aviso": grupos_com_erro,
    }, pastas_alteradas=pastas_alteradas)

    print("\n" + "=" * 65)
    print(f"✅ Etapa 03 (FFT) Concluída!")
    print(f"   Grupos processados: {grupos_ok} | Grupos com algum aviso: {grupos_com_erro}")
    print(f"💾 Espectros salvos em: {output_dir.resolve()}")
    print(f"📝 Log de parâmetros: {caminho_log.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()