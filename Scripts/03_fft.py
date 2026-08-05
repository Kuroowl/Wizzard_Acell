from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
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

_estilo = _carregar_modulo("estilo_grafico", "estilo_grafico.py")
My_axis = _estilo.My_axis


# ==============================================================================
# 🛠️ 1. CÁLCULO DA FFT
# ==============================================================================
def calcular_fft(sinal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula o espectro de amplitude de banda completa (0 até Nyquist = fs/2)
    de UMA VEZ, com janela de Hann para reduzir vazamento espectral.

    Os recortes por faixa (baixa/média/alta) são feitos DEPOIS, apenas
    fatiando o resultado — não se filtra o sinal no tempo nem se recalcula
    a FFT por faixa, então nenhum evento na borda de uma faixa é perdido.
    """
    n = len(sinal)
    sinal = sinal - np.mean(sinal)  # garante média zero (defensivo; já vem tratado da etapa 02)

    janela = np.hanning(n)
    sinal_janelado = sinal * janela

    espectro = np.fft.rfft(sinal_janelado)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # Correção de amplitude pela janela (espectro de amplitude, unilateral)
    soma_janela = np.sum(janela)
    amplitude = np.abs(espectro) * 2.0 / soma_janela if soma_janela != 0 else np.abs(espectro)
    amplitude[0] = amplitude[0] / 2.0  # componente DC não deve ser dobrada

    return freqs, amplitude


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
                         help="Limite entre a faixa BAIXA e MÉDIA, em Hz (padrão: 15.0).")
    parser.add_argument("--f2", type=float, default=400.0,
                         help="Limite entre a faixa MÉDIA e ALTA, em Hz (padrão: 400.0).")
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

    print(f"⚙️ Calculando FFT para {len(grupos)} grupo(s) sensor/condição...")

    for sensor, condicao, caminho_parquet in grupos:
        fs = obter_fs(sensor)
        nyquist = fs / 2.0

        # Faixas: baixa [0, f1], média (f1, f2], alta (f2, Nyquist]
        faixas = [
            (0.0, args.f1, "baixa"),
            (args.f1, args.f2, "media"),
            (args.f2, nyquist, "alta"),
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
                freqs, amplitude = calcular_fft(sinal, fs=fs)
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
        "fs_acl_hz": args.fs_acl,
        "fs_pzt_hz": args.fs_pzt,
        "fs_fallback_hz": args.fs,
        "faixa_baixa_hz": f"0-{args.f1:.0f}",
        "faixa_media_hz": f"{args.f1:.0f}-{args.f2:.0f}",
        "faixa_alta_hz": f"{args.f2:.0f}-Nyquist",
        "janela": "hanning",
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