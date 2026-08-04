import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.ticker import AutoMinorLocator


# ==============================================================================
# 🎨 1. FUNÇÃO DE ESTILIZAÇÃO GRÁFICA (Padrão Científico)
# ==============================================================================
def My_axis(ax, font=14, 
            ticklengthmajor=10, ticklengthminor=5,
            tickwidthmajor=2, tickwidthminor=1.5,
            setaxis=['', '', ''], xlim=[0, 1], ylim=[-1, 1],
            legbox=[0.98, 0.98, 1, 10], logx=False, logy=False):
    """
    Aplica a estilização científica avançada nos eixos do Matplotlib.
    Adiciona minor ticks, direciona marcas para dentro e espelha marcas nos eixos opostos.
    """
    ticksize = font
    
    # Configuração de escala e marcadores secundários (minor locators)
    if logx:
        ax.set_xscale("log")
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        
    if logy:
        ax.set_yscale("log")
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))

    # Estilização das marcas dos eixos (ticks para dentro)
    ax.tick_params(axis='both', which='major', labelsize=ticksize,
                    width=tickwidthmajor, length=ticklengthmajor, direction='in', pad=8)
    ax.tick_params(axis='both', which='minor',
                    width=tickwidthminor, length=ticklengthminor, direction='in', pad=8)
    
    # Ativa marcas no topo e na direita (espelhamento)
    ax.tick_params(axis='x', which='both', top=True, labeltop=False)
    ax.tick_params(axis='y', which='both', right=True, labelright=False)
    
    # Rótulos e Título
    ax.set_title(setaxis[0], fontsize=font + 2)
    ax.set_xlabel(setaxis[1], fontsize=font)
    ax.set_ylabel(setaxis[2], fontsize=font)
    
    # Limites dos eixos
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    
    # Legenda (apenas se existirem rótulos declarados no plot)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc='upper right', bbox_to_anchor=(legbox[0], legbox[1]), 
                  fancybox=True, shadow=True, ncol=legbox[2], fontsize=legbox[3])
    
    return ax


# ==============================================================================
# 🛠️ 2. MÓDULOS DE TRATAMENTO DO SINAL
# ==============================================================================
def remover_dc_offset(sinal: np.ndarray) -> np.ndarray:
    """Remove o offset DC / tendência do sinal."""
    return signal.detrend(sinal, type='constant')


def aplicar_filtro_passa_baixa(sinal: np.ndarray, fs: float = 1000.0, cutoff: float = 200.0, ordem: int = 4) -> np.ndarray:
    """Aplica filtro Butterworth passa-baixa (zero-phase com filtfilt)."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(ordem, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, sinal)


def normalizar_sinal(sinal: np.ndarray, metodo: str = "zscore") -> np.ndarray:
    """Normaliza a amplitude do sinal."""
    if metodo == "zscore":
        std = np.std(sinal)
        return (sinal - np.mean(sinal)) / std if std != 0 else sinal
    elif metodo == "max":
        max_val = np.max(np.abs(sinal))
        return sinal / max_val if max_val != 0 else sinal
    return sinal


def pipeline_preprocessamento(sinal: np.ndarray, fs: float = 1000.0) -> np.ndarray:
    """Encadeamento sequencial das etapas de tratamento do sinal."""
    sinal_tratado = remover_dc_offset(sinal)
    sinal_tratado = aplicar_filtro_passa_baixa(sinal_tratado, fs=fs, cutoff=200.0)
    sinal_tratado = normalizar_sinal(sinal_tratado, metodo="zscore")
    return sinal_tratado


# ==============================================================================
# 🚀 3. EXECUÇÃO PRINCIPAL DO SCRIPT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    raiz_path = Path(args.data_dir)
    nome_projeto = raiz_path.name

    # 1. Entrada esperada do script 01_leitura.py
    input_file = raiz_path / "DadosTratados" / f"{nome_projeto}_leitura.pkl"

    if not input_file.exists():
        print(f"❌ Arquivo de entrada não encontrado: {input_file.resolve()}")
        print(" Certifique-se de executar a etapa 01_leitura antes desta.")
        exit(1)

    print(f"📖 Lendo dados da etapa de leitura: {input_file.name}")
    df = pd.read_pickle(input_file)

    colunas_metadados = ["sensor", "condicao", "ordem_ensaio", "arquivo_origem"]
    colunas_sinal = [col for col in df.columns if col not in colunas_metadados]

    df_processado_lista = []

    # 2. Iteração sobre cada grupo de Sensor (ACL, PZT) e Condição (T1, T2, ...)
    grupos = df.groupby(["sensor", "condicao"])
    print(f"⚙️ Processando sinais e gerando figuras para {len(grupos)} combinações...")

    for (sensor, condicao), group in grupos:
        group_copy = group.copy().reset_index(drop=True)

        # Pasta de saída das figuras: DadosTratados/Figuras/ACL/T1/
        pasta_figuras = raiz_path / "DadosTratados" / "Figuras" / str(sensor) / str(condicao)
        pasta_figuras.mkdir(parents=True, exist_ok=True)

        for idx_ch, col_canal in enumerate(colunas_sinal, start=1):
            sinal_bruto = group_copy[col_canal].to_numpy()

            if not np.issubdtype(sinal_bruto.dtype, np.number):
                continue

            # Aplica o pipeline de tratamento de sinal
            sinal_tratado = pipeline_preprocessamento(sinal_bruto)
            group_copy[f"{col_canal}_tratado"] = sinal_tratado

            # Definição do eixo temporal
            tempo_plot = range(len(sinal_tratado))

            # --- Gerar Gráfico ---
            fig, ax1 = plt.subplots(figsize=(10, 5))

            cor_linha = 'green' if str(sensor).upper() == 'ACL' else 'black'
            ax1.plot(tempo_plot, sinal_tratado, label=f"{sensor} Ch{idx_ch}", c=cor_linha, alpha=0.85, linewidth=1.2)

            # Cálculo dinâmico dos limites do eixo Y
            y_lim = max(abs(sinal_tratado.min()), abs(sinal_tratado.max())) * 1.2
            y_lim = max(y_lim, 1.0)

            # Aplicação da formatação personalizada
            My_axis(
                ax1,
                font=12,
                xlim=[0, len(sinal_tratado)],
                ylim=[-y_lim, y_lim],
                legbox=[0.98, 0.98, 1, 9],
                setaxis=[
                    f"Série Temporal - {sensor} | {condicao}\n",
                    "Amostras",
                    "Amplitude Normalizada"
                ]
            )

            # Salvamento no padrão
            nome_figura = f"time_serie_ch{idx_ch}.png"
            caminho_figura = pasta_figuras / nome_figura

            plt.tight_layout()
            plt.savefig(caminho_figura, dpi=150)
            plt.close(fig)

            print(f"   └── 🖼️ Figura salva: Figuras/{sensor}/{condicao}/{nome_figura}")

        df_processado_lista.append(group_copy)

    # Recompõe o DataFrame completo com os sinais tratados
    df_final = pd.concat(df_processado_lista, ignore_index=True)

    # 3. Exportação do arquivo esperado pela próxima etapa (03_fft.py)
    output_file = raiz_path / "DadosTratados" / f"{nome_projeto}_preprocessamento.pkl"
    df_final.to_pickle(output_file)

    print("\n" + "=" * 65)
    print(f"✅ Etapa 02 (Pré-processamento) Concluída com Sucesso!")
    print(f"💾 Arquivo salvo para a próxima etapa: {output_file.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()