import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# ==============================================================================
# 🎨 1. PADRÃO ESTÉTICO (my_axis)
# ==============================================================================
def my_axis(ax, title="", xlabel="Tempo (s)", ylabel="Aceleração (m/s²)", grid=True):
    """Aplica a estilização padrão aos eixos do Matplotlib.
    
    Ajuste este bloco para alterar o estilo visual de todos os gráficos do projeto.
    """
    # Remove as bordas superior e direita (estilo limpo)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    # Cores e fontes
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12, color='#2C3E50')
    ax.set_xlabel(xlabel, fontsize=10, fontweight='bold', color='#34495E')
    ax.set_ylabel(ylabel, fontsize=10, fontweight='bold', color='#34495E')

    # Marcadores dos eixos
    ax.tick_params(axis='both', which='major', labelsize=9, colors='#2C3E50')

    # Grid
    if grid:
        ax.grid(True, linestyle='--', alpha=0.5, color='#BDC3C7')

    plt.tight_layout()
    return ax


# ==============================================================================
# 🛠️ 2. MÓDULOS DE TRATAMENTO DO SINAL (Fácil edição e aprimoramento)
# ==============================================================================
def remover_dc_offset(sinal: np.ndarray) -> np.ndarray:
    """Remove a tendência linear / componente DC (offset) do sinal."""
    return signal.detrend(sinal, type='constant')


def aplicar_filtro_passa_baixa(sinal: np.ndarray, fs: float = 1000.0, cutoff: float = 200.0, ordem: int = 4) -> np.ndarray:
    """Aplica filtro Butterworth passa-baixa (zero-phase / filtfilt).
    
    Parâmetros editáveis:
        - fs: Frequência de amostragem em Hz
        - cutoff: Frequência de corte em Hz
        - ordem: Ordem do filtro
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(ordem, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, sinal)


def normalizar_sinal(sinal: np.ndarray, metodo: str = "zscore") -> np.ndarray:
    """Normaliza o sinal.
    
    Opções de 'metodo':
        - 'zscore': Média 0 e Desvio Padrão 1
        - 'max': Escala pelo valor absoluto máximo [-1, 1]
        - 'none': Mantém a amplitude original
    """
    if metodo == "zscore":
        std = np.std(sinal)
        return (sinal - np.mean(sinal)) / std if std != 0 else sinal
    elif metodo == "max":
        max_val = np.max(np.abs(sinal))
        return sinal / max_val if max_val != 0 else sinal
    return sinal


def pipeline_preprocessamento(sinal: np.ndarray, fs: float = 1000.0) -> np.ndarray:
    """Encadeamento sequencial das etapas de pré-processamento."""
    # 1. Remoção de DC Offset
    sinal_tratado = remover_dc_offset(sinal)
    
    # 2. Filtragem Passa-Baixa (Ajuste a frequência de corte aqui se necessário)
    sinal_tratado = aplicar_filtro_passa_baixa(sinal_tratado, fs=fs, cutoff=200.0)
    
    # 3. Normalização
    sinal_tratado = normalizar_sinal(sinal_tratado, metodo="zscore")
    
    return sinal_tratado


# ==============================================================================
# 📊 3. PROCESSAMENTO DOS DADOS E GERAÇÃO DAS FIGURAS
# ==============================================================================
def processar_e_gerar_figuras(df_bruto: pd.DataFrame, raiz_path: Path):
    """Aplica o pré-processamento e salva os gráficos na pasta DadosTratados/Wizzard_Output/Figuras/"""
    
    # Cria a pasta de destino dentro do diretório selecionado
    pasta_figuras = raiz_path / "DadosTratados" / "Wizzard_Output" / "Figuras"
    pasta_figuras.mkdir(parents=True, exist_ok=True)

    # Identifica colunas de valores (excluindo colunas de metadados)
    colunas_metadados = ["sensor", "condicao", "ordem_ensaio", "arquivo_origem"]
    colunas_sinal = [c for c in df_bruto.columns if c not in colunas_metadados]

    df_processado_lista = []

    # Agrupa por Sensor (ACL / PZT) e Condição/Ensaio (T1, T2, ..., Tn)
    grupos = df_bruto.groupby(["sensor", "condicao"])

    print(f"🎨 Gerando figuras e aplicando pré-processamento para {len(grupos)} combinações...")

    for (sensor, condicao), df_grupo in grupos:
        df_grupo = df_grupo.copy().reset_index(drop=True)
        
        # Subpasta para organizar figuras por sensor (ex: Figuras/ACL/ e Figuras/PZT/)
        pasta_sensor_fig = pasta_figuras / sensor
        pasta_sensor_fig.mkdir(parents=True, exist_ok=True)

        # Identifica e aplica o pré-processamento em cada canal/coluna de dados
        for canal in colunas_sinal:
            sinal_bruto = df_grupo[canal].to_numpy()

            # Evita processar colunas que não contenham dados numéricos de sinal
            if not np.issubdtype(sinal_bruto.dtype, np.number):
                continue

            # Aplica a cadeia de tratamento
            sinal_tratado = pipeline_preprocessamento(sinal_bruto)
            df_grupo[f"{canal}_tratado"] = sinal_tratado

            # Eixo do tempo (se não houver coluna de tempo, cria uma baseada no índice)
            tempo = df_grupo["tempo"].to_numpy() if "tempo" in df_grupo.columns else np.arange(len(sinal_tratado))

            # --- PLOTAGEM DA SÉRIE TEMPORAL ---
            fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
            ax.plot(tempo, sinal_tratado, color='#1F77B4', linewidth=1.0, label='Sinal Tratado')
            
            # Aplica o padrão visual estético
            my_axis(
                ax, 
                title=f"Série Temporal - Sensor: {sensor} | Condição: {condicao} | Canal: {canal}",
                xlabel="Tempo (s)" if "tempo" in df_grupo.columns else "Amostras",
                ylabel="Amplitude Normalizada"
            )

            # Salva o gráfico
            nome_figura = f"{condicao}_{canal}_timeseries.png"
            caminho_figura = pasta_sensor_fig / nome_figura
            plt.savefig(caminho_figura, bbox_inches='tight')
            plt.close(fig)

            print(f"  └── 📈 Gráfico salvo: {caminho_figura.relative_to(raiz_path)}")

        df_processado_lista.append(df_grupo)

    # Consolida os dados pré-processados
    df_processado_final = pd.concat(df_processado_lista, ignore_index=True)
    return df_processado_final


def main():
    parser = argparse.ArgumentParser(
        description="02_preprocessamento: Tratamento de sinais e geração de séries temporais."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Caminho raiz selecionado no main",
    )
    args = parser.parse_args()

    raiz_path = Path(args.data_dir)
    input_parquet = Path("outputs") / "intermediarios" / "01_dados_brutos.parquet"

    if not input_parquet.exists():
        print(f"❌ Arquivo de entrada não encontrado: {input_parquet}")
        exit(1)

    try:
        print("📥 Lendo dados consolidados da etapa 01...")
        df_bruto = pd.read_parquet(input_parquet)

        # Processa os sinais e gera as figuras
        df_tratado = processar_e_gerar_figuras(df_bruto, raiz_path)

        # Salva o dataset pré-processado para uso no 03_fft.py
        output_parquet = Path("outputs") / "intermediarios" / "02_dados_preprocessados.parquet"
        df_tratado.to_parquet(output_parquet, index=False)

        print("\n" + "="*50)
        print("✅ Pré-processamento concluído com sucesso!")
        print(f"📁 Figuras salvas em: {(raiz_path / 'DadosTratados' / 'Wizzard_Output' / 'Figuras').resolve()}")
        print(f"💾 Dados processados salvos em: {output_parquet.resolve()}")
        print("="*50 + "\n")

    except Exception as e:
        print(f"\n❌ Erro na etapa 02_preprocessamento: {e}")
        exit(1)


if __name__ == "__main__":
    main()