# -*- coding: utf-8 -*-
"""
Script para Varredura Multicondição (T1 a T7), Processamento Espectral Automatizado,
Geração de Mapas Espectrais (Heatmaps) e Formatação Customizada via My_axis.
Processamento ÚNICO de arquivos com conversão dinâmica de escalas (RAW, Norm, dB).
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from matplotlib.ticker import AutoMinorLocator

# =====================================================
# Função Customizada de Edição/Formatação de Gráfico
# =====================================================
def My_axis(ax, font=16, 
            ticklengthmajor=12, ticklengthminor=6,
            tickwidthmajor=3, tickwidthminor=3,
            setaxis=['', '', ''], xlim=[0, 1], ylim=[0, 1],
            legbox=[0.98, 0.98, 2, 14], logx=False, logy=False, locator=False):
    
    ticksize = font
    
    if logx:
        ax.set_xscale("log")
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        
    if logy:
        ax.set_yscale("log")
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))

    ax.tick_params(axis='both', which='major', labelsize=ticksize,
                    width=tickwidthmajor, length=ticklengthmajor, direction='in', pad=10)
    ax.tick_params(axis='both', which='minor',
                    width=tickwidthminor, length=ticklengthminor, direction='in', pad=10)
    
    ax.tick_params(axis='x', which='both', top=True, labeltop=True)
    ax.tick_params(axis='y', which='both', right=True, labelright=True)
    
    ax.set_title(setaxis[0], fontsize=font + 2)
    ax.set_xlabel(setaxis[1], fontsize=font + 2)
    ax.set_ylabel(setaxis[2], fontsize=font + 2)
    
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    
    ax.legend(loc='upper right', bbox_to_anchor=(legbox[0], legbox[1]), 
               fancybox=True, shadow=True, ncol=legbox[2], fontsize=legbox[3])
    
    return ax

# =====================================================
# Configurações Globais e Mapeamento de Testes
# =====================================================
BASE_DIR = r"C:\Users\profr\Desktop\TesteBomba\PCP01"

CANAL_ACL = "Channel 3"
CANAL_PZT = "Channel 6"

MAPEAMENTO_TESTES = {
    'T1': {'vazao': 4.0, 'f_inv': 42.5},
    'T2': {'vazao': 3.5, 'f_inv': 31.5},
    'T3': {'vazao': 3.0, 'f_inv': 24.0},
    'T4': {'vazao': 2.5, 'f_inv': 18.6},
    'T5': {'vazao': 2.0, 'f_inv': 14.3},
    'T6': {'vazao': 1.5, 'f_inv': 10.8},
    'T7': {'vazao': 1.0, 'f_inv': 7.4}
}

REDUCAO_SEW = 4.28       # Relação de redução do moto-redutor
REDUCAO_CAVIDADE = 9.55   # Relação aproximada de redução da cavidade PCP

# =====================================================
# Funções de Processamento de Sinal (Executa Apenas 1 Vez)
# =====================================================
def extrair_segundos_arquivo(caminho_file):
    nome = os.path.basename(caminho_file)
    partes = nome.split('_')
    horario = partes[3]  # Pega 'HHMMSS'
    hh, mm, ss = int(horario[:2]), int(horario[2:4]), int(horario[4:6])
    return hh * 3600 + mm * 60 + ss

def aplicar_filtro_passa_banda(sinal, fs, f_corte_inferior=3.0, f_corte_superior=200.0, ordem=4):
    nyquist = 0.5 * fs
    low = f_corte_inferior / nyquist
    high = f_corte_superior / nyquist
    b, a = butter(ordem, [low, high], btype='band')
    return filtfilt(b, a, sinal)

def calcular_fft(sinal, fs):
    sinal = sinal - np.mean(sinal)
    N = len(sinal)
    janela = np.hanning(N)
    sinal_janelado = sinal * janela
    
    fft_vals = np.fft.rfft(sinal_janelado)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    magnitude = np.abs(fft_vals) * (2.0 / N)
    return freqs, magnitude

def processar_ensaio_individual(base_dir, teste, f_max_heatmap=200.0, df_heatmap=0.1):
    """Lê os CSVs e calcula a FFT Bruta (RAW) do ensaio."""
    path_acl = os.path.join(base_dir, "ACL", teste)
    path_pzt = os.path.join(base_dir, "PZT", teste)
    
    files_acl = glob.glob(os.path.join(path_acl, "*.csv"))
    files_pzt = glob.glob(os.path.join(path_pzt, "*.csv"))
    
    if not files_acl or not files_pzt:
        print(f"[AVISO] Arquivos não encontrados para o teste {teste}. Pulando...")
        return None, None, None
        
    file_acl, file_pzt = files_acl[0], files_pzt[0]
    
    df1 = pd.read_csv(file_acl)
    df2 = pd.read_csv(file_pzt)
    
    segundos_acl = extrair_segundos_arquivo(file_acl)
    segundos_pzt = extrair_segundos_arquivo(file_pzt)
    offset_tempo = segundos_pzt - segundos_acl
    
    t_acl = df1["Time (s)"].to_numpy()
    t_pzt = df2["Time (s)"].to_numpy() + offset_tempo
    
    x_acl = df1[CANAL_ACL].to_numpy()
    x_pzt = df2[CANAL_PZT].to_numpy()
    
    fs_acl = 1.0 / np.mean(np.diff(t_acl))
    fs_pzt = 1.0 / np.mean(np.diff(t_pzt))
    fs_comum = min(fs_acl, fs_pzt)
    
    t_inicio = max(t_acl[0], t_pzt[0])
    t_fim = min(t_acl[-1], t_pzt[-1])
    t_sync = np.arange(t_inicio, t_fim, 1.0 / fs_comum)
    
    f_acl_interp = interp1d(t_acl, x_acl, kind='linear', fill_value='extrapolate')(t_sync)
    f_pzt_interp = interp1d(t_pzt, x_pzt, kind='linear', fill_value='extrapolate')(t_sync)
    
    acl_norm = (f_acl_interp - np.mean(f_acl_interp)) / np.std(f_acl_interp)
    pzt_norm = (f_pzt_interp - np.mean(f_pzt_interp)) / np.std(f_pzt_interp)
    
    acl_filt = aplicar_filtro_passa_banda(acl_norm, fs_comum, f_corte_inferior=3.0, f_corte_superior=f_max_heatmap)
    pzt_filt = aplicar_filtro_passa_banda(pzt_norm, fs_comum, f_corte_inferior=3.0, f_corte_superior=f_max_heatmap)
    
    freqs_acl, fft_acl = calcular_fft(acl_filt, fs_comum)
    freqs_pzt, fft_pzt = calcular_fft(pzt_filt, fs_comum)
    
    freqs_grid = np.arange(0, f_max_heatmap, df_heatmap)
    interp_fft_acl = interp1d(freqs_acl, fft_acl, bounds_error=False, fill_value=0)(freqs_grid)
    interp_fft_pzt = interp1d(freqs_pzt, fft_pzt, bounds_error=False, fill_value=0)(freqs_grid)
    
    return freqs_grid, interp_fft_acl, interp_fft_pzt

# =====================================================
# Carregamento Único na Memória
# =====================================================
testes_ordenados = sorted(MAPEAMENTO_TESTES.keys(), key=lambda k: MAPEAMENTO_TESTES[k]['f_inv'])

f_inv_list = []
vazao_list = []
matrix_acl_raw = []
matrix_pzt_raw = []
freqs_grid_ref = None

print("=== CARREGANDO E PROCESSANDO ARQUIVOS (UMA ÚNICA VEZ) ===")
for teste in testes_ordenados:
    f_inv = MAPEAMENTO_TESTES[teste]['f_inv']
    vazao = MAPEAMENTO_TESTES[teste]['vazao']
    
    print(f"Lendo {teste} | Inversor: {f_inv:4.1f} Hz | Vazão: {vazao:3.1f} m³/h...")
    freqs_grid, fft_acl, fft_pzt = processar_ensaio_individual(BASE_DIR, teste, f_max_heatmap=200.0)
    
    if fft_acl is not None:
        freqs_grid_ref = freqs_grid
        f_inv_list.append(f_inv)
        vazao_list.append(vazao)
        matrix_acl_raw.append(fft_acl)
        matrix_pzt_raw.append(fft_pzt)

matrix_acl_raw = np.array(matrix_acl_raw)
matrix_pzt_raw = np.array(matrix_pzt_raw)
print("=== CARREGAMENTO CONCLUÍDO COM SUCESSO! ===\n")

# =====================================================
# Conversão Matemática Instantânea em Memória
# =====================================================
def obter_matriz_convertida(matrix_raw, modo_escala='norm'):
    """Converte a matriz RAW carregada para 'norm' ou 'db' sem reler arquivos."""
    eps = 1e-12
    if modo_escala == 'norm':
        # Normaliza cada linha pelo seu próprio pico máximo
        picos = np.max(matrix_raw, axis=1, keepdims=True) + eps
        return matrix_raw / picos
    elif modo_escala == 'db':
        # Calcula dB relativo ao pico de cada linha
        picos = np.max(matrix_raw, axis=1, keepdims=True) + eps
        return 20 * np.log10((matrix_raw / picos) + eps)
    else:
        return matrix_raw

# =====================================================
# Função de Plotagem
# =====================================================
def plotar_heatmap_espectral(matrix_raw, freqs_grid, f_inv_array, titulo_sensor, cmap_cor='viridis', modo_escala='norm'):
    
    # 1. Aplica a conversão matemática na memória
    dados_plot = obter_matriz_convertida(matrix_raw, modo_escala)
    
    # 2. Configuração de escala e eixos (Seu código customizado)
    if modo_escala == 'db':
        v_min = -40  # Trunca ruídos abaixo de -40 dB para destacar o sinal principal
        v_max = 0    # Pico máximo em 0 dB
        label_cbar = 'FFT Magnitude (dB)'
    elif modo_escala == 'norm':
        v_min = 0
        v_max = 0.2    # Saturação de 0 a 1
        label_cbar = 'FFT Magnitude - Normalized'
    else:
        v_min = np.min(dados_plot)
        v_max = np.max(dados_plot)
        label_cbar = 'FFT Magnitude - Raw'

    # 3. Plot do Heatmap (imshow)
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    
    im = ax.imshow(dados_plot, aspect='auto', origin='lower', cmap=cmap_cor,
                   extent=[freqs_grid[0], freqs_grid[-1], min(f_inv_array), max(f_inv_array)],
                   vmin=v_min, vmax=v_max, interpolation='bicubic')
    
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.14, aspect=50)
    cbar.set_label(label_cbar, fontsize=12, labelpad=8)
    cbar.ax.tick_params(labelsize=12)
    
    # 4. Linhas Teóricas Cinemáticas
    f_inv_cont = np.linspace(min(f_inv_array), max(f_inv_array), 200)
    
    ax.plot(f_inv_cont, f_inv_cont, c='black', alpha=0.6, linewidth=3, label=r'1X Motor / VFD')
    ax.plot(f_inv_cont / REDUCAO_SEW, f_inv_cont, c='red', alpha=0.6, linewidth=3, label=r'1X Pump Shaft')
    
    f_cav_sinal = f_inv_cont / REDUCAO_CAVIDADE
    ax.plot(f_cav_sinal, f_inv_cont, c='red', alpha=0.7, linestyle='--', linewidth=3, label=r'1X Cavity')
    ax.plot(2 * f_cav_sinal, f_inv_cont, c='red', alpha=0.7, linestyle='-.', linewidth=3, label=r'2X Cavity')
    ax.plot(4 * f_cav_sinal, f_inv_cont, c='red', alpha=0.7, linestyle=':', linewidth=3, label=r'4X Cavity')
    
    # 5. Formatação Estética via My_axis
    ax = My_axis(ax,
                 font=18,
                 ticklengthmajor=10, ticklengthminor=5,
                 tickwidthmajor=4, tickwidthminor=2,
                 setaxis=[f'Spectral Map - {titulo_sensor}', 
                          'Frequency (Hz)', 
                          'VFD Frequency (Hz)'],
                 xlim=[0, 100],
                 ylim=[7.4, 42],
                 legbox=[0.98, 0.43, 1, 14],
                 logx=False, logy=False)
    
    plt.tight_layout()
    plt.show()

# =====================================================
# GERAÇÃO DOS GRÁFICOS (INSTANTÂNEA!)
# =====================================================
# # Plot 3: Se quiser em amplitude Bruta (Raw):
# plotar_heatmap_espectral(matrix_acl_raw, freqs_grid_ref, f_inv_list, 
#                           "Accelerometer (Raw)", cmap_cor='viridis', modo_escala='raw')


# # Plot 1: Normalizado Linearmente (0 a 1)
# plotar_heatmap_espectral(matrix_acl_raw, freqs_grid_ref, f_inv_list, 
#                          "Accelerometer (Normalized)", cmap_cor='viridis', modo_escala='norm')

# # Plot 2: Escala Logarítmica (dB)
# plotar_heatmap_espectral(matrix_acl_raw, freqs_grid_ref, f_inv_list, 
#                          "Accelerometer (dB)", cmap_cor='plasma', modo_escala='db')


# Plot 3: Se quiser em amplitude Bruta (Raw):
plotar_heatmap_espectral(matrix_pzt_raw, freqs_grid_ref, f_inv_list, 
                          "Piezo (Raw)", cmap_cor='viridis', modo_escala='raw')


# Plot 1: Normalizado Linearmente (0 a 1)
plotar_heatmap_espectral(matrix_pzt_raw, freqs_grid_ref, f_inv_list, 
                         "Piezo (Normalized)", cmap_cor='viridis', modo_escala='norm')

# Plot 2: Escala Logarítmica (dB)
plotar_heatmap_espectral(matrix_pzt_raw, freqs_grid_ref, f_inv_list, 
                         "Piezo (dB)", cmap_cor='plasma', modo_escala='db')
