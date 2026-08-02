# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:06:32 2026

@author: profr
"""

# -*- coding: utf-8 -*-
"""
Script para Varredura Multicondição (T1 a T7), Processamento Espectral Automatizado,
e Geração de Gráficos Waterfall 3D (Cascata Espectral).
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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

# =====================================================
# Funções de Processamento de Sinal (Leitura Única)
# =====================================================
def extrair_segundos_arquivo(caminho_file):
    nome = os.path.basename(caminho_file)
    partes = nome.split('_')
    horario = partes[3]  # 'HHMMSS'
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
# Carregamento Único dos Dados
# =====================================================
testes_ordenados = sorted(MAPEAMENTO_TESTES.keys(), key=lambda k: MAPEAMENTO_TESTES[k]['f_inv'])

f_inv_list = []
vazao_list = []
matrix_acl_raw = []
matrix_pzt_raw = []
freqs_grid_ref = None

print("=== CARREGANDO DADOS PARA O WATERFALL 3D ===")
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
f_inv_array = np.array(f_inv_list)
vazao_array = np.array(vazao_list)

# Converter Vazão de m³/h para L/min para ficar idêntico à figura do artigo
vazao_lmin = (vazao_array * 1000) / 60  

# Conversor de escala dinâmico em memória
def obter_matriz_convertida(matrix_raw, modo_escala='norm'):
    eps = 1e-12
    if modo_escala == 'norm':
        picos = np.max(matrix_raw, axis=1, keepdims=True) + eps
        return matrix_raw / picos
    elif modo_escala == 'db':
        picos = np.max(matrix_raw, axis=1, keepdims=True) + eps
        return 20 * np.log10((matrix_raw / picos) + eps)
    else:
        return matrix_raw

# =====================================================
# Função para Plotar o Waterfall 3D
# =====================================================
def plotar_waterfall_3d(matrix_raw, freqs_grid, eixo_y, titulo_sensor, 
                        nome_eixo_y=r"Flow rate (m$^3$/)", f_max=100.0, 
                        modo_escala='norm', cmap_cor='jet'):
    
    # 1. Obtém matriz na escala desejada
    matrix_plot = obter_matriz_convertida(matrix_raw, modo_escala)
    
    # Restringe ao limite x_max (zoom de frequência desejado)
    idx_max = np.searchsorted(freqs_grid, f_max)
    X_freq = freqs_grid[:idx_max]
    Z_data = matrix_plot[:, :idx_max]
    
    # 2. Criar Grade 3D Meshgrid
    X, Y = np.meshgrid(X_freq, eixo_y)
    
    fig = plt.figure(figsize=(12, 9), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    
    # 3. Plotagem das Linhas espectrais 3D (Waterfall)
    # Plotamos cada perfil de vazão/frequência como uma linha 3D individual colorida pela amplitude
    norm = plt.Normalize(np.min(Z_data), np.max(Z_data))
    colormap = cm.get_cmap(cmap_cor)

    for i in range(len(eixo_y)):
        y_val = Y[i, :]
        x_val = X[i, :]
        z_val = Z_data[i, :]
        
        # Desenha a linha 3D do espectro
        ax.plot(x_val, y_val, z_val, color=colormap(norm(np.max(z_val))), linewidth=1.5)
        
        # Preenchimento opcional transparente abaixo da linha para efeito cascatas sólidas
        ax.add_collection3d(
            plt.fill_between(x_val, 0, z_val, color=colormap(norm(np.mean(z_val))), alpha=0.15),
            zs=eixo_y[i], zdir='y'
        )

    # 4. Ajustes do Título e Rótulos
    #ax.set_title(f"Waterfall plot of {titulo_sensor}\n", fontsize=16, fontweight='bold')
    ax.set_xlabel('\nFrequency (Hz)', fontsize=14, labelpad=10)
    ax.set_ylabel(f'\n{nome_eixo_y}', fontsize=14, labelpad=10)
    
    if modo_escala == 'db':
        ax.set_zlabel('\nAmplitude (dB)', fontsize=14, labelpad=10)
    elif modo_escala == 'norm':
        ax.set_zlabel('\nNormalized Amplitude', fontsize=14, labelpad=10)
    else:
        ax.set_zlabel('\nAmplitude', fontsize=14, labelpad=10)
        
    # 5. Ajustes de Ângulo de Visão (idêntico ao MATLAB da imagem de referência)
    #ax.view_init(elev=25, azim=-60)
    #ax.view_init(elev=25, azim=30)
    #ax.view_init(elev=25, azim=-150)
    
    # Ajusta os limites das coordenadas
    ax.set_xlim(0, f_max)
    ax.set_ylim(min(eixo_y), max(eixo_y))
    
    # Estilo do grid das paredes 3D (estilo MATLAB)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    
    plt.tight_layout()
    plt.show()

# =====================================================
# GERAÇÃO DOS GRÁFICOS WATERFALL 3D
# =====================================================

# 1. Waterfall do Acelerômetro por Vazão (L/min)
plotar_waterfall_3d(matrix_acl_raw, freqs_grid_ref, vazao_array, 
                    titulo_sensor="Accelerometer ", 
                    nome_eixo_y=r"Flow rate (m$^3$/h)", 
                    f_max= 150 , modo_escala='norm', cmap_cor='jet')

# 2. Waterfall do Piezo elétrico por Frequência do Inversor (Hz)
plotar_waterfall_3d(matrix_pzt_raw, freqs_grid_ref, vazao_array, 
                    titulo_sensor="Piezoelectric ", 
                    nome_eixo_y=r"Flow rate (m$^3$/h)", 
                    f_max= 150, modo_escala='norm', cmap_cor='jet')