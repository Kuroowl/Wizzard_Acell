# -*- coding: utf-8 -*-
"""
Script para análise automatizada de 7 testes (T1 a T7):
1. Identificação automatizada das frequências do VFD, Bomba (1X) e Cavidades (1X, 2X, 4X).
2. Cálculo de Erro (%) e Amplitude dos picos na FFT.
3. Histograma de Distribuição da Energia Espectral (Picos Dominantes).
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from matplotlib.ticker import MultipleLocator

# =====================================================
# Função de formatação do gráfico
# =====================================================
def My_axis(ax, font=16, 
            ticklengthmajor=12, ticklengthminor=6,
            tickwidthmajor=3, tickwidthminor=3,
            setaxis=['', '', ''], xlim=[0, 1], ylim=[0, 1],
            legbox=[0.98, 0.98, 2, 14], logx=False, logy=False, locator=False):
    
    ticksize = font
    
    # 1. Ajuste dos Limites dos Eixos Primeiro
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    
    # 2. Configuração do Locador X (Log, Fixo/Customizado ou Automático)
    if logx:
        ax.set_xscale("log")
    else:
        if locator:
            # Se locator for True (ou um número), força N divisões principais entre xlim
            nbins = 5 if isinstance(locator, bool) else locator
            ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, prune=None))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        
    # 3. Configuração do Locador Y (Log, Fixo/Customizado ou Automático)
    if logy:
        ax.set_yscale("log")
    else:
        if locator:
            # Se locator for True (ou um número), força N divisões principais entre ylim
            nbins = 5 if isinstance(locator, bool) else locator
            ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune=None))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        
    # 4. Estilo dos Ticks
    ax.tick_params(axis='both', which='major', labelsize=ticksize,
                    width=tickwidthmajor, length=ticklengthmajor, direction='in', pad=10)
    ax.tick_params(axis='both', which='minor',
                    width=tickwidthminor, length=ticklengthminor, direction='in', pad=10)
    
    ax.tick_params(axis='x', which='both', top=True, labeltop=True)
    ax.tick_params(axis='y', which='both', right=True, labelright=True)
    
    # 5. Rótulos e Título
    ax.set_title(setaxis[0], fontsize=font + 2)
    ax.set_xlabel(setaxis[1], fontsize=font + 2)
    ax.set_ylabel(setaxis[2], fontsize=font + 2)
    
    # 6. Legenda
    ax.legend(loc='upper right', bbox_to_anchor=(legbox[0], legbox[1]), 
                fancybox=True, shadow=True, ncol=legbox[2], fontsize=legbox[3])
    
    return ax

# =====================================================
# Tabela de Parâmetros Operacionais Teóricos (T1 a T7)
# =====================================================
PARAMETROS_TESTES = {
    'T7': {'vazao_m3h': 1.0, 'freq_vfd_hz': 7.4, 'freq_bomba_hz': 1.72},
    'T6': {'vazao_m3h': 1.5, 'freq_vfd_hz': 10.8, 'freq_bomba_hz': 2.52},
    'T5': {'vazao_m3h': 2.0, 'freq_vfd_hz': 14.3, 'freq_bomba_hz': 3.34},
    'T4': {'vazao_m3h': 2.5, 'freq_vfd_hz': 18.6, 'freq_bomba_hz': 4.34},
    'T3': {'vazao_m3h': 3.0, 'freq_vfd_hz': 24.0, 'freq_bomba_hz': 5.61},
    'T2': {'vazao_m3h': 3.5, 'freq_vfd_hz': 31.5, 'freq_bomba_hz': 7.36},
    'T1': {'vazao_m3h': 4.0, 'freq_vfd_hz': 42.5, 'freq_bomba_hz': 9.93},
}

BASE_DIR = r"C:\Users\profr\Desktop\TesteBomba\PCP01"
CANAL_ACL = "Channel 3"
CANAL_PZT = "Channel 6"
LISTA_TESTES = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']

# =====================================================
# Funções de Processamento de Sinal
# =====================================================
def aplicar_filtro_passa_banda(sinal, fs, f_corte_inferior=0.5, f_corte_superior=200.0, ordem=2):
    nyquist = 0.5 * fs
    low = f_corte_inferior / nyquist
    high = min(f_corte_superior / nyquist, 0.99)
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

def carregar_e_sincronizar(base_dir, teste):
    path_acl = os.path.join(base_dir, "ACL", teste)
    path_pzt = os.path.join(base_dir, "PZT", teste)
    
    files_acl = glob.glob(os.path.join(path_acl, "*.csv"))
    files_pzt = glob.glob(os.path.join(path_pzt, "*.csv"))
    
    if not files_acl or not files_pzt:
        raise FileNotFoundError(f"Arquivos CSV não encontrados em {path_acl} ou {path_pzt}")
    
    file_acl = files_acl[0]
    file_pzt = files_pzt[0]
    
    df1 = pd.read_csv(file_acl)
    df2 = pd.read_csv(file_pzt)
    
    def extrair_segundos(caminho_file):
        nome = os.path.basename(caminho_file)
        partes = nome.split('_')
        horario = partes[3]
        hh, mm, ss = int(horario[:2]), int(horario[2:4]), int(horario[4:6])
        return hh * 3600 + mm * 60 + ss

    offset_tempo = extrair_segundos(file_pzt) - extrair_segundos(file_acl)
    
    t_acl = df1["Time (s)"].to_numpy()
    t_pzt = df2["Time (s)"].to_numpy() + offset_tempo
    
    x_acl = df1[CANAL_ACL].to_numpy()
    x_pzt = df2[CANAL_PZT].to_numpy()
    
    fs_acl = 1 / np.mean(np.diff(t_acl))
    fs_pzt = 1 / np.mean(np.diff(t_pzt))
    fs_comum = min(fs_acl, fs_pzt)
    
    t_inicio = max(t_acl[0], t_pzt[0])
    t_fim = min(t_acl[-1], t_pzt[-1])
    t_sync = np.arange(t_inicio, t_fim, 1/fs_comum)
    
    f_acl = interp1d(t_acl, x_acl, kind='linear', fill_value='extrapolate')
    f_pzt = interp1d(t_pzt, x_pzt, kind='linear', fill_value='extrapolate')
    
    acl_norm = (f_acl(t_sync) - np.mean(f_acl(t_sync))) / np.std(f_acl(t_sync))
    pzt_norm = (f_pzt(t_sync) - np.mean(f_pzt(t_sync))) / np.std(f_pzt(t_sync))
    
    return acl_norm, pzt_norm, fs_comum

def buscar_pico_proximo(freqs, magnitude, freq_alvo, tolerancia_relativa=0.15):
    """Busca o valor máximo na vizinhança da frequência teórica com tolerância adaptativa"""
    janela_hz = max(0.8, freq_alvo * tolerancia_relativa)
    indices = np.where((freqs >= freq_alvo - janela_hz) & (freqs <= freq_alvo + janela_hz))[0]
    
    if len(indices) == 0:
        return np.nan, np.nan, np.nan
    
    idx_max = indices[np.argmax(magnitude[indices])]
    freq_medida = freqs[idx_max]
    amp_medida = magnitude[idx_max]
    erro_pct = (abs(freq_medida - freq_alvo) / freq_alvo) * 100
    
    return freq_medida, amp_medida, erro_pct

def obter_top_picos(freqs, magnitude, top_n=5, min_distancia_hz=1.5):
    """Coleta os top N picos de maior magnitude no espectro"""
    df = freqs[1] - freqs[0]
    distance_samples = max(1, int(min_distancia_hz / df))
    
    peaks, _ = find_peaks(magnitude, distance=distance_samples)
    
    if len(peaks) == 0:
        peaks = np.argsort(magnitude)[::-1][:top_n]
    else:
        ordem = np.argsort(magnitude[peaks])[::-1]
        peaks = peaks[ordem[:top_n]]
    
    return freqs[peaks], magnitude[peaks]

# =====================================================
# Execução da Análise nos 7 Testes
# =====================================================
resultados_tabela = []
todos_picos_acl = []
todos_picos_pzt = []

print("Iniciando o processamento dos 7 ensaios...\n")

for teste in LISTA_TESTES:
    try:
        acl_norm, pzt_norm, fs_comum = carregar_e_sincronizar(BASE_DIR, teste)
        
        # Filtro Passa-Banda suave (0.5 Hz a 200.0 Hz)
        acl_filt = aplicar_filtro_passa_banda(acl_norm, fs_comum, f_corte_inferior=0.5, f_corte_superior=200.0, ordem=2)
        pzt_filt = aplicar_filtro_passa_banda(pzt_norm, fs_comum, f_corte_inferior=0.5, f_corte_superior=200.0, ordem=2)
        
        freqs, fft_acl = calcular_fft(acl_filt, fs_comum)
        _, fft_pzt = calcular_fft(pzt_filt, fs_comum)
        
        # Parâmetros Teóricos
        vazao = PARAMETROS_TESTES[teste]['vazao_m3h']
        vfd_teorico = PARAMETROS_TESTES[teste]['freq_vfd_hz']
        bomba_1x_teorico = PARAMETROS_TESTES[teste]['freq_bomba_hz']
        cavity_2x_teorico = 2.0 * bomba_1x_teorico
        cavity_4x_teorico = 4.0 * bomba_1x_teorico
        
        # Busca no Acelerômetro
        vfd_f_acl, vfd_a_acl, vfd_e_acl = buscar_pico_proximo(freqs, fft_acl, vfd_teorico)
        c1x_f_acl, c1x_a_acl, c1x_e_acl = buscar_pico_proximo(freqs, fft_acl, bomba_1x_teorico)
        c2x_f_acl, c2x_a_acl, c2x_e_acl = buscar_pico_proximo(freqs, fft_acl, cavity_2x_teorico)
        c4x_f_acl, c4x_a_acl, c4x_e_acl = buscar_pico_proximo(freqs, fft_acl, cavity_4x_teorico)
        
        # Busca no PZT
        vfd_f_pzt, vfd_a_pzt, vfd_e_pzt = buscar_pico_proximo(freqs, fft_pzt, vfd_teorico)
        c1x_f_pzt, c1x_a_pzt, c1x_e_pzt = buscar_pico_proximo(freqs, fft_pzt, bomba_1x_teorico)
        c2x_f_pzt, c2x_a_pzt, c2x_e_pzt = buscar_pico_proximo(freqs, fft_pzt, cavity_2x_teorico)
        c4x_f_pzt, c4x_a_pzt, c4x_e_pzt = buscar_pico_proximo(freqs, fft_pzt, cavity_4x_teorico)
        
        # Guardar na Tabela
        resultados_tabela.append({
            'Teste': teste,
            'Vazão': vazao,
            'VFD Teó': vfd_teorico, 'VFD ACL': vfd_f_acl, 'VFD Err% ACL': vfd_e_acl, 'VFD PZT': vfd_f_pzt, 'VFD Err% PZT': vfd_e_pzt,
            'Cav 1X Teó': bomba_1x_teorico, '1X ACL': c1x_f_acl, '1X Err% ACL': c1x_e_acl, '1X PZT': c1x_f_pzt, '1X Err% PZT': c1x_e_pzt,
            'Cav 2X Teó': cavity_2x_teorico, '2X ACL': c2x_f_acl, '2X Err% ACL': c2x_e_acl, '2X PZT': c2x_f_pzt, '2X Err% PZT': c2x_e_pzt,
            'Cav 4X Teó': cavity_4x_teorico, '4X ACL': c4x_f_acl, '4X Err% ACL': c4x_e_acl, '4X PZT': c4x_f_pzt, '4X Err% PZT': c4x_e_pzt,
        })
        
        # Coletar Top 5 picos para o Histograma
        top_f_acl, _ = obter_top_picos(freqs, fft_acl, top_n=10)
        top_f_pzt, _ = obter_top_picos(freqs, fft_pzt, top_n=10)
        
        todos_picos_acl.extend(top_f_acl)
        todos_picos_pzt.extend(top_f_pzt)
        print(f"[{teste}] Sucesso: {len(top_f_acl)} picos ACL / {len(top_f_pzt)} picos PZT coletizados.")
        
    except Exception as e:
        print(f"[{teste}] ERRO: {e}")

# =====================================================
# Exibição dos Resultados em Tabela
# =====================================================
df_resultados = pd.DataFrame(resultados_tabela)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("\n" + "="*100)
print("TABELA COMPARATIVA - FREQUÊNCIAS DETECTADAS (VFD, CAVITY 1X, 2X, 4X)")
print("="*100)
print(df_resultados.to_string(index=False))

df_resultados.to_csv("resultados_frequencias_detectadas.csv", index=False)
print("\n[INFO] Tabela exportada com sucesso para 'resultados_frequencias_detectadas.csv'.")

# =====================================================
# PLOT: Histograma de Frequências Dominantes
# =====================================================
if len(todos_picos_acl) > 0 and len(todos_picos_pzt) > 0:
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    bins = np.arange(0, 205, 2)  # Bins de 5 Hz

    ax.hist(todos_picos_acl, bins=bins, alpha=0.6, color='green', label='Accelerometer', edgecolor='black')
    ax.hist(todos_picos_pzt, bins=bins, alpha=0.5, color='black', label='Piezo', edgecolor='black')

    My_axis(ax,
            xlim=[0, 100],
            ylim=[0, 15],
            legbox=[0.98, 0.98, 2, 12],
            font=16,
            locator=True,
            setaxis=["Histogram of Dominant Spectral Peaks",
                     "Frequency (Hz)",
                     "Peak Count"])
    
    # 1. Linhas horizontais tracejadas nos major e minor ticks do eixo Y
    ax.grid(True, axis='y', which='major', linestyle='--', alpha=0.7, color='gray', zorder=0)
    ax.grid(True, axis='y', which='minor', linestyle=':', alpha=0.4, color='gray', zorder=0)

    # 2. Configuração dos marcadores finos de 2 em 2 Hz no eixo X
    
    ax.xaxis.set_minor_locator(MultipleLocator(2))  # Ticks secundários exatamente a cada 2 Hz

    # Garante que as barras do histograma fiquem à frente da grade
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.show()

else:
    print("\n[AVISO] Não foi possível gerar o histograma pois nenhum pico foi retornado.")