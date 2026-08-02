# -*- coding: utf-8 -*-
"""
Script para sincronização por tempo relativo (offset automático),
cálculo de FFT com filtro Passa-Banda e identificação dos 5 picos dominantes.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks

# =====================================================
# Função de formatação do gráfico
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
        from matplotlib.ticker import AutoMinorLocator
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        
    if logy:
        ax.set_yscale("log")
    else:
        from matplotlib.ticker import AutoMinorLocator
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
# Configuração Simplificada (Só mude aqui!)
# =====================================================
TESTE = 'T1'  # Altere para 'T1', 'T2', etc.

BASE_DIR = r"C:\Users\profr\Desktop\TesteBomba\PCP01"

CANAL_ACL = "Channel 3"
CANAL_PZT = "Channel 6"

# =====================================================
# Função para Carregar e Sincronizar Automático
# =====================================================
def carregar_dados_teste(base_dir, teste):
    path_acl = os.path.join(base_dir, "ACL", teste)
    path_pzt = os.path.join(base_dir, "PZT", teste)
    
    file_acl = glob.glob(os.path.join(path_acl, "*.csv"))[0]
    file_pzt = glob.glob(os.path.join(path_pzt, "*.csv"))[0]
    
    print(f"\n=================== CARREGANDO {teste} ===================")
    print(f"ACL File: {os.path.basename(file_acl)}")
    print(f"PZT File: {os.path.basename(file_pzt)}")
    
    df1 = pd.read_csv(file_acl)
    df2 = pd.read_csv(file_pzt)
    
    def extrair_segundos_arquivo(caminho_file):
        nome = os.path.basename(caminho_file)
        partes = nome.split('_')
        horario = partes[3]  # Pega 'HHMMSS'
        hh, mm, ss = int(horario[:2]), int(horario[2:4]), int(horario[4:6])
        return hh * 3600 + mm * 60 + ss

    segundos_acl = extrair_segundos_arquivo(file_acl)
    segundos_pzt = extrair_segundos_arquivo(file_pzt)
    
    offset_tempo = segundos_pzt - segundos_acl
    print(f"Offset automático calculado: {offset_tempo:.2f} s")
    
    return df1, df2, offset_tempo

# =====================================================
# Execução da Leitura
# =====================================================
df1, df2, OFFSET_TEMPO = carregar_dados_teste(BASE_DIR, TESTE)

t_acl = df1["Time (s)"].to_numpy()
t_pzt = df2["Time (s)"].to_numpy() + OFFSET_TEMPO  # Alinha a escala do PZT com o ACL

x_acl = df1[CANAL_ACL].to_numpy()
x_pzt = df2[CANAL_PZT].to_numpy()

dt_acl = np.mean(np.diff(t_acl))
dt_pzt = np.mean(np.diff(t_pzt))

fs_acl = 1 / dt_acl
fs_pzt = 1 / dt_pzt

print(f"Taxa Amostragem ACL: {fs_acl:.2f} Hz")
print(f"Taxa Amostragem PZT: {fs_pzt:.2f} Hz")

# =====================================================
# Sincronização no Domínio do Tempo (Interpolação)
# =====================================================
t_inicio_comum = max(t_acl[0], t_pzt[0])
t_fim_comum = min(t_acl[-1], t_pzt[-1])

fs_comum = min(fs_acl, fs_pzt)
dt_comum = 1 / fs_comum
t_sync = np.arange(t_inicio_comum, t_fim_comum, dt_comum)

f_acl = interp1d(t_acl, x_acl, kind='linear', fill_value='extrapolate')
f_pzt = interp1d(t_pzt, x_pzt, kind='linear', fill_value='extrapolate')

acl_sync = f_acl(t_sync)
pzt_sync = f_pzt(t_sync)

# Normalização
acl_norm = (acl_sync - np.mean(acl_sync)) / np.std(acl_sync)
pzt_norm = (pzt_sync - np.mean(pzt_sync)) / np.std(pzt_sync)

tempo_plot = t_sync - t_sync[0]

# =====================================================
# Cálculo da FFT
# =====================================================
def calcular_fft(sinal, fs):
    sinal = sinal - np.mean(sinal)
    N = len(sinal)
    
    janela = np.hanning(N)
    sinal_janelado = sinal * janela
    
    fft_vals = np.fft.rfft(sinal_janelado)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    
    magnitude = np.abs(fft_vals) * (2.0 / N)
    return freqs, magnitude

freqs_acl, fft_acl = calcular_fft(acl_norm, fs_comum)
freqs_pzt, fft_pzt = calcular_fft(pzt_norm, fs_comum)

# =====================================================
# Função para encontrar e imprimir o TOP 5 Picos
# =====================================================
def obter_top_picos(freqs, magnitude, top_n=5, min_distancia_hz=0.5):
    # Converte distância em Hz para número de pontos de amostragem
    df = freqs[1] - freqs[0]
    distance_samples = max(1, int(min_distancia_hz / df))
    
    # Encontra picos locais
    picos_idx, _ = find_peaks(magnitude, distance=distance_samples)
    
    # Ordena os picos pela maior amplitude
    picos_ordenados = sorted(picos_idx, key=lambda idx: magnitude[idx], reverse=True)[:top_n]
    
    freqs_top = freqs[picos_ordenados]
    mags_top = magnitude[picos_ordenados]
    
    return freqs_top, mags_top

# =====================================================
# Filtro Passa-Banda (Butterworth)
# =====================================================
def aplicar_filtro_passa_banda(sinal, fs, f_corte_inferior=3, f_corte_superior=200.0, ordem=4):
    nyquist = 0.5 * fs
    low = f_corte_inferior / nyquist
    high = f_corte_superior / nyquist
    
    b, a = butter(ordem, [low, high], btype='band')
    sinal_filtrado = filtfilt(b, a, sinal)
    return sinal_filtrado

# Aplicação do Filtro
acl_filtrado = aplicar_filtro_passa_banda(acl_norm, fs_comum, f_corte_inferior=3.0, f_corte_superior=200.0)
pzt_filtrado = aplicar_filtro_passa_banda(pzt_norm, fs_comum, f_corte_inferior=3.0, f_corte_superior=200.0)

# Recalcular FFTs filtradas
freqs_acl, fft_acl_filt = calcular_fft(acl_filtrado, fs_comum)
freqs_pzt, fft_pzt_filt = calcular_fft(pzt_filtrado, fs_comum)

# Obter Top 5 picos dos sinais filtrados
top_freqs_acl, top_mags_acl = obter_top_picos(freqs_acl, fft_acl_filt, top_n=5)
top_freqs_pzt, top_mags_pzt = obter_top_picos(freqs_pzt, fft_pzt_filt, top_n=5)

# --- IMPRESSÃO NO CONSOLE ---
print("\n---------------------------------------------------")
print(f"TOP 5 FREQUÊNCIAS DOMINANTES - ACELERÔMETRO ({TESTE})")
print("---------------------------------------------------")
for i, (f, m) in enumerate(zip(top_freqs_acl, top_mags_acl), 1):
    rpm = f * 60
    print(f"#{i} Pico: {f:6.2f} Hz  ({rpm:7.1f} RPM)  |  Amplitude: {m:.5f}")

print("\n---------------------------------------------------")
print(f"TOP 5 FREQUÊNCIAS DOMINANTES - PIEZO / PZT ({TESTE})")
print("---------------------------------------------------")
for i, (f, m) in enumerate(zip(top_freqs_pzt, top_mags_pzt), 1):
    rpm = f * 60
    print(f"#{i} Pico: {f:6.2f} Hz  ({rpm:7.1f} RPM)  |  Amplitude: {m:.5f}")
print("---------------------------------------------------\n")

# =====================================================
# PLOT 1: Sinal Temporal Sincronizado
# =====================================================
fig1, ax1 = plt.subplots(figsize=(12, 6), dpi=150)

ax1.plot(tempo_plot, acl_norm, label="Accelerometer", c='green', alpha=0.8)
ax1.plot(tempo_plot, pzt_norm, label="Piezo", c='black', alpha=0.7)

My_axis(ax1,
        xlim=[0, 10],
        ylim=[-15, 15],
        legbox=[0.98, 0.98, 2, 12],
        font=18,
        locator=True,
        setaxis=[f"Signals Synchronized \n",
                 "Time (s)",
                 "Normalized Amplitude"])

plt.tight_layout()
plt.show()

# =====================================================
# PLOT 2: FFT Bruta (Log)
# =====================================================
fig2, ax2 = plt.subplots(figsize=(12, 6), dpi=150)

ax2.plot(freqs_acl, fft_acl, label="Accelerometer", c='green', alpha=0.8)
ax2.plot(freqs_pzt, fft_pzt, label="Piezo", c='black', alpha=0.8)

My_axis(ax2,
        xlim=[0, 6000],
        ylim=[1e-4, 1e2],
        logy=True,
        legbox=[0.98, 0.98, 2, 12],
        font=14,
        locator=True,
        setaxis=[f"FFT Comparison \n",
                 "Frequency (Hz)",
                 "Magnitude (Log)"])

plt.tight_layout()
plt.show()

# =====================================================
# PLOT 3: FFT Filtrada com Marcação dos Top 5 Picos
# =====================================================
fig3, ax3 = plt.subplots(figsize=(12, 6), dpi=150)

ax3.plot(freqs_acl, fft_acl_filt, label="Accelerometer (Filtered)", c='green', alpha=0.8)
ax3.plot(freqs_pzt, fft_pzt_filt, label="Piezo (Filtered)", c='black', alpha=0.8)

# Destaca os 5 picos com marcadores no gráfico
ax3.scatter(top_freqs_acl, top_mags_acl, color='red', zorder=5, s=50, label='Peaks (ACL)')
ax3.scatter(top_freqs_pzt, top_mags_pzt, color='orange', zorder=5, s=50, label='Peaks (PZT)')

My_axis(ax3,
        xlim=[0, 250],
        ylim=[0, max(np.max(fft_acl_filt), np.max(fft_pzt_filt)) * 1.15],
        logy=False,
        legbox=[0.48, 0.98, 2, 12],
        font=18,
        locator=True,
        setaxis=[f"Filtered FFT (3 Hz - 200 Hz Bandpass)\n",
                 "Frequency (Hz)",
                 "Linear Amplitude"])

plt.tight_layout()
plt.show()