# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator

# =====================================================
# Sua Função Customizada de Formatação
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
    
    ax.legend(loc='upper left', bbox_to_anchor=(legbox[0], legbox[1]), 
               fancybox=True, shadow=True, ncol=legbox[2], fontsize=legbox[3])
    
    return ax

# =====================================================
# Dados Extraídos da Tabela LaTeX
# =====================================================
data = [
    # T7
    (7.40, 9.07, 'ACL'), (7.40, 9.90, 'ACL'), (7.40, 7.43, 'ACL'), (7.40, 2.72, 'ACL'), (7.40, 48.97, 'ACL'),
    (7.40, 28.28, 'PZT'), (7.40, 39.59, 'PZT'), (7.40, 19.79, 'PZT'), (7.40, 25.45, 'PZT'), (7.40, 31.11, 'PZT'),
    # T5
    (14.30, 89.11, 'ACL'), (14.30, 9.83, 'ACL'), (14.30, 72.73, 'ACL'), (14.30, 7.38, 'ACL'), (14.30, 9.01, 'ACL'),
    (14.30, 25.49, 'PZT'), (14.30, 46.94, 'PZT'), (14.30, 114.92, 'PZT'), (14.30, 31.16, 'PZT'), (14.30, 19.83, 'PZT'),
    # T3
    (24.00, 130.11, 'ACL'), (24.00, 112.12, 'ACL'), (24.00, 129.29, 'ACL'), (24.00, 112.94, 'ACL'), (24.00, 111.30, 'ACL'),
    (24.00, 87.13, 'PZT'), (24.00, 155.13, 'PZT'), (24.00, 3.28, 'PZT'), (24.00, 25.50, 'PZT'), (24.00, 28.33, 'PZT'),
    # T1
    (42.50, 151.61, 'ACL'), (42.50, 162.09, 'ACL'), (42.50, 154.03, 'ACL'), (42.50, 168.53, 'ACL'), (42.50, 170.94, 'ACL'),
    (42.50, 135.71, 'PZT'), (42.50, 132.87, 'PZT'), (42.50, 127.18, 'PZT'), (42.50, 186.89, 'PZT'), (42.50, 130.02, 'PZT')
]

df = pd.DataFrame(data, columns=['f_VFD', 'f_peak', 'Sensor'])

# Separando por sensor
df_acl = df[df['Sensor'] == 'ACL']
df_pzt = df[df['Sensor'] == 'PZT']

# =====================================================
# Geração do Gráfico Scatter
# =====================================================
fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

# 1. Linhas Teóricas de Referência
REDUCAO_SEW = 4.28
REDUCAO_CAVIDADE = 9.55

f_vfd_cont = np.linspace(5, 45, 200)
ax.plot(f_vfd_cont, f_vfd_cont, 'k--', linewidth=2, alpha=0.6, label=r'1X $f_{\mathrm{VFD}}$')
ax.plot(f_vfd_cont, f_vfd_cont / REDUCAO_SEW, 'r--', linewidth=2, alpha=0.6, label=r'1X $f_{\mathrm{pump}}$ (Shaft)')
ax.plot(f_vfd_cont, f_vfd_cont / REDUCAO_CAVIDADE, 'g:', linewidth=2, alpha=0.8, label=r'1X Cavity')

# 2. Scatter Plots dos Sensores
ax.scatter(df_acl['f_VFD'], df_acl['f_peak'], 
           color='navy', marker='o', s=100, alpha=0.85, edgecolors='black', linewidth=1.2, label='ACL')

ax.scatter(df_pzt['f_VFD'], df_pzt['f_peak'], 
           color='crimson', marker='^', s=110, alpha=0.85, edgecolors='black', linewidth=1.2, label='PZT ')

# 3. Formatação via My_axis
ax = My_axis(ax,
             font=16,
             ticklengthmajor=10, ticklengthminor=5,
             tickwidthmajor=3, tickwidthminor=1.5,
             setaxis=['Dominant Peak Frequencies vs. VFD Operating Speed', 
                      'VFD Frequency $f_{\mathrm{VFD}}$ (Hz)', 
                      'Dominant Peak Frequency $f_{\mathrm{peak}}$ (Hz)'],
             xlim=[4, 46],
             ylim=[0, 200],
             legbox=[0.02, 0.98, 1, 13], # Legenda no canto superior esquerdo
             logx=False, logy=False)

# Re-posicionando a legenda no canto superior esquerdo para não cobrir os dados
ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fancybox=True, shadow=True, fontsize=12)

ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()