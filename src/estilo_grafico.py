# -*- coding: utf-8 -*-
"""
estilo_grafico.py
Estilização científica padrão dos gráficos do pipeline (usada por
02_preprocessamento, 03_fft, e demais etapas que vierem a seguir).
"""

from matplotlib.ticker import AutoMinorLocator


def My_axis(ax, font=14,
            ticklengthmajor=10, ticklengthminor=5,
            tickwidthmajor=2, tickwidthminor=1.5,
            setaxis=['', '', ''], xlim=[0, 1], ylim=[-1, 1],
            legbox=[0.98, 0.98, 1, 10], logx=False, logy=False):
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
                    width=tickwidthmajor, length=ticklengthmajor, direction='in', pad=8)
    ax.tick_params(axis='both', which='minor',
                    width=tickwidthminor, length=ticklengthminor, direction='in', pad=8)

    ax.tick_params(axis='x', which='both', top=True, labeltop=False)
    ax.tick_params(axis='y', which='both', right=True, labelright=False)

    ax.set_title(setaxis[0], fontsize=font + 2)
    ax.set_xlabel(setaxis[1], fontsize=font)
    ax.set_ylabel(setaxis[2], fontsize=font)

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc='upper right', bbox_to_anchor=(legbox[0], legbox[1]),
                  fancybox=True, shadow=True, ncol=legbox[2], fontsize=legbox[3])

    return ax