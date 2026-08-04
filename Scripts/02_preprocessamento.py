from __future__ import annotations  # compatibilidade com "np.ndarray | None" em Python < 3.10
import argparse
import importlib.util
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.ticker import AutoMinorLocator


def _carregar_pipeline_io():
    """Carrega src/pipeline_io.py diretamente pelo caminho do arquivo (ver 01_leitura.py)."""
    caminho = Path(__file__).resolve().parent.parent / "src" / "pipeline_io.py"
    spec = importlib.util.spec_from_file_location("pipeline_io", caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


_pipeline_io = _carregar_pipeline_io()
listar_grupos = _pipeline_io.listar_grupos
carregar_grupo = _pipeline_io.carregar_grupo
salvar_grupo = _pipeline_io.salvar_grupo
registrar_log = _pipeline_io.registrar_log


# ==============================================================================
# 🎨 1. FUNÇÃO DE ESTILIZAÇÃO GRÁFICA (Padrão Científico)
# ==============================================================================
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


# ==============================================================================
# 🛠️ 2. MÓDULOS DE TRATAMENTO DO SINAL
# ==============================================================================
def sanitizar_nome(nome: str) -> str:
    """Transforma um nome de coluna (ex.: 'Channel 0') em algo seguro para nome de arquivo."""
    nome = str(nome).strip().lower()
    nome = re.sub(r"[^a-z0-9]+", "_", nome)
    return nome.strip("_") or "canal"


def tratar_nans_e_infs(sinal: np.ndarray, nome_canal: str = "") -> np.ndarray | None:
    """
    Trata NaNs E Infs por interpolação linear antes de qualquer filtragem
    (filtfilt/detrend/plot quebram na presença de NaN ou Inf).
    Retorna None se o canal for inutilizável (tudo inválido, ou vazio).
    """
    if sinal.size == 0:
        return None

    sinal = sinal.astype("float64", copy=True)

    invalido = ~np.isfinite(sinal)  # cobre NaN, +Inf e -Inf de uma vez
    n_invalido = invalido.sum()

    if n_invalido == 0:
        return sinal

    if n_invalido == sinal.size:
        print(f"      ⚠️ Canal {nome_canal}: 100% de valores inválidos (NaN/Inf), pulando.")
        return None

    frac_invalido = n_invalido / sinal.size
    if frac_invalido > 0.5:
        print(f"      ⚠️ Canal {nome_canal}: {frac_invalido:.0%} de valores inválidos (acima de 50%), pulando.")
        return None

    sinal[invalido] = np.nan  # normaliza +-Inf para NaN antes de interpolar
    s = pd.Series(sinal).interpolate(method="linear", limit_direction="both")
    s = s.fillna(0.0)
    print(f"      ℹ️ Canal {nome_canal}: {n_invalido} valor(es) inválido(s) (NaN/Inf) interpolado(s).")
    return s.to_numpy()


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


def pipeline_preprocessamento(sinal: np.ndarray, fs: float = 1000.0, nome_canal: str = "") -> np.ndarray | None:
    """Encadeamento sequencial das etapas de tratamento do sinal."""
    sinal_tratado = tratar_nans_e_infs(sinal, nome_canal=nome_canal)
    if sinal_tratado is None:
        return None
    sinal_tratado = remover_dc_offset(sinal_tratado)
    sinal_tratado = aplicar_filtro_passa_baixa(sinal_tratado, fs=fs, cutoff=200.0)
    sinal_tratado = normalizar_sinal(sinal_tratado, metodo="zscore")

    # Guarda final: se o filtro/normalização reintroduziu algo inválido
    # (ex.: canal instável no Butterworth), não deixa chegar no plot.
    if not np.all(np.isfinite(sinal_tratado)):
        print(f"      ⚠️ Canal {nome_canal}: sinal ficou inválido (NaN/Inf) após filtragem, pulando.")
        return None

    return sinal_tratado


# ==============================================================================
# 🚀 3. EXECUÇÃO PRINCIPAL DO SCRIPT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--quick", action="store_true",
                         help="Processa (e plota) apenas o primeiro grupo sensor/condicao, para teste rápido.")
    parser.add_argument("--fs", type=float, default=1000.0,
                         help="Taxa de amostragem real dos dados, em Hz. Usada no filtro e no eixo de tempo (padrão: 1000.0).")
    args = parser.parse_args()
    fs = args.fs

    raiz_path = Path(args.data_dir)

    input_dir = raiz_path / "DadosTratados" / "Leitura"
    grupos = listar_grupos(input_dir)

    if not grupos:
        print(f"❌ Nenhum grupo encontrado em: {input_dir.resolve()}")
        print(" Certifique-se de executar a etapa 01_leitura antes desta.")
        exit(1)

    if args.quick:
        grupos = grupos[:1]
        print("⚡ Modo rápido (--quick): processando apenas o primeiro grupo.\n")

    pasta_figuras_raiz = raiz_path / "DadosTratados" / "Figuras"
    output_dir = raiz_path / "DadosTratados" / "Preprocessamento"

    colunas_metadados = ["sensor", "condicao", "ordem_ensaio", "arquivo_origem"]
    grupos_ok, grupos_com_erro = 0, 0

    print(f"⚙️ Processando {len(grupos)} grupo(s) sensor/condição...")

    for sensor, condicao, caminho_parquet in grupos:
        print(f"\n📖 Grupo: [{sensor}] | [{condicao}]  ←  {caminho_parquet.name}")
        try:
            group_copy = carregar_grupo(caminho_parquet)
        except Exception as e:
            print(f"   ⚠️ Erro ao carregar grupo: {e}. Pulando.")
            grupos_com_erro += 1
            continue

        colunas_sinal = [c for c in group_copy.columns if c not in colunas_metadados]

        pasta_figuras = pasta_figuras_raiz / str(sensor) / str(condicao)
        pasta_figuras.mkdir(parents=True, exist_ok=True)

        houve_erro_no_grupo = False

        for col_canal in colunas_sinal:
            sinal_bruto = group_copy[col_canal].to_numpy()

            if not np.issubdtype(sinal_bruto.dtype, np.number):
                continue

            # Nome do canal vem sempre da COLUNA REAL do dado, nunca de posição.
            # Assim, se o programa de aquisição mudar a ordem/quantidade de
            # colunas, a figura continua identificando o canal certo.
            nome_canal_legivel = str(col_canal)
            nome_canal_arquivo = sanitizar_nome(col_canal)

            try:
                sinal_tratado = pipeline_preprocessamento(sinal_bruto, fs=fs, nome_canal=nome_canal_legivel)
            except Exception as e:
                print(f"      ⚠️ Erro ao processar canal {nome_canal_legivel}: {e}. Canal pulado.")
                houve_erro_no_grupo = True
                continue

            if sinal_tratado is None:
                houve_erro_no_grupo = True
                continue

            group_copy[f"{col_canal}_tratado"] = sinal_tratado
            tempo_plot = np.arange(len(sinal_tratado)) / fs  # eixo X em segundos

            fig, ax1 = plt.subplots(figsize=(10, 5))
            cor_linha = 'green' if str(sensor).upper() == 'ACL' else 'black'
            ax1.plot(tempo_plot, sinal_tratado, label=f"{sensor} {nome_canal_legivel}", c=cor_linha, alpha=0.85, linewidth=1.2)

            y_lim = max(abs(sinal_tratado.min()), abs(sinal_tratado.max())) * 1.2
            y_lim = max(y_lim, 1.0)

            My_axis(
                ax1,
                font=12,
                xlim=[0, tempo_plot[-1] if len(tempo_plot) else 1],
                ylim=[-y_lim, y_lim],
                legbox=[0.98, 0.98, 1, 9],
                setaxis=[
                    f"Time Series - {sensor} | {condicao} | {nome_canal_legivel}\n",
                    "Time (s)",
                    "Normalized Amplitude"
                ]
            )

            nome_figura = f"time_serie_{nome_canal_arquivo}.png"
            caminho_figura = pasta_figuras / nome_figura

            plt.tight_layout()
            plt.savefig(caminho_figura, dpi=150)
            plt.close(fig)

            print(f"      🖼️ Figura salva: Figuras/{sensor}/{condicao}/{nome_figura}")

        salvar_grupo(group_copy, sensor, condicao, output_dir)
        grupos_ok += 1
        if houve_erro_no_grupo:
            grupos_com_erro += 1

    caminho_log = registrar_log(raiz_path, "02_preprocessamento", {
        "data_dir": raiz_path.resolve(),
        "fs_hz": fs,
        "filtro_lowpass_cutoff_hz": 200.0,
        "filtro_lowpass_ordem": 4,
        "normalizacao": "zscore",
        "quick": args.quick,
        "grupos_processados": grupos_ok,
        "grupos_com_aviso": grupos_com_erro,
    })

    print("\n" + "=" * 65)
    print(f"✅ Etapa 02 (Pré-processamento) Concluída!")
    print(f"   Grupos processados: {grupos_ok} | Grupos com algum aviso: {grupos_com_erro}")
    print(f"💾 Saída salva em: {output_dir.resolve()}")
    print(f"📝 Log de parâmetros: {caminho_log.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()