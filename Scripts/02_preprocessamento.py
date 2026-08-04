from __future__ import annotations  # compatibilidade com "np.ndarray | None" em Python < 3.10
import argparse
import importlib.util
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def _carregar_modulo(nome: str, arquivo: str):
    """Carrega um módulo de src/ diretamente pelo caminho (ver explicação em 01_leitura.py)."""
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
# 🛠️ 1. MÓDULOS DE TRATAMENTO DO SINAL
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


def pipeline_preprocessamento(sinal: np.ndarray, nome_canal: str = "") -> np.ndarray | None:
    """
    Encadeamento de limpeza do sinal nesta etapa.

    Propositalmente SEM filtro de frequência e SEM normalização de
    amplitude aqui: a filtragem por banda passa a ser responsabilidade
    da etapa 03_fft (que pode olhar múltiplas faixas do mesmo sinal),
    e normalizar (ex.: z-score) destruiria a amplitude física — que é
    justamente o que se quer comparar entre ensaios/condições para
    detectar eventos e ressonâncias.
    """
    sinal_tratado = tratar_nans_e_infs(sinal, nome_canal=nome_canal)
    if sinal_tratado is None:
        return None
    sinal_tratado = remover_dc_offset(sinal_tratado)

    if not np.all(np.isfinite(sinal_tratado)):
        print(f"      ⚠️ Canal {nome_canal}: sinal ficou inválido (NaN/Inf) após limpeza, pulando.")
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
    parser.add_argument("--fs-acl", type=float, default=30000.0,
                         help="Taxa de amostragem (Hz) dos sensores ACL (padrão: 30000.0).")
    parser.add_argument("--fs-pzt", type=float, default=12500.0,
                         help="Taxa de amostragem (Hz) dos sensores PZT (padrão: 12500.0).")
    parser.add_argument("--fs", type=float, default=None,
                         help="Taxa de amostragem (Hz) para qualquer sensor fora do mapeamento ACL/PZT (fallback).")
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
        fs = obter_fs(sensor)
        print(f"\n📖 Grupo: [{sensor}] | [{condicao}]  ←  {caminho_parquet.name}  (fs={fs:.1f} Hz)")
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
                sinal_tratado = pipeline_preprocessamento(sinal_bruto, nome_canal=nome_canal_legivel)
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
            y_lim = max(y_lim, 1e-9)  # evita limite zero em sinal completamente plano

            My_axis(
                ax1,
                font=12,
                xlim=[0, tempo_plot[-1] if len(tempo_plot) else 1],
                ylim=[-y_lim, y_lim],
                legbox=[0.98, 0.98, 1, 9],
                setaxis=[
                    f"Time Series - {sensor} | {condicao} | {nome_canal_legivel}\n",
                    "Time (s)",
                    "Amplitude"
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
        "fs_acl_hz": args.fs_acl,
        "fs_pzt_hz": args.fs_pzt,
        "fs_fallback_hz": args.fs,
        "tratamento": "remocao_dc_offset + interpolacao_nan_inf (sem filtro de frequencia, sem normalizacao)",
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