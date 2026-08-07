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
filtrar_desde_condicao = _pipeline_io.filtrar_desde_condicao
carregar_grupo = _pipeline_io.carregar_grupo
salvar_grupo = _pipeline_io.salvar_grupo
registrar_log = _pipeline_io.registrar_log
ler_metadados_calibracao = _pipeline_io.ler_metadados_calibracao
buscar_fator_calibracao = _pipeline_io.buscar_fator_calibracao

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


def eh_coluna_tempo(nome_coluna) -> bool:
    """
    Identifica colunas de tempo (ex.: 'Time (s)', 'Tempo', 'T (s)', 'Time')
    para excluí-las da lista de canais de sinal — são só o eixo temporal do
    próprio CSV, não faz sentido gerar série temporal/FFT/heatmap "do tempo".
    """
    nome = str(nome_coluna).strip().lower()
    return bool(re.match(r"^(time|tempo|t)(\s*\(.*\))?$", nome))


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
    parser.add_argument("--from", dest="from_condicao", type=str, default=None,
                         help="Retoma a etapa a partir desta condição, inclusive (ex.: --from T3 processa "
                              "T3, T4, T5... e pula T1/T2). Extrai o número do padrão T<N> no nome da "
                              "condição; nomes fora desse padrão nunca são descartados.")
    parser.add_argument("--fs-acl", type=float, default=30000.0,
                         help="Taxa de amostragem (Hz) dos sensores ACL (padrão: 30000.0).")
    parser.add_argument("--fs-pzt", type=float, default=12500.0,
                         help="Taxa de amostragem (Hz) dos sensores PZT (padrão: 12500.0).")
    parser.add_argument("--fs", type=float, default=None,
                         help="Taxa de amostragem (Hz) para qualquer sensor fora do mapeamento ACL/PZT (fallback).")
    parser.add_argument("--metadados-calibracao", type=str, default=None,
                         help="Caminho de um CSV opcional (sensor,canal,condicao,sensibilidade_mv_por_unidade,"
                              "ganho,unidade_saida) para converter o sinal BRUTO (mV, como já é lido do "
                              "arquivo de origem) em unidade física (ex.: g), ANTES de qualquer outro "
                              "tratamento. Se omitido, procura automaticamente um 'calibracao.csv' na raiz "
                              "de --data_dir. Canais sem entrada no CSV continuam em mV brutos (sem "
                              "conversão) — ver README para o formato e por que isso importa quando há "
                              "sensores/ganhos diferentes entre canais.")
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

    input_dir = raiz_path / "DadosTratados" / "Etapas" / "Leitura"
    grupos = listar_grupos(input_dir)

    if not grupos:
        print(f"❌ Nenhum grupo encontrado em: {input_dir.resolve()}")
        print(" Certifique-se de executar a etapa 01_leitura antes desta.")
        exit(1)

    if args.quick:
        grupos = grupos[:1]
        print("⚡ Modo rápido (--quick): processando apenas o primeiro grupo.\n")

    grupos = filtrar_desde_condicao(grupos, args.from_condicao, indice_condicao=1)
    if args.from_condicao:
        print(f"⏩ Retomando a partir de {args.from_condicao} (--from): {len(grupos)} grupo(s) a processar.\n")
        if not grupos:
            print(f"❌ Nenhum grupo com condição >= {args.from_condicao} encontrado. Nada a fazer.")
            exit(1)

    pasta_figuras_raiz = raiz_path / "DadosTratados" / "Figuras"
    output_dir = raiz_path / "DadosTratados" / "Etapas" / "Preprocessamento"

    # --- Calibração opcional (V bruto -> unidade física) ------------------
    metadados_calibracao = {}
    caminho_calibracao = None
    if args.metadados_calibracao:
        caminho_calibracao = Path(args.metadados_calibracao)
    else:
        candidato = raiz_path / "calibracao.csv"
        if candidato.exists():
            caminho_calibracao = candidato
            print(f"📋 Encontrado calibracao.csv na pasta base, usando automaticamente: {candidato.resolve()}")

    if caminho_calibracao:
        try:
            metadados_calibracao = ler_metadados_calibracao(caminho_calibracao)
            print(f"📋 Calibração carregada: {caminho_calibracao.resolve()} ({len(metadados_calibracao)} entrada(s))")
        except Exception as e:
            print(f"⚠️ Não foi possível ler {caminho_calibracao} ({e}). Seguindo SEM calibração (sinal em Volts brutos).")
            metadados_calibracao = {}
    else:
        print("⚠️ Nenhum calibracao.csv encontrado — sinal será mantido em mV BRUTOS (sem conversão de "
              "sensibilidade/ganho). Comparações de amplitude absoluta entre condições só são válidas se "
              "TODOS os canais usarem o mesmo sensor e o mesmo ganho no condicionador. Ver README.")

    canais_calibrados = set()   # (sensor, canal) que tiveram fator aplicado em pelo menos 1 grupo
    canais_sem_calibracao = set()  # (sensor, canal) que ficaram em Volts brutos em pelo menos 1 grupo

    colunas_metadados = ["sensor", "condicao", "ordem_ensaio", "arquivo_origem"]
    grupos_ok, grupos_com_erro = 0, 0
    pastas_alteradas = {output_dir}

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

        colunas_sinal = [c for c in group_copy.columns if c not in colunas_metadados and not eh_coluna_tempo(c)]

        # Padrão: Figuras/{sensor}/{condicao}/TimeSerie/ (mesmo nível que a
        # pasta FFTs/ gerada na etapa 03, evitando misturar tudo direto em
        # Figuras/{sensor}/{condicao}/).
        pasta_figuras = pasta_figuras_raiz / str(sensor) / str(condicao) / "TimeSerie"
        pasta_figuras.mkdir(parents=True, exist_ok=True)
        pastas_alteradas.add(pasta_figuras)

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

            # Calibração (V bruto -> unidade física), ANTES de qualquer outro
            # tratamento — precisa vir primeiro porque é a etapa que dá
            # significado físico à amplitude que todo o resto do pipeline
            # (FFT, picos, heatmaps) vai comparar entre condições.
            fator_calibracao, unidade_saida = buscar_fator_calibracao(
                metadados_calibracao, sensor, nome_canal_legivel, condicao
            )
            if fator_calibracao is not None:
                sinal_bruto = sinal_bruto * fator_calibracao
                unidade_canal = unidade_saida
                canais_calibrados.add((str(sensor), nome_canal_legivel))
            else:
                unidade_canal = "mV (bruto)"
                canais_sem_calibracao.add((str(sensor), nome_canal_legivel))

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
                    f"Amplitude ({unidade_canal})"
                ]
            )

            nome_figura = f"time_serie_{nome_canal_arquivo}.png"
            caminho_figura = pasta_figuras / nome_figura

            plt.tight_layout()
            plt.savefig(caminho_figura, dpi=150)
            plt.close(fig)

            print(f"      🖼️ Figura salva: Figuras/{sensor}/{condicao}/TimeSerie/{nome_figura}")

        salvar_grupo(group_copy, sensor, condicao, output_dir)
        grupos_ok += 1
        if houve_erro_no_grupo:
            grupos_com_erro += 1

    if canais_calibrados and canais_sem_calibracao:
        print("\n⚠️ ATENÇÃO: alguns canais foram calibrados (unidade física) e outros ficaram em mV "
              "brutos NO MESMO CONJUNTO DE DADOS. Comparações de amplitude absoluta entre esses "
              "canais/condições NÃO são válidas até completar o calibracao.csv para todos eles.")
        print(f"   ✅ Calibrados: {sorted(canais_calibrados)}")
        print(f"   ⚠️ Sem calibração (mV bruto): {sorted(canais_sem_calibracao)}")
    elif canais_calibrados:
        print(f"\n✅ Todos os canais processados foram calibrados: {sorted(canais_calibrados)}")

    caminho_log = registrar_log(raiz_path, "02_preprocessamento", {
        "data_dir": raiz_path.resolve(),
        "fs_acl_hz": args.fs_acl,
        "fs_pzt_hz": args.fs_pzt,
        "fs_fallback_hz": args.fs,
        "tratamento": "remocao_dc_offset + interpolacao_nan_inf (sem filtro de frequencia, sem normalizacao)",
        "quick": args.quick,
        "from_condicao": args.from_condicao,
        "grupos_processados": grupos_ok,
        "grupos_com_aviso": grupos_com_erro,
        "metadados_calibracao": str(caminho_calibracao.resolve()) if caminho_calibracao else None,
        "canais_calibrados": sorted(f"{s}/{c}" for s, c in canais_calibrados),
        "canais_sem_calibracao_mv_bruto": sorted(f"{s}/{c}" for s, c in canais_sem_calibracao),
    }, pastas_alteradas=pastas_alteradas)

    print("\n" + "=" * 65)
    print(f"✅ Etapa 02 (Pré-processamento) Concluída!")
    print(f"   Grupos processados: {grupos_ok} | Grupos com algum aviso: {grupos_com_erro}")
    print(f"💾 Saída salva em: {output_dir.resolve()}")
    print(f"📝 Log de parâmetros: {caminho_log.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()