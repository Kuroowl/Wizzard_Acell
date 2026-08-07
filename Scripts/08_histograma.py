from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _carregar_modulo(nome: str, arquivo: str):
    """Carrega um módulo de src/ diretamente pelo caminho (evita import via pacote 'src', ver 01_leitura.py)."""
    caminho = Path(__file__).resolve().parent.parent / "src" / arquivo
    spec = importlib.util.spec_from_file_location(nome, caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


_pipeline_io = _carregar_modulo("pipeline_io", "pipeline_io.py")
listar_grupos = _pipeline_io.listar_grupos
salvar_grupo = _pipeline_io.salvar_grupo
registrar_log = _pipeline_io.registrar_log
filtrar_desde_condicao = _pipeline_io.filtrar_desde_condicao

_estilo = _carregar_modulo("estilo_grafico", "estilo_grafico.py")
My_axis = _estilo.My_axis

_espectro = _carregar_modulo("espectro", "espectro.py")
sanitizar_nome = _espectro.sanitizar_nome

# "global" é o nome salvo pela etapa 04 (busca no espectro inteiro, sem
# recorte de faixa); aqui chamamos de "full" por figura/rótulo, pra ficar
# consistente com o nome já usado na etapa 07 (waterfall) pro mesmo conceito
# ("tudo considerado").
ESCOPO_PARA_ROTULO_FIGURA = {"low": "low", "mid": "mid", "high": "high", "global": "full"}

COR_HISTOGRAMA = {"low": "#2979FF", "mid": "#00C853", "high": "#FF1744", "global": "#7C4DFF"}

METODO_DESCRICAO = (
    "Agrega (pool) os picos JÁ IDENTIFICADOS pela etapa 04 (scipy.signal.find_peaks, "
    "Etapas/Picos) de TODAS as condições de um sensor/canal num histograma só, por "
    "faixa (low/mid/high/global). Não recalcula picos nem FFT — reaproveita o que já "
    "foi salvo. Um pico que aparece sempre na mesma faixa de Hz, em toda condição, "
    "empilha no mesmo bin (barra alta e estreita) — assinatura de algo fixo na "
    "máquina (ressonância estrutural, defeito de rolamento, folga mecânica), "
    "independente do ponto de operação. Um pico que se desloca com a condição "
    "(ex.: a própria frequência do VFD) cai em bins diferentes a cada condição e "
    "se espalha no histograma agregado — assinatura de algo ligado ao ponto de "
    "operação (hidráulico/VFD), não à máquina em si."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--quick", action="store_true",
                         help="Processa apenas o primeiro sensor encontrado, para teste rápido.")
    parser.add_argument("--from", dest="from_condicao", type=str, default=None,
                         help="Inclui no agregado só as condições a partir desta, inclusive (ex.: "
                              "--from T3 usa T3, T4, T5... e ignora T1/T2). Extrai o número do padrão "
                              "T<N> no nome da condição; nomes fora desse padrão nunca são descartados.")
    parser.add_argument("--n-bins", type=int, default=60,
                         help="Número de bins do histograma, POR FIGURA (cada faixa tem sua própria "
                              "largura de bin, calculada a partir do próprio intervalo de dados — "
                              "assim low/mid/high/full ficam todos com resolução visual comparável, "
                              "mesmo tendo larguras de faixa muito diferentes). Padrão: 60.")
    parser.add_argument("--peso-amplitude", action="store_true",
                         help="Em vez de contar quantas vezes um pico caiu em cada bin, soma a "
                              "amplitude dos picos daquele bin. Realça bins com picos fortes mesmo "
                              "que raros; por padrão (sem esta flag) o histograma é por CONTAGEM "
                              "(quantas condições tiveram um pico ali), igual ao protótipo original.")
    args = parser.parse_args()

    raiz_path = Path(args.data_dir)

    # Lê direto de Etapas/Picos (saída da etapa 04) — não recalcula picos
    # nem depende de nenhuma etapa depois dela (05/06/07).
    input_dir = raiz_path / "DadosTratados" / "Etapas" / "Picos"
    grupos = listar_grupos(input_dir)

    if not grupos:
        print(f"❌ Nenhum grupo encontrado em: {input_dir.resolve()}")
        print(" Certifique-se de executar a etapa 04_picos antes desta (não precisa rodar 05/06/07).")
        exit(1)

    sensores = {}
    for sensor, condicao, caminho_parquet in grupos:
        sensores.setdefault(sensor, {})[condicao] = caminho_parquet

    lista_sensores = list(sensores.keys())
    if args.quick:
        lista_sensores = lista_sensores[:1]
        print("⚡ Modo rápido (--quick): processando apenas o primeiro sensor.\n")

    pasta_figuras_raiz = raiz_path / "DadosTratados" / "Figuras"
    output_dir = raiz_path / "DadosTratados" / "Etapas" / "Histograma"
    pastas_alteradas = {output_dir}
    sensores_ok, sensores_com_erro = 0, 0

    print(f"⚙️ Agregando histograma de picos para {len(lista_sensores)} tipo(s) de sensor...")
    print(f"   Método: {METODO_DESCRICAO}\n")

    for sensor in lista_sensores:
        condicoes_disponiveis = sorted(sensores[sensor].keys())
        condicoes_disponiveis = filtrar_desde_condicao(condicoes_disponiveis, args.from_condicao)
        if args.from_condicao and not condicoes_disponiveis:
            print(f"   ℹ️ Sensor [{sensor}]: nenhuma condição >= {args.from_condicao}, pulando sensor.")
            continue
        print(f"\n📖 Sensor: [{sensor}]  ←  agregando {len(condicoes_disponiveis)} condição(ões): {condicoes_disponiveis}")

        # --- carrega e empilha os picos de TODAS as condições desse sensor ---
        partes = []
        for condicao in condicoes_disponiveis:
            try:
                df_picos = pd.read_parquet(sensores[sensor][condicao])
            except Exception as e:
                print(f"   ⚠️ Erro ao carregar {sensor}/{condicao}: {e}. Condição pulada.")
                continue
            colunas_esperadas = {"canal", "escopo", "freq_hz", "amplitude"}
            if not colunas_esperadas.issubset(df_picos.columns):
                print(f"   ⚠️ {sensor}/{condicao}: parquet sem as colunas esperadas {colunas_esperadas}. Pulado.")
                continue
            df_picos = df_picos.copy()
            df_picos["condicao"] = condicao
            partes.append(df_picos)

        if not partes:
            print(f"   ⚠️ Nenhum dado de pico válido para o sensor [{sensor}]. Pulando sensor.")
            sensores_com_erro += 1
            continue

        df_todos = pd.concat(partes, ignore_index=True)

        pasta_figuras = pasta_figuras_raiz / str(sensor) / "Histograma"
        pasta_figuras.mkdir(parents=True, exist_ok=True)
        pastas_alteradas.add(pasta_figuras)

        houve_erro_no_sensor = False

        for canal, df_canal in df_todos.groupby("canal"):
            nome_canal_arquivo = sanitizar_nome(str(canal))
            linhas_saida = []

            for escopo, df_escopo in df_canal.groupby("escopo"):
                rotulo_figura = ESCOPO_PARA_ROTULO_FIGURA.get(str(escopo), str(escopo))
                freqs = df_escopo["freq_hz"].to_numpy()
                amplitudes = df_escopo["amplitude"].to_numpy()
                n_condicoes_com_pico = df_escopo["condicao"].nunique()

                if freqs.size == 0:
                    continue

                f_min, f_max = float(freqs.min()), float(freqs.max())
                if f_max <= f_min:
                    f_max = f_min + 1.0  # evita bins degenerados quando só há 1 valor único

                bordas = np.linspace(f_min, f_max, args.n_bins + 1)
                contagem, _ = np.histogram(freqs, bins=bordas)
                soma_amplitude, _ = np.histogram(freqs, bins=bordas, weights=amplitudes)
                centros = (bordas[:-1] + bordas[1:]) / 2.0

                valores_plot = soma_amplitude if args.peso_amplitude else contagem
                y_label = "Amplitude somada (picos no bin)" if args.peso_amplitude else "Contagem de picos (condições)"

                for i in range(len(centros)):
                    linhas_saida.append({
                        "escopo": str(escopo),
                        "bin_centro_hz": float(centros[i]),
                        "bin_min_hz": float(bordas[i]),
                        "bin_max_hz": float(bordas[i + 1]),
                        "contagem": int(contagem[i]),
                        "amplitude_somada": float(soma_amplitude[i]),
                    })

                fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
                largura_bin = bordas[1] - bordas[0]
                ax.bar(centros, valores_plot, width=largura_bin * 0.95,
                       color=COR_HISTOGRAMA.get(str(escopo), "#607D8B"),
                       edgecolor="black", alpha=0.75)

                ax.grid(True, axis="y", which="major", linestyle="--", alpha=0.6, color="gray", zorder=0)
                ax.set_axisbelow(True)

                My_axis(
                    ax, font=12,
                    xlim=[f_min, f_max],
                    ylim=[0, max(float(valores_plot.max()) * 1.15, 1e-9)],
                    setaxis=[
                        f"Histogram of Peaks ({rotulo_figura}) - {sensor} | {canal} | "
                        f"{f_min:.0f}-{f_max:.0f} Hz | {n_condicoes_com_pico}/{len(condicoes_disponiveis)} condições\n",
                        "Frequency (Hz)",
                        y_label,
                    ],
                )

                nome_figura = f"histograma_{nome_canal_arquivo}_{rotulo_figura}_{f_min:.0f}-{f_max:.0f}hz.png"
                caminho_figura = pasta_figuras / nome_figura
                plt.savefig(caminho_figura, dpi=150, bbox_inches="tight")
                plt.close(fig)

                print(f"      🖼️ Figura salva: Figuras/{sensor}/Histograma/{nome_figura}")

            if linhas_saida:
                df_saida = pd.DataFrame(linhas_saida)
                salvar_grupo(df_saida, sensor, nome_canal_arquivo, output_dir)
                print(f"      💾 Histograma salvo: Etapas/Histograma/{sensor}/{nome_canal_arquivo}.parquet")
            else:
                houve_erro_no_sensor = True

        sensores_ok += 1
        if houve_erro_no_sensor:
            sensores_com_erro += 1

    caminho_log = registrar_log(raiz_path, "08_histograma", {
        "data_dir": raiz_path.resolve(),
        "metodo": METODO_DESCRICAO,
        "n_bins": args.n_bins,
        "peso_amplitude": args.peso_amplitude,
        "quick": args.quick,
        "from_condicao": args.from_condicao,
        "tipos_de_sensor_processados": sensores_ok,
        "tipos_de_sensor_com_aviso": sensores_com_erro,
    }, pastas_alteradas=pastas_alteradas)

    print("\n" + "=" * 65)
    print(f"✅ Etapa 08 (Histograma) Concluída!")
    print(f"   Tipos de sensor processados: {sensores_ok} | Tipos de sensor com algum aviso: {sensores_com_erro}")
    print(f"💾 Figuras/dados salvos em: {pasta_figuras_raiz.resolve()} / {output_dir.resolve()}")
    print(f"📝 Log de parâmetros: {caminho_log.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
