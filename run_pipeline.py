import os
import sys
import argparse
import subprocess
from pathlib import Path

PIPELINE_STEPS = [
    "01_leitura.py",
    "02_preprocessamento.py",
    "03_fft.py",
    "04_picos.py",
    "05_heatmap.py",
    "06_mapa_espacial.py",
    "07_waterfall.py",
    "08_histograma.py",
    "09_relatorio.py",
]

# Scripts que hoje sabem responder ao teste rápido (aceitam --quick)
STEPS_SUPORTAM_QUICK = {"01_leitura.py", "02_preprocessamento.py", "03_fft.py", "04_picos.py", "05_heatmap.py", "06_mapa_espacial.py", "07_waterfall.py", "08_histograma.py"}

# Scripts que hoje aceitam --fs / --fs-acl / --fs-pzt (taxa de amostragem)
STEPS_SUPORTAM_FS = {"02_preprocessamento.py", "03_fft.py"}

# Scripts que aceitam --f1/--f2 (limites de faixa da FFT)
STEPS_SUPORTAM_FAIXAS_FFT = {"03_fft.py", "04_picos.py", "05_heatmap.py", "06_mapa_espacial.py"}

# Script que aceita --salvar-figuras (por padrão a etapa 03 não gera mais
# figuras de FFT, já que a 04 gera o mesmo gráfico com os picos marcados)
STEPS_SUPORTAM_FIGURAS_OPCIONAIS = {"03_fft.py"}

# Script que aceita os parâmetros de identificação de picos
STEPS_SUPORTAM_PICOS = {"04_picos.py"}

# Script que aceita os parâmetros do welch (FFT) da etapa 03
STEPS_SUPORTAM_WELCH = {"03_fft.py"}

# Script que aceita os parâmetros do mapa espectral por condição
STEPS_SUPORTAM_HEATMAP = {"05_heatmap.py"}

# Script que aceita os parâmetros do mapa espacial (entre sensores/canais)
STEPS_SUPORTAM_MAPA_ESPACIAL = {"06_mapa_espacial.py"}
STEPS_SUPORTAM_CALIBRACAO = {"02_preprocessamento.py"}

# Script do waterfall 3D (etapa 07) — lê direto de Etapas/FFT, independente
# do heatmap (05)/mapa espacial (06); só precisa da etapa 03 já ter rodado.
STEPS_SUPORTAM_WATERFALL = {"07_waterfall.py"}

# Script do histograma de picos agregados (etapa 08) — lê direto de
# Etapas/Picos (saída da etapa 04), independente de 05/06/07.
STEPS_SUPORTAM_HISTOGRAMA = {"08_histograma.py"}

# --metadados-condicoes vale para qualquer etapa que monte eixo Y por
# condição (heatmap 2D e waterfall 3D usam o mesmo CSV/formato).
STEPS_SUPORTAM_METADADOS_CONDICOES = STEPS_SUPORTAM_HEATMAP | STEPS_SUPORTAM_WATERFALL

# --escala/--cmap/--freq-max/--freq-resolucao/--db-min: parâmetros de
# escala espectral compartilhados por heatmap, mapa espacial e waterfall.
# --sem-picos fica de fora (só existe overlay de picos no heatmap 2D/mapa
# espacial; no waterfall 3D ainda não há esse overlay).
STEPS_SUPORTAM_ESCALA_MAPA = STEPS_SUPORTAM_HEATMAP | STEPS_SUPORTAM_MAPA_ESPACIAL
STEPS_SUPORTAM_ESCALA_ESPECTRAL = STEPS_SUPORTAM_ESCALA_MAPA | STEPS_SUPORTAM_WATERFALL


def selecionar_pasta_windows() -> str:
    """Abre uma caixa de diálogo nativa do Windows para escolher a pasta."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    print("📂 Abrindo seletor de pasta do Windows...")
    caminho_selecionado = filedialog.askdirectory(
        title="Selecione a pasta raiz dos dados a serem analisados"
    )
    root.destroy()
    return caminho_selecionado


def resolver_indice_etapa(valor: str) -> int:
    """
    Aceita tanto o número da etapa (1, 2, 3...) quanto o nome do arquivo
    (ex.: '02_preprocessamento.py' ou apenas '02_preprocessamento').
    Retorna o índice (0-based) dentro de PIPELINE_STEPS.
    """
    valor = valor.strip()

    if valor.isdigit():
        idx = int(valor) - 1
        if 0 <= idx < len(PIPELINE_STEPS):
            return idx
        raise ValueError(f"Número de etapa fora do intervalo (1-{len(PIPELINE_STEPS)}): {valor}")

    nome = valor if valor.endswith(".py") else f"{valor}.py"
    for i, step in enumerate(PIPELINE_STEPS):
        if step == nome or step.startswith(valor):
            return i

    raise ValueError(f"Etapa não reconhecida: '{valor}'. Opções: {PIPELINE_STEPS}")


def run_step(script_name: str, input_path: Path, quick: bool = False, fs: float = None,
             fs_acl: float = None, fs_pzt: float = None, f1: float = None, f2: float = None,
             salvar_figuras: bool = False, n_picos: int = None,
             min_dist_acl: float = None, min_dist_pzt: float = None, min_dist: float = None,
             nperseg: int = None, noverlap: int = None, janela: str = None,
             metadados_condicoes: str = None, metadados_canais: str = None,
             metadados_calibracao: str = None,
             escala: str = None, cmap: str = None,
             freq_max: float = None, freq_resolucao: float = None, db_min: float = None,
             sem_picos: bool = False, from_condicao: str = None, ler_todos: bool = False,
             grupo_alvo: str = None, elev: float = None, azim: float = None,
             n_bins: int = None, peso_amplitude: bool = False, freq_min: float = None) -> bool:
    """Executa um script de etapa passando o caminho dos dados."""
    script_path = Path("Scripts") / script_name
    if not script_path.exists():
        print(f"❌ Script não encontrado: {script_path}")
        return False

    cmd = [sys.executable, str(script_path), "--data_dir", str(input_path)]
    if quick:
        if script_name in STEPS_SUPORTAM_QUICK:
            cmd.append("--quick")
        else:
            print(f"   ℹ️ {script_name} ainda não implementa --quick; rodando normalmente.")
    if from_condicao:
        # Todas as etapas (01-08) já suportam --from.
        cmd.extend(["--from", from_condicao])
    if ler_todos:
        if script_name == "01_leitura.py":
            cmd.append("--all")
        else:
            print(f"   ℹ️ --all é específico da etapa 01_leitura.py; ignorado em {script_name}.")
    if grupo_alvo:
        if script_name == "01_leitura.py":
            cmd.extend(["--grupo", grupo_alvo])
        else:
            print(f"   ℹ️ --grupo é específico da etapa 01_leitura.py; ignorado em {script_name}.")
    if script_name in STEPS_SUPORTAM_FS:
        if fs is not None:
            cmd.extend(["--fs", str(fs)])
        if fs_acl is not None:
            cmd.extend(["--fs-acl", str(fs_acl)])
        if fs_pzt is not None:
            cmd.extend(["--fs-pzt", str(fs_pzt)])
    elif any(v is not None for v in (fs, fs_acl, fs_pzt)):
        print(f"   ℹ️ {script_name} ainda não implementa parâmetros de fs; ignorando.")
    if script_name in STEPS_SUPORTAM_FAIXAS_FFT:
        if f1 is not None:
            cmd.extend(["--f1", str(f1)])
        if f2 is not None:
            cmd.extend(["--f2", str(f2)])
    if script_name in STEPS_SUPORTAM_FIGURAS_OPCIONAIS and salvar_figuras:
        cmd.append("--salvar-figuras")
    if script_name in STEPS_SUPORTAM_WELCH:
        if nperseg is not None:
            cmd.extend(["--nperseg", str(nperseg)])
        if noverlap is not None:
            cmd.extend(["--noverlap", str(noverlap)])
        if janela is not None:
            cmd.extend(["--janela", janela])
    if script_name in STEPS_SUPORTAM_PICOS:
        if n_picos is not None:
            cmd.extend(["--n-picos", str(n_picos)])
        if min_dist_acl is not None:
            cmd.extend(["--min-dist-acl", str(min_dist_acl)])
        if min_dist_pzt is not None:
            cmd.extend(["--min-dist-pzt", str(min_dist_pzt)])
        if min_dist is not None:
            cmd.extend(["--min-dist", str(min_dist)])
    if script_name in STEPS_SUPORTAM_METADADOS_CONDICOES and metadados_condicoes is not None:
        cmd.extend(["--metadados-condicoes", metadados_condicoes])
    if script_name in STEPS_SUPORTAM_MAPA_ESPACIAL and metadados_canais is not None:
        cmd.extend(["--metadados-canais", metadados_canais])
    if script_name in STEPS_SUPORTAM_CALIBRACAO and metadados_calibracao is not None:
        cmd.extend(["--metadados-calibracao", metadados_calibracao])
    if script_name in STEPS_SUPORTAM_ESCALA_ESPECTRAL:
        if escala is not None:
            cmd.extend(["--escala", escala])
        if cmap is not None:
            cmd.extend(["--cmap", cmap])
        if freq_max is not None:
            cmd.extend(["--freq-max", str(freq_max)])
        if freq_resolucao is not None:
            cmd.extend(["--freq-resolucao", str(freq_resolucao)])
        if db_min is not None:
            cmd.extend(["--db-min", str(db_min)])
        if sem_picos and script_name in STEPS_SUPORTAM_ESCALA_MAPA:
            cmd.append("--sem-picos")
    if script_name in STEPS_SUPORTAM_WATERFALL:
        if elev is not None:
            cmd.extend(["--elev", str(elev)])
        if azim is not None:
            cmd.extend(["--azim", str(azim)])
    if script_name in STEPS_SUPORTAM_HISTOGRAMA:
        if n_bins is not None:
            cmd.extend(["--n-bins", str(n_bins)])
        if freq_min is not None:
            cmd.extend(["--freq-min", str(freq_min)])
        if freq_max is not None:
            cmd.extend(["--freq-max", str(freq_max)])
        if peso_amplitude:
            cmd.append("--peso-amplitude")

    print(f"\n▶️ Executando: {script_name}...")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"⚠️ Erro ao executar {script_name}. Pipeline interrompido.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Wizzard Acell - Pipeline de Análise")
    parser.add_argument("data_dir", nargs="?", default=None,
                         help="Pasta raiz dos dados. Se omitida, abre o seletor de pasta.")
    parser.add_argument("--start", dest="start", default=None,
                         help="Etapa inicial: número (ex.: 2) ou nome (ex.: 02_preprocessamento).")
    parser.add_argument("--end", "--to", dest="end", default=None,
                         help="Etapa final (inclusive): número ou nome. Padrão: última etapa.")
    parser.add_argument("--quick", action="store_true",
                         help="Propaga --quick para as etapas que suportam teste rápido (só o 1º grupo/pasta).")
    parser.add_argument("--from", dest="from_condicao", type=str, default=None,
                         help="Retoma o pipeline a partir desta condição, inclusive (ex.: --start 02 --from T3 "
                              "processa T3, T4, T5... e pula T1/T2). Propagado para todas as etapas (01-06).")
    parser.add_argument("--all", dest="ler_todos", action="store_true",
                         help="Propagado apenas para 01_leitura.py: quando uma condição T<N> tem subgrupos "
                              "G1/G2/G3 (blocos sequenciais no tempo), concatena todos os arquivos de todos "
                              "os subgrupos em ordem sequencial. Padrão: lê só o primeiro arquivo de cada T<N>.")
    parser.add_argument("--grupo", dest="grupo_alvo", type=str, default=None,
                         help="Propagado apenas para 01_leitura.py: concatena só os arquivos do subgrupo "
                              "indicado (ex.: --grupo G1). Mutuamente exclusivo com --all.")
    parser.add_argument("--fs", type=float, default=None,
                         help="Taxa de amostragem (Hz) de fallback, propagada para as etapas que usam esse parâmetro.")
    parser.add_argument("--fs-acl", type=float, default=None,
                         help="Taxa de amostragem (Hz) dos sensores ACL (padrão do script: 30000.0).")
    parser.add_argument("--fs-pzt", type=float, default=None,
                         help="Taxa de amostragem (Hz) dos sensores PZT (padrão do script: 12500.0).")
    parser.add_argument("--f1", type=float, default=None,
                         help="Limite entre faixa baixa e média da FFT, em Hz (padrão do script: 15.0).")
    parser.add_argument("--f2", type=float, default=None,
                         help="Limite entre faixa média e alta da FFT, em Hz (padrão do script: 400.0).")
    parser.add_argument("--salvar-figuras-fft", action="store_true",
                         help="Gera as figuras de FFT por faixa na etapa 03 (desligado por padrão; "
                              "a etapa 04 já gera o mesmo gráfico com os picos marcados).")
    parser.add_argument("--n-picos", type=int, default=None,
                         help="Número de picos a identificar por canal na etapa 04 (padrão do script: 5).")
    parser.add_argument("--min-dist-acl", type=float, default=None,
                         help="Distância mínima (Hz) entre picos para sensores ACL (padrão do script: 2.0).")
    parser.add_argument("--min-dist-pzt", type=float, default=None,
                         help="Distância mínima (Hz) entre picos para sensores PZT (padrão do script: 5.0).")
    parser.add_argument("--min-dist", type=float, default=None,
                         help="Distância mínima (Hz) de fallback entre picos para outros sensores (padrão do script: 2.0).")
    parser.add_argument("--nperseg", type=int, default=None,
                         help="Tamanho do segmento (nperseg) do scipy.signal.welch na etapa 03, em amostras "
                              "(padrão do script: 8192).")
    parser.add_argument("--noverlap", type=int, default=None,
                         help="Sobreposição entre segmentos (noverlap) do welch na etapa 03, em amostras "
                              "(padrão do script: metade do nperseg).")
    parser.add_argument("--janela", type=str, default=None,
                         help="Janela usada pelo welch na etapa 03 (padrão do script: hann).")
    parser.add_argument("--metadados-condicoes", type=str, default=None,
                         help="CSV opcional (condicao,f_vfd_hz,vazao_m3h,reducao_shaft,reducao_cavidade) "
                              "para a etapa 05 usar eixo Y contínuo + linhas teóricas no heatmap.")
    parser.add_argument("--metadados-canais", type=str, default=None,
                         help="CSV opcional (sensor,canal,posicao_m,rotulo) para a etapa 06 usar posição "
                              "física no eixo Y do mapa espacial.")
    parser.add_argument("--metadados-calibracao", type=str, default=None,
                         help="CSV opcional (sensor,canal,condicao,sensibilidade_mv_por_unidade,ganho,"
                              "unidade_saida) para a etapa 02 converter o sinal bruto (mV) em unidade "
                              "física (ex.: g) antes de qualquer outro tratamento. Se omitido, procura "
                              "automaticamente um 'calibracao.csv' na raiz de --data_dir.")
    parser.add_argument("--escala", type=str, default=None,
                         choices=["db-global", "abs-global", "abs-condicao", "pico-canal", "rms-canal", "db"],
                         help="Escala de cor/eixo Z compartilhada pelas etapas 05 (heatmap), 06 (mapa "
                              "espacial) e 07 (waterfall) (padrão do script: db-global).")
    parser.add_argument("--db-min", type=float, default=None,
                         help="Piso (dB) quando --escala db ou db-global (padrão do script: -40.0).")
    parser.add_argument("--cmap", type=str, default=None,
                         help="Colormap do matplotlib para as etapas 05/06/07 (padrão de cada script: "
                              "viridis no heatmap/mapa espacial, jet no waterfall).")
    parser.add_argument("--freq-max", type=float, default=None,
                         help="Frequência máxima (Hz) nas etapas 05/06/07 (padrão do script: automático) "
                              "e limite superior opcional do histograma da etapa 08.")
    parser.add_argument("--freq-min", type=float, default=None,
                         help="Limite inferior opcional (Hz) do histograma da etapa 08. Padrão do script: "
                              "sem recorte (usa todo o intervalo dos picos encontrados).")
    parser.add_argument("--freq-resolucao", type=float, default=None,
                         help="Resolução (Hz) do grid de frequência nas etapas 05/06/07 (padrão do script: 0.5).")
    parser.add_argument("--sem-picos", action="store_true",
                         help="Não sobrepõe os picos (Etapas/Picos) no heatmap da etapa 05/06 (a etapa 07 "
                              "ainda não tem esse overlay).")
    parser.add_argument("--elev", type=float, default=None,
                         help="Elevação (graus) da câmera 3D do waterfall (etapa 07; padrão do script: 25.0).")
    parser.add_argument("--azim", type=float, default=None,
                         help="Azimute (graus) da câmera 3D do waterfall (etapa 07; padrão do script: -60.0).")
    parser.add_argument("--n-bins", type=int, default=None,
                         help="Número de bins do histograma de picos agregados da etapa 08 (padrão do script: 60).")
    parser.add_argument("--peso-amplitude", action="store_true",
                         help="Etapa 08: histograma por amplitude somada em vez de contagem de picos.")
    args = parser.parse_args()

    if args.ler_todos and args.grupo_alvo:
        print("❌ --all e --grupo são mutuamente exclusivos. Use um ou outro.")
        return

    print("=== Wizzard Acell - Pipeline de Análise ===")

    data_dir = args.data_dir or selecionar_pasta_windows()
    if not data_dir:
        print("❌ Nenhuma pasta foi selecionada. Pipeline cancelado.")
        return

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Pasta informada não existe: {data_path}")
        return

    print(f"📁 Pasta selecionada: {data_path.resolve()}")

    try:
        idx_inicio = resolver_indice_etapa(args.start) if args.start else 0
        idx_fim = resolver_indice_etapa(args.end) if args.end else len(PIPELINE_STEPS) - 1
    except ValueError as e:
        print(f"❌ {e}")
        return

    if idx_inicio > idx_fim:
        print("❌ A etapa inicial (--start) não pode vir depois da etapa final (--end).")
        return

    etapas_a_rodar = PIPELINE_STEPS[idx_inicio:idx_fim + 1]
    print(f"🧭 Etapas a executar: {etapas_a_rodar}")
    if args.quick:
        print("⚡ Modo rápido ativado (--quick).")

    for step in etapas_a_rodar:
        success = run_step(step, data_path, quick=args.quick, fs=args.fs,
                            fs_acl=args.fs_acl, fs_pzt=args.fs_pzt, f1=args.f1, f2=args.f2,
                            salvar_figuras=args.salvar_figuras_fft, n_picos=args.n_picos,
                            min_dist_acl=args.min_dist_acl, min_dist_pzt=args.min_dist_pzt,
                            min_dist=args.min_dist, nperseg=args.nperseg, noverlap=args.noverlap,
                            janela=args.janela, metadados_condicoes=args.metadados_condicoes,
                            metadados_canais=args.metadados_canais,
                            metadados_calibracao=args.metadados_calibracao,
                            escala=args.escala, cmap=args.cmap, freq_max=args.freq_max,
                            freq_resolucao=args.freq_resolucao, db_min=args.db_min,
                            sem_picos=args.sem_picos, from_condicao=args.from_condicao,
                            ler_todos=args.ler_todos, grupo_alvo=args.grupo_alvo,
                            elev=args.elev, azim=args.azim,
                            n_bins=args.n_bins, peso_amplitude=args.peso_amplitude, freq_min=args.freq_min)
        if not success:
            break
    else:
        print("\n✅ Pipeline concluído com sucesso!")


if __name__ == "__main__":
    main()