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
    "06_waterfall.py",
    "07_relatorio.py",
]

# Scripts que hoje sabem responder ao teste rápido (aceitam --quick)
STEPS_SUPORTAM_QUICK = {"01_leitura.py", "02_preprocessamento.py", "03_fft.py", "04_picos.py"}

# Scripts que hoje aceitam --fs / --fs-acl / --fs-pzt (taxa de amostragem)
STEPS_SUPORTAM_FS = {"02_preprocessamento.py", "03_fft.py"}

# Scripts que aceitam --f1/--f2 (limites de faixa da FFT)
STEPS_SUPORTAM_FAIXAS_FFT = {"03_fft.py", "04_picos.py"}

# Script que aceita --salvar-figuras (por padrão a etapa 03 não gera mais
# figuras de FFT, já que a 04 gera o mesmo gráfico com os picos marcados)
STEPS_SUPORTAM_FIGURAS_OPCIONAIS = {"03_fft.py"}

# Script que aceita os parâmetros de identificação de picos
STEPS_SUPORTAM_PICOS = {"04_picos.py"}


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
             min_dist_acl: float = None, min_dist_pzt: float = None, min_dist: float = None) -> bool:
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
    if script_name in STEPS_SUPORTAM_PICOS:
        if n_picos is not None:
            cmd.extend(["--n-picos", str(n_picos)])
        if min_dist_acl is not None:
            cmd.extend(["--min-dist-acl", str(min_dist_acl)])
        if min_dist_pzt is not None:
            cmd.extend(["--min-dist-pzt", str(min_dist_pzt)])
        if min_dist is not None:
            cmd.extend(["--min-dist", str(min_dist)])

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
    parser.add_argument("--start", "--from", dest="start", default=None,
                         help="Etapa inicial: número (ex.: 2) ou nome (ex.: 02_preprocessamento).")
    parser.add_argument("--end", "--to", dest="end", default=None,
                         help="Etapa final (inclusive): número ou nome. Padrão: última etapa.")
    parser.add_argument("--quick", action="store_true",
                         help="Propaga --quick para as etapas que suportam teste rápido (só o 1º grupo/pasta).")
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
    args = parser.parse_args()

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
                            min_dist=args.min_dist)
        if not success:
            break
    else:
        print("\n✅ Pipeline concluído com sucesso!")


if __name__ == "__main__":
    main()