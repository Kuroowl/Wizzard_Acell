import os
import sys
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

def selecionar_pasta_windows() -> str:
    """Abre uma caixa de diálogo nativa do Windows para escolher a pasta."""
    root = tk.Tk()
    root.withdraw()  # Esconde a janela principal do Tkinter
    root.attributes('-topmost', True)  # Traz a janela de seleção para a frente

    print("📂 Abrindo seletor de pasta do Windows...")
    caminho_selecionado = filedialog.askdirectory(
        title="Selecione a pasta raiz dos dados a serem analisados"
    )
    
    root.destroy()  # Destroi a instância do Tkinter após a escolha
    return caminho_selecionado

def run_step(script_name, input_path):
    """Executa um script de etapa passando o caminho dos dados."""
    script_path = Path("Scripts") / script_name
    if not script_path.exists():
        print(f"❌ Script não encontrado: {script_path}")
        return False

    print(f"\n▶️ Executando: {script_name}...")
    result = subprocess.run([sys.executable, str(script_path), "--data_dir", str(input_path)])
    
    if result.returncode != 0:
        print(f"⚠️ Erro ao executar {script_name}. Pipeline interrompido.")
        return False
    return True

def main():
    print("=== Wizzard Acell - Pipeline de Análise ===")
    
    # 1. Se um caminho for passado por argumento (terminal/VS Code), usa ele
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        # 2. Caso contrário, abre a janela de seleção de pasta
        data_dir = selecionar_pasta_windows()

    # Validação do caminho
    if not data_dir:
        print("❌ Nenhuma pasta foi selecionada. Pipeline cancelado.")
        return

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Pasta informada não existe: {data_path}")
        return

    print(f"📁 Pasta selecionada: {data_path.resolve()}")

    # Sequência de execução das etapas 
    pipeline_steps = [
        "01_leitura.py",
        "02_preprocessamento.py",
        "03_fft.py",
        "04_picos.py",
        "05_heatmap.py",
        "06_waterfall.py",
        "07_relatorio.py"
    ]

    for step in pipeline_steps:
        success = run_step(step, data_path)
        if not success:
            break
    else:
        print("\n✅ Pipeline concluído com sucesso!")

if __name__ == "__main__":
    main()