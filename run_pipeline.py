import os
import sys
import subprocess
from pathlib import Path

def run_step(script_name, input_path):
    """Executa um script de etapa passando o caminho dos dados."""
    script_path = Path("Scripts") / script_name
    if not script_path.exists():
        print(f"❌ Script não encontrado: {script_path}")
        return False

    print(f"\n▶️ Executando: {script_name}...")
    # Executa o script passando o caminho de entrada como argumento
    result = subprocess.run([sys.executable, str(script_path), "--data_dir", str(input_path)])
    
    if result.returncode != 0:
        print(f"⚠️ Erro ao executar {script_name}. Pipeline interrompido.")
        return False
    return True

def main():
    print("=== Wizzard Acell - Pipeline de Análise ===")
    
    # Recebe a pasta alvo por argumento ou via prompt
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = input("Digite o caminho da pasta com os dados a serem analisados: ").strip()

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Pasta informada não existe: {data_path}")
        return

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

    print(f"📁 Pasta selecionada: {data_path.resolve()}")

    for step in pipeline_steps:
        success = run_step(step, data_path)
        if not success:
            break
    else:
        print("\n✅ Pipeline concluído com sucesso!")

if __name__ == "__main__":
    main()