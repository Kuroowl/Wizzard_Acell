# 🧙‍♂️ Wizzard Acell

Ferramenta de análise e processamento de dados de acelerometria/sinais, com pipeline automatizado para leitura, pré-processamento, análise no domínio da frequência (FFT) e geração de relatórios visuais.

---

## 📁 Estrutura do Projeto

```text
Wizzard_Acell/
│
├── Scripts/
│   ├── 01_leitura.py          # Leitura e parsing dos dados brutos
│   ├── 02_preprocessamento.py # Filtragem e limpeza do sinal
│   ├── 03_fft.py              # Transformada Rápida de Fourier
│   ├── 04_picos.py            # Detecção e identificação de picos
│   ├── 05_heatmap.py          # Geração de mapa de calor espacial/temporal
│   ├── 06_waterfall.py        # Gráfico Waterfall espectral
│   └── 07_relatorio.py        # Consolidação e exportação de relatórios
│
├── outputs/                   # Saídas geradas pelo pipeline (criada automaticamente)
├── run_pipeline.py            # Script principal orquestrador
├── requirements.txt           # Dependências do projeto
└── README.md