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
├── run_pipeline.py            # Script principal orquestrador
├── requirements.txt           # Dependências do projeto
└── README.md

DadosTratados/                 # Gerada automaticamente na raiz de --data_dir
│
├── Etapas/                    # Todas as saídas .parquet, agrupadas por etapa
│   ├── Leitura/{sensor}/{condicao}.parquet
│   ├── Preprocessamento/{sensor}/{condicao}.parquet
│   └── FFT/{sensor}/{condicao}.parquet
│
├── Figuras/                   # Figuras de análise, por sensor/condição
│   └── {sensor}/{condicao}/
│       ├── TimeSerie/         # gerada na etapa 02
│       └── FFTs/              # gerada na etapa 03
│
└── Logs/
    └── pipeline_log.txt       # log cumulativo (append), 1 entrada por execução:
                                # timestamp, usuário, parâmetros e pastas alteradas