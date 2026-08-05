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
│   ├── 05_heatmap.py          # Mapa espectral por condição (freq x T1..Tn), por sensor
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
│   ├── FFT/{sensor}/{condicao}.parquet             # freq_hz + amplitude por canal
│   ├── Picos/{sensor}/{condicao}.parquet           # canal, escopo (low/mid/high/global), ordem_pico, freq_hz, amplitude
│   └── Heatmap/{sensor}.parquet                    # planejado (etapa 05): matriz condicao x freq_hz consolidada
│
├── Figuras/                   # Figuras de análise, por sensor/condição
│   └── {sensor}/{condicao}/
│       ├── TimeSerie/         # gerada na etapa 02
│       ├── FFTs/              # etapa 03, OPCIONAL (--salvar-figuras, desligado por padrão)
│       └── Picos/             # etapa 04 — mesmo gráfico da FFT + marcador vermelho nos picos
│
└── Logs/
    └── pipeline_log.txt       # log cumulativo (append), 1 entrada por execução:
                                # timestamp, usuário, parâmetros e pastas alteradas
```

---

## 🗺️ Etapa 05 (planejada) — Mapa espectral por condição

Gera, por sensor, um heatmap com eixo X = frequência (Hz), eixo Y = condição
de ensaio (T1...Tn), cor = amplitude — consolidando as saídas de
`Etapas/FFT` e `Etapas/Picos` de todas as condições num único gráfico por
sensor.

**Eixo Y por padrão**: rótulo categórico da própria condição (`T1`, `T2`,
..., `Tn`), na ordem em que aparecem — sem exigir nenhuma informação além do
que já foi gerado pelas etapas 01-04. Não são desenhadas linhas teóricas
(1X motor, 1X eixo, 1X/2X/4X cavidade), pois isso depende de saber a
frequência real do inversor e as razões de redução do equipamento em cada
condição, que não fazem sentido genérico pra um sensor qualquer distribuído
na tubulação.

**Metadados por condição (opcional)** — `--metadados-condicoes caminho.csv`:
quando informado, o eixo Y passa a ser a frequência real do inversor
(contínuo, em Hz) em vez do rótulo categórico, e — se as colunas de redução
também estiverem preenchidas — as linhas teóricas voltam a ser desenhadas
sobre o mapa.

Formato esperado do CSV (cabeçalho obrigatório, 1 linha por condição):

| coluna            | obrigatória? | descrição                                                                 |
|-------------------|:---:|-----------------------------------------------------------------------------------|
| `condicao`        | sim | Nome da condição, deve bater exatamente com a pasta/rótulo usado nas demais etapas (ex.: `T1`). |
| `f_vfd_hz`        | não | Frequência do inversor (Hz) nessa condição. Se vazio, essa condição usa o rótulo categórico no eixo Y mesmo com o CSV presente. |
| `vazao_m3h`       | não | Vazão (m³/h) nessa condição. Informativo (aparece em legendas/relatório); não afeta os eixos do heatmap. |
| `reducao_shaft`   | não | Razão de redução do motorredutor (motor → eixo da bomba). Constante do equipamento — repita o mesmo valor em todas as linhas. Só habilita a linha teórica "1X Pump Shaft" se preenchida. |
| `reducao_cavidade`| não | Razão de redução da cavidade (eixo → cavidade da PCP). Constante do equipamento — repita o mesmo valor em todas as linhas. Só habilita as linhas teóricas "1X/2X/4X Cavity" se preenchida. |

Exemplo:

```csv
condicao,f_vfd_hz,vazao_m3h,reducao_shaft,reducao_cavidade
T1,42.5,4.0,4.28,9.55
T2,31.5,3.5,4.28,9.55
T3,24.0,3.0,4.28,9.55
```

Linhas de condições ausentes do CSV (ou CSV ausente) simplesmente caem no
comportamento padrão (eixo categórico, sem linhas teóricas) para aquela
condição — a etapa não falha por metadado incompleto.