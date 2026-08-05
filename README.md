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
│   ├── 03_fft.py              # Espectro via scipy.signal.welch (método de Welch)
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
│   └── Heatmap/{sensor}/{canal}.parquet            # condicao, freq_hz, amplitude, y_valor (RAW, não escalado)
│
├── Figuras/                   # Figuras de análise
│   └── {sensor}/
│       ├── {condicao}/
│       │   ├── TimeSerie/     # etapa 02
│       │   ├── FFTs/          # etapa 03, OPCIONAL (--salvar-figuras, desligado por padrão)
│       │   └── Picos/         # etapa 04 — mesmo gráfico da FFT + marcador vermelho nos picos
│       └── Heatmap/           # etapa 05 — 1 figura por canal, consolidando TODAS as condições
│                               # (por isso fica em {sensor}/Heatmap/, não dentro de {condicao}/)
│
└── Logs/
    └── pipeline_log.txt       # log cumulativo (append), 1 entrada por execução:
                                # timestamp, usuário, parâmetros e pastas alteradas
```

---

## 🗺️ Etapa 05 — Mapa espectral por condição

Gera, por sensor e por canal, **3 heatmaps** — um por faixa (`low` 0-f1,
`mid` f1-f2, `high` f2-freq_max) — com eixo X = frequência (Hz), eixo
Y = condição de ensaio (T1...Tn), cor = amplitude. Cada figura consolida
o espectro (`Etapas/FFT`) de todas as condições daquele sensor/canal
**dentro daquela faixa**, com os picos daquela faixa (`Etapas/Picos`)
sobrepostos como marcadores coloridos (azul/verde/vermelho — ver etapa 04;
desligável com `--sem-picos`). A matriz salva em `Etapas/Heatmap/{sensor}/
{canal}.parquet` continua cobrindo o espectro inteiro (0 até freq_max), só
as figuras são divididas por faixa.

Precisa de pelo menos 2 condições por sensor/canal (heatmap de 1 linha só
não faz sentido); condições/canais sem dado suficiente são pulados com
aviso, sem interromper o restante.

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
sobre o mapa. Essa decisão (contínuo vs. categórico) é tomada **por
sensor**: se faltar `f_vfd_hz` de qualquer condição daquele sensor, o
sensor inteiro cai no eixo categórico (evita misturar as duas escalas no
mesmo gráfico) e um aviso lista quais condições ficaram sem o dado.

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
sensor — a etapa não falha por metadado incompleto.

**Escala de cor (`--escala`)** — a referência usada pra colorir cada
heatmap é calculada **dentro de cada faixa** (não no espectro inteiro), pra
uma faixa muito mais forte não afogar visualmente as outras duas:

| valor              | o que faz |
|--------------------|-----------|
| `abs-global` **(padrão)** | Opção A: uma única referência (a maior amplitude absoluta entre TODAS as condições daquela faixa) pra toda a figura. Unidade original preservada, comparável entre condições. |
| `abs-condicao`     | Opção B: cada condição usa o próprio pico como referência (equivalente, na prática, ao mesmo cálculo do `pico-canal` abaixo — é a única forma de dar a cada linha seu próprio teto de cor numa imagem só). |
| `pico-canal`       | Normalização relativa: cada condição dividida pelo próprio pico (0 a 1, sem unidade). |
| `rms-canal`        | Cada condição dividida pelo próprio RMS (realça sinal acima do "nível médio de ruído" daquela condição). |
| `db`               | Amplitude em dB relativa ao pico de cada condição; piso configurável via `--db-min` (padrão -40 dB). |

**Outros parâmetros**: `--cmap` (colormap do matplotlib), `--freq-max`
(teto da faixa `high`; padrão automático), `--freq-resolucao` (grid comum
de frequência usado para interpolar as condições, já que cada uma pode ter
resolução espectral diferente), `--f1`/`--f2` (limites `low`/`mid`/`high`,
mesma convenção das etapas 03/04).