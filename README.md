# 🧙‍♂️ Wizzard Acell

Ferramenta de análise e processamento de dados de acelerometria/sinais, com pipeline automatizado para leitura, pré-processamento, análise no domínio da frequência (FFT) e geração de relatórios visuais.

---

## 🏷️ Terminologia: tipo de sensor vs. canal (sensor físico)

`sensor` no código (`ACL`, `PZT`) é o **tipo/família de aquisição**, não um
sensor físico individual. Os sensores físicos de verdade são os **canais**
dentro de cada arquivo (`Channel 0`, `Channel 1`...) — hoje 4 canais nos
arquivos ACL e 7 nos arquivos PZT, totalizando 11 sensores distribuídos ao
longo da tubulação. Todo o processamento (FFT, picos, heatmap) já opera por
canal individual (cada um com sua própria coluna de amplitude, seus
próprios picos, sua própria normalização); só os logs/prints que diziam
"sensores processados: N" se referiam a **tipos** de sensor, não à
contagem de canais — texto corrigido, sem mudança de comportamento.

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
│   ├── 05_heatmap.py          # Mapa espectral por condição (freq x T1..Tn), por sensor/canal
│   ├── 06_mapa_espacial.py    # Mapa espacial (freq x canal/sensor), por condição
│   ├── 07_waterfall.py        # Gráfico Waterfall espectral
│   └── 08_relatorio.py        # Consolidação e exportação de relatórios
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
│   ├── Heatmap/{sensor}/{canal}.parquet            # condicao, freq_hz, amplitude, y_valor (RAW, não escalado)
│   └── MapaEspacial/{sensor}/{condicao}.parquet    # canal, freq_hz, amplitude, y_valor (RAW, não escalado)
│
├── Figuras/                   # Figuras de análise
│   └── {sensor}/
│       ├── {condicao}/
│       │   ├── TimeSerie/     # etapa 02
│       │   ├── FFTs/          # etapa 03, OPCIONAL (--salvar-figuras, desligado por padrão)
│       │   ├── Picos/         # etapa 04 — mesmo gráfico da FFT + marcador colorido nos picos
│       │   └── MapaEspacial/  # etapa 06 — 3 figuras (low/mid/high), eixo Y = canal
│       └── Heatmap/           # etapa 05 — 3 figuras por canal (low/mid/high), consolidando TODAS as condições
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

**Metadados por condição (opcional)** — `--metadados-condicoes caminho.csv`
(ou, se omitido, um arquivo `condicoes.csv` direto na raiz de `--data_dir`
é usado automaticamente, se existir): quando informado, o eixo Y passa a
ser a frequência real do inversor (contínuo, em Hz) em vez do rótulo
categórico, e — se as colunas de redução também estiverem preenchidas —
as linhas teóricas voltam a ser desenhadas sobre o mapa. Essa decisão
(contínuo vs. categórico) é tomada **por sensor**: se faltar `f_vfd_hz` de
qualquer condição daquele sensor, o sensor inteiro cai no eixo categórico
(evita misturar as duas escalas no mesmo gráfico) e um aviso lista quais
condições ficaram sem o dado.

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
| `db-global` **(padrão)** | `20·log10(A / A_max_global)` — só a condição que contém o pico global bate 0 dB; as demais ficam abaixo, preservando a comparação de intensidade ABSOLUTA entre condições. Piso configurável via `--db-min` (padrão -40 dB). |
| `abs-global`       | Opção A "crua": uma única referência (a maior amplitude absoluta entre TODAS as condições daquela faixa) pra toda a figura, mas em unidade original (não em dB) — comparável entre condições, porém sem a compressão logarítmica que costuma facilitar a leitura visual. |
| `abs-condicao`     | Opção B: cada condição usa o próprio pico como referência (equivalente, na prática, ao mesmo cálculo do `pico-canal` abaixo — é a única forma de dar a cada linha seu próprio teto de cor numa imagem só). |
| `pico-canal`       | Normalização relativa: cada condição dividida pelo próprio pico (0 a 1, sem unidade). |
| `rms-canal`        | Cada condição dividida pelo próprio RMS (realça sinal acima do "nível médio de ruído" daquela condição). |
| `db`               | Amplitude em dB relativa ao pico de CADA condição (todas batem 0 dB no próprio pico — não preserva intensidade absoluta entre condições, ao contrário do `db-global`); piso via `--db-min`. |

O texto da linha escolhida na tabela acima (fórmula, funções do numpy,
parâmetros) é registrado no log a cada execução (`metodo_escala`), mesmo
espírito do `metodo_espectral` da etapa 03 — dá pra saber exatamente como
uma figura antiga foi gerada sem precisar adivinhar pelo nome curto da opção.

**Outros parâmetros**: `--cmap` (colormap do matplotlib), `--freq-max`
(teto da faixa `high`; padrão automático), `--freq-resolucao` (grid comum
de frequência usado para interpolar as condições, já que cada uma pode ter
resolução espectral diferente), `--f1`/`--f2` (limites `low`/`mid`/`high`,
mesma convenção das etapas 03/04).

**Nota técnica — eixo Y sempre com bandas de altura igual**: mesmo no modo
contínuo (`f_vfd_hz` na etapa 05, `posicao_m` na etapa 06), o eixo Y usa
posições uniformes (uma banda por condição/canal, todas do mesmo tamanho).
Os valores reais (`42.5`, `31.5`, `24.0`...) viram só o RÓTULO de cada
banda, não a posição em si — porque não são uma variável amostrada
continuamente, são pontos discretos escolhidos no ensaio. Usar o valor real
como posição faria bandas de tamanho desigual (proporcional ao espaçamento
real entre eles) e empurraria os pontos das condições extremas pra beira
do gráfico. Cada figura também traz uma linha branca sólida marcando a
divisão entre bandas (nos major ticks do eixo Y) e uma linha branca sutil
no centro de cada banda, lembrando que aquele ponto é um valor discreto —
a altura da banda é só espaçamento visual, não uma faixa de valores. Nas
linhas teóricas da etapa 05 (1X motor/shaft/cavidade), isso significa que
elas são desenhadas como uma polilinha ligando os N pontos realmente
ensaiados (com um marcador em cada), não uma curva contínua inventada
entre eles.

---

## 🧭 Etapa 06 — Mapa espacial (entre sensores/canais)

Enquanto a etapa 05 compara uma condição ao longo do tempo de operação
(mesmo canal, várias condições), a etapa 06 faz o oposto: compara os
**canais entre si, para uma condição fixa**. Gera, por sensor e por
condição, **3 heatmaps** (`low`/`mid`/`high`) com eixo X = frequência,
eixo Y = canal (posição física, se disponível), cor = amplitude.
Responde "nessa condição, qual sensor está vibrando mais forte, e em que
frequência?" — o mapa espacial propriamente dito.

Como os canais dentro de um mesmo grupo (sensor, condição) já compartilham
o mesmo grid de frequência (saída da etapa 03/welch), não há interpolação
aqui — só empilha as colunas na ordem certa.

Feito **dentro do mesmo tipo de sensor** (ACL entre si, PZT entre si):
comparar amplitude bruta entre ACL e PZT diretamente pode não ter sentido
físico, já que são famílias de sensor diferentes com sensibilidade/unidade
de calibração possivelmente distintas.

Ao contrário da etapa 05, aqui **não** são desenhadas linhas teóricas (1X
motor, shaft, cavidade) — elas dependem do eixo Y ser uma frequência (VFD),
o que não é o caso quando o eixo Y é posição física.

**Eixo Y por padrão**: nome cru do canal (`Channel 0`, `Channel 1`...),
ordenado numericamente.

**Metadados por canal (opcional)** — `--metadados-canais caminho.csv`
(ou, se omitido, um arquivo `canais.csv` direto na raiz de `--data_dir` é
usado automaticamente, se existir): quando informado E completo para
todos os canais daquele tipo de sensor, o eixo Y passa a ser a posição
física real (metros, ao longo da tubulação), com rótulos descritivos no
lugar do nome cru do canal. Mesma lógica "tudo ou nada por tipo de sensor"
da etapa 05: se faltar `posicao_m` de qualquer canal daquele tipo, cai pro
eixo categórico com aviso.

Formato esperado do CSV (cabeçalho obrigatório, 1 linha por canal):

| coluna       | obrigatória? | descrição |
|--------------|:---:|-----------|
| `sensor`     | sim | Tipo de sensor (`ACL` ou `PZT`), deve bater com o usado nas demais etapas. |
| `canal`      | sim | Nome exato da coluna de origem (ex.: `Channel 0`). |
| `posicao_m`  | não | Posição física ao longo da tubulação, em metros. Se vazio para qualquer canal do tipo, todo o tipo de sensor cai no eixo categórico. |
| `rotulo`     | não | Nome descritivo pro eixo Y (ex.: "Sucção bomba"). Se vazio, usa o nome cru do canal. |

Exemplo:

```csv
sensor,canal,posicao_m,rotulo
ACL,Channel 0,0.0,Sucção bomba
ACL,Channel 1,2.5,Após bomba
ACL,Channel 2,8.0,Meio da linha
ACL,Channel 3,15.0,Saída sistema
```

**Outros parâmetros**: `--escala` (mesmas 6 opções da etapa 05, mas a
referência agora é por CANAL/SENSOR em vez de por condição — padrão `db-global`),
`--cmap`, `--freq-max`, `--f1`/`--f2`, `--sem-picos`.