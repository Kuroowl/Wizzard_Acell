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

## ⚖️ Calibração (mV bruto → unidade física)

O pipeline lê o sinal bruto **em mV**, exatamente como sai do arquivo de
origem (sem nenhuma conversão de unidade). Isso só é seguro comparar entre
condições/canais se **todos** usarem o mesmo modelo de sensor (mesma
sensibilidade, em mV por unidade física) e o mesmo ganho no condicionador.
Na prática isso pode não ser verdade — dois modelos de acelerômetro podem
ter sensibilidades bem diferentes (ex.: 100 mV/g vs. 1000 mV/g), e o
condicionador de sinal pode ter ganho configurável por canal (x1/x10/x100).
Sem correção, comparar amplitude bruta entre canais/condições que usaram
sensor ou ganho diferentes dá um resultado sem sentido físico, mesmo que o
número em mV pareça "maior" ou "menor".

**Antes de confiar em qualquer comparação absoluta de amplitude** (série
temporal, `--escala abs-global`/`db-global` nos heatmaps), confirme com
quem fez a aquisição:
- Qual o modelo/sensibilidade (mV/g ou mV por unidade) de cada sensor físico
  (canal), e se isso é uniforme entre todos os canais/condições.
- Qual o ganho do condicionador usado em cada canal — e se ele mudou entre
  condições/ensaios (comum quando a amplitude esperada varia bastante).
- Se essa informação está registrada em algum lugar (planilha de ensaio,
  configuração do software de aquisição, etc.).

**Etapa 02 (`02_preprocessamento.py`)** aplica a correção automaticamente
se você fornecer um CSV opcional — `--metadados-calibracao caminho.csv`
(ou, se omitido, um arquivo `calibracao.csv` direto na raiz de
`--data_dir`, se existir):

```
sensor,canal,condicao,sensibilidade_mv_por_unidade,ganho,unidade_saida
ACL,Channel 0,,100,1,g
ACL,Channel 1,,1000,10,g
```

- `condicao` vazio vale para **todas** as condições daquele sensor/canal;
  só preencha uma condição específica se o ganho mudou durante o ensaio
  (nesse caso, uma linha por condição em que o ganho for diferente).
- Canais sem entrada no CSV continuam em mV bruto (nenhuma quebra de
  compatibilidade — sem `calibracao.csv`, o pipeline se comporta
  exatamente como antes desta funcionalidade existir).
- Se **parte** dos canais tiver calibração e parte não, a etapa 02 avisa
  explicitamente no console e no log — misturar unidade física com mV
  bruto no mesmo conjunto de dados invalida comparações absolutas entre
  eles.
- A calibração é aplicada uma única vez, no início da etapa 02, antes de
  qualquer outro tratamento (remoção de offset DC, etc.). As etapas 03-06
  (FFT, picos, heatmaps) recebem o sinal já na unidade física e não sabem
  (nem precisam saber) que uma calibração aconteceu. Os rótulos de eixo
  dessas etapas continuam genéricos ("Amplitude") de propósito — nas
  escalas já normalizadas (`pico-canal`, `rms-canal`, `db`/`db-global`
  relativos), a unidade física não muda a leitura do gráfico, então não
  faz sentido rotular como "g" ali.

---

## 📁 Estrutura do Projeto

```text
Wizzard_Acell/
│
├── Scripts/
│   ├── 01_leitura.py          # Leitura e parsing dos dados brutos
│   ├── 02_preprocessamento.py # Calibração (opcional) + limpeza do sinal
│   ├── 03_fft.py              # Espectro via scipy.signal.welch (método de Welch), truncado no teto do sensor
│   ├── 04_picos.py            # Detecção e identificação de picos
│   ├── 05_heatmap.py          # Mapa espectral por condição (freq x T1..Tn), por sensor/canal
│   ├── 06_mapa_espacial.py    # Mapa espacial (freq x canal/sensor), por condição
│   ├── 07_waterfall.py        # Waterfall 3D (freq x condição x amplitude)
│   ├── 08_histograma.py       # Histograma de picos agregados entre condições
│   ├── 09_rms_psd.py          # RMS (tempo + por banda) e PSD
│   └── 10_relatorio.py        # Consolidação e exportação de relatórios
│
├── run_pipeline.py            # Script principal orquestrador
├── requirements.txt           # Dependências do projeto
└── README.md

DadosTratados/                 # Gerada automaticamente na raiz de --data_dir
│
├── Etapas/                    # Todas as saídas .parquet, agrupadas por etapa
│   ├── Leitura/{sensor}/{condicao}.parquet
│   ├── Preprocessamento/{sensor}/{condicao}.parquet
│   ├── FFT/{sensor}/{condicao}.parquet             # freq_hz + amplitude por canal, já truncado no teto do sensor
│   ├── Picos/{sensor}/{condicao}.parquet           # canal, escopo (low/mid/high/global), ordem_pico, freq_hz, amplitude
│   ├── Heatmap/{sensor}/{canal}.parquet            # condicao, freq_hz, amplitude, y_valor (RAW, não escalado)
│   ├── MapaEspacial/{sensor}/{condicao}.parquet    # canal, freq_hz, amplitude, y_valor (RAW, não escalado)
│   ├── Histograma/{sensor}/{canal}.parquet         # escopo, bin_centro_hz, bin_min_hz, bin_max_hz, contagem, amplitude_somada
│   ├── RMS/{sensor}/{canal}.parquet                # condicao, rms_broadband_tempo, rms_low, rms_mid, rms_high, rms_full_espectro
│   └── PSD/{sensor}/{canal}.parquet                # freq_hz + {condicao}_psd
│
├── Figuras/                   # Figuras de análise
│   └── {sensor}/
│       ├── {condicao}/
│       │   ├── TimeSerie/     # etapa 02
│       │   ├── FFTs/          # etapa 03, OPCIONAL (--salvar-figuras, desligado por padrão)
│       │   ├── Picos/         # etapa 04 — mesmo gráfico da FFT + marcador colorido nos picos
│       │   └── MapaEspacial/  # etapa 06 — 3 figuras (low/mid/high), eixo Y = canal
│       ├── Heatmap/           # etapa 05 — 3 figuras por canal (low/mid/high), consolidando TODAS as condições
│       ├── Waterfall/         # etapa 07 — 1 figura por canal (espectro inteiro), consolidando TODAS as condições
│       ├── Histograma/        # etapa 08 — 1 figura por canal (espectro inteiro), picos agregados entre condições
│       ├── RMS/                # etapa 09 — 1 figura por canal (tendência de RMS x condição)
│       └── PSD/                # etapa 09 — 2 figuras por canal (full + low), PSD sobreposta entre condições
│                               # (Heatmap/Waterfall/Histograma/RMS/PSD ficam em {sensor}/, não dentro de {condicao}/)
│
└── Logs/
    └── pipeline_log.txt       # log cumulativo (append), 1 entrada por execução:
                                # timestamp, usuário, parâmetros e pastas alteradas
```

---

## 📡 Teto de frequência por sensor (etapa 03)

O datasheet do acelerômetro especifica a faixa de frequência em que a
resposta é confiável (ex.: `Frequency Range (±5%) 2-10000 Hz` ou
`0.5-10000 Hz`, dependendo do modelo) — em ambos os casos, **10000 Hz é o
teto**. Acima disso, o número que o sensor devolve não representa mais a
vibração real com a mesma fidelidade.

A etapa 03 trunca o espectro **nesse teto antes de salvar** em
`Etapas/FFT` — não é um recorte visual numa figura, é a base de dados que
todas as etapas seguintes (04-09) leem. Nenhuma delas precisa de ajuste
próprio: como todas partem de `Etapas/FFT`, o teto já vem embutido.

```bash
python Scripts/03_fft.py --data_dir <pasta> --freq-teto-acl 10000
```

- `--freq-teto-acl` (padrão **10000.0**): teto para sensores ACL.
- `--freq-teto-pzt` (padrão **sem teto**): mesma ideia para PZT — não há
  datasheet de PZT confirmando um limite ainda; informe se/quando houver.
- `--freq-teto`: fallback para sensores fora do mapeamento ACL/PZT.
- Use `0` (ou negativo) em qualquer um desses pra desligar o teto daquele
  sensor (usa Nyquist = fs/2 inteiro).

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

---

## 🌊 Etapa 07 — Waterfall 3D

Mesma pergunta da etapa 05 (como o espectro de um canal muda entre
condições), mas em cascata 3D em vez de mapa de cor 2D: uma linha por
condição, empilhada ao longo do eixo Y, cada uma na sua altura de
amplitude (eixo Z).

**Independência da etapa 05/06**: a etapa 07 lê direto de `Etapas/FFT`
(saída da etapa 03) — **não** depende do heatmap (05) nem do mapa espacial
(06), e nenhuma delas depende da 07. As três são irmãs, todas consumindo o
mesmo insumo (etapa 03). Isso foi deliberado: dá pra rodar só `01→02→03→07`
sem nunca rodar 05/06, ou mudar um parâmetro de uma sem precisar
reprocessar as outras — mantém a vantagem de velocidade/independência que
já existe entre as demais etapas do pipeline, em vez de encadear o
waterfall depois do heatmap.

Sem subdivisão por faixa (diferente da etapa 05) — **uma figura por canal**,
espectro inteiro. Use `--freq-max` (e, se quiser, comece do zero mesmo) pra
restringir a uma faixa específica em vez de plotar tudo.

```bash
python Scripts/07_waterfall.py --data_dir <pasta> \
    --escala db-global --cmap jet --elev 25 --azim -60 --freq-max 300
```

- `--escala`: as mesmas 6 opções da etapa 05 (`db-global`, `abs-global`,
  `abs-condicao`, `pico-canal`, `rms-canal`, `db`) — mesma matemática,
  documentada acima, aplicada ao eixo Z em vez da cor.
- `--metadados-condicoes`: mesmo CSV/formato da etapa 05 (eixo Y contínuo
  por `f_vfd_hz` quando disponível para todas as condições daquele
  sensor). Sem linhas teóricas (1X/2X cavidade) — não implementadas em 3D
  por enquanto.
- `--freq-max`: frequência máxima (Hz) plotada. Padrão: a menor frequência
  máxima entre as condições daquele sensor/canal (evita extrapolar).
- `--cmap` (padrão `jet`, diferente do `viridis` da etapa 05 — visual
  clássico de waterfall), `--elev`/`--azim`: ângulo da câmera 3D em graus
  (padrão 25/-60).
- Sem overlay de picos (etapa 04) por enquanto — só existe hoje no
  heatmap 2D (05/06).

Não salva nenhum parquet em `Etapas/` — é uma etapa puramente visual, sem
nenhuma etapa posterior que dependa da sua saída (ao contrário da 04, cujos
picos são reaproveitados pela 05/06).

---

## 📊 Etapa 08 — Histograma de picos agregados

Pergunta diferente das etapas anteriores: em vez de "como o espectro muda
entre condições" (05/06/07), é "que frequências aparecem **toda vez**,
independente da condição, vs. que frequências só aparecem em pontos de
operação específicos".

**Independência**: lê direto de `Etapas/Picos` (saída da etapa 04) — não
recalcula pico nem FFT, não depende de 05/06/07. Dá pra rodar só
`01→02→03→04→08`.

**Método**: empilha (pool) os picos que a etapa 04 já identificou buscando
no **espectro inteiro** (escopo `global`, sem recorte de faixa) de
**todas as condições** de um sensor/canal, num histograma só. Um pico que
cai sempre no mesmo bin (mesma frequência, em toda condição) vira uma
barra alta e estreita — assinatura de algo **fixo na máquina** (ressonância
estrutural, defeito de rolamento, folga mecânica). Um pico que se desloca
com a condição (ex.: a própria frequência do VFD) cai em bins diferentes a
cada condição e se espalha no histograma agregado — assinatura de algo
ligado ao **ponto de operação** (hidráulico/VFD), não à máquina em si.

```bash
python Scripts/08_histograma.py --data_dir <pasta> --n-bins 100
```

- `--n-bins`: número de bins do histograma (padrão **100**), largura
  calculada a partir do intervalo do eixo X (`--freq-min`/`--freq-max`).
- `--freq-min`/`--freq-max`: intervalo do eixo X, **fixo por padrão**
  (`0`-`1000` Hz) — antes era calculado a partir do menor/maior pico
  encontrado, o que fazia o eixo mudar de figura pra figura e dificultava
  comparar; agora fica sempre igual, a menos que você informe outro valor.
- `--y-max`: teto fixo do eixo Y (padrão **30**, no modo contagem — mesmo
  motivo do `--freq-min`/`--freq-max`, eixo comparável entre figuras). No
  modo `--peso-amplitude`, sem teto fixo por padrão (a escala de amplitude
  varia demais entre canais/calibração pra um valor fixo fazer sentido por
  padrão); informe `--y-max` explicitamente se quiser fixar também.
- `--peso-amplitude`: por padrão, o histograma é por **contagem** (quantas
  condições tiveram um pico naquele bin — igual ao protótipo original,
  `src/histogram_and_picosdetector.py`). Com esta flag, soma a amplitude
  dos picos no bin em vez de contar — realça bins com picos fortes mesmo
  que raros.
- Cor da barra por tipo de sensor: **verde para ACL, preto para os
  demais** (PZT incluso) — mesma convenção já usada no gráfico de FFT da
  etapa 04.
- Salva o resultado consolidado em `Etapas/Histograma/{sensor}/{canal}.parquet`
  (colunas: `escopo`, `bin_centro_hz`, `bin_min_hz`, `bin_max_hz`,
  `contagem`, `amplitude_somada`) — pra a futura etapa 10 (relatório) poder
  reaproveitar sem reprocessar nada.

---

## 📈 Etapa 09 — RMS e PSD

Duas métricas de energia, calculadas **sem recalcular nenhuma FFT** — só
reaproveitando o que as etapas 02 e 03 já produziram:

- **RMS de banda larga (broadband)**: `sqrt(mean(sinal**2))` calculado
  **direto no domínio do tempo**, sobre o sinal já calibrado/tratado da
  etapa 02. Não depende de janela, `nperseg` nem de nenhuma escolha da
  etapa 03 — é o RMS "oficial" de severidade de vibração (base de normas
  tipo ISO 10816/20816).
- **RMS por banda (low/mid/high) e PSD (densidade espectral)**:
  derivados do espectro que a etapa 03 já calculou (`scipy.signal.welch`,
  `scaling='spectrum'`, salvo em `Etapas/FFT` como `amplitude=sqrt(Pxx)`).
  Pelo teorema de Parseval, a soma de `amplitude**2` nos bins de uma faixa
  é o RMS² daquela faixa; dividir `Pxx` (`=amplitude**2`) pela resolução em
  Hz (`df`) dá a densidade espectral de potência (PSD, em unidade²/Hz)
  normalizada — comparável entre ensaios com configurações de Welch
  diferentes, ao contrário do espectro de amplitude cru da etapa 03. RMS
  por banda e PSD são duas leituras da mesma informação: PSD é a
  "densidade", RMS por banda é a PSD integrada naquela faixa.

**Independência**: lê de `Etapas/Preprocessamento` (etapa 02, RMS no
tempo) e `Etapas/FFT` (etapa 03, RMS por banda + PSD) — não depende de
04/05/06/07/08. Dá pra rodar só `01→02→03→09`.

```bash
python Scripts/09_rms_psd.py --data_dir <pasta> --f1 15 --f2 400
```

Gera, por sensor/canal:
1. **Gráfico de tendência de RMS** (`Figuras/{sensor}/RMS/`): RMS x
   condição, uma linha por banda (`broadband` + `low`/`mid`/`high`).
   Eixo X contínuo (`f_vfd_hz`, via `--metadados-condicoes`/`condicoes.csv`,
   mesma regra "tudo ou nada" da etapa 05/07) ou categórico (T1..Tn).
2. **PSD sobreposta** (`Figuras/{sensor}/PSD/`): todas as condições no
   mesmo gráfico (escala log), pra comparar níveis diretamente — diferente
   do heatmap (cor) e do waterfall (3D), aqui dá pra ler o valor com
   precisão. **Duas figuras por canal por padrão**: `full` (espectro
   inteiro) e `low` (0-`f1`) — a faixa low costuma concentrar as tônicas
   de VFD/hidráulicas, e no gráfico `full` ela fica espremida perto do
   eixo Y (escala linear em X cobrindo até dezenas de kHz).
3. Dados consolidados em `Etapas/RMS/{sensor}/{canal}.parquet` (colunas:
   `condicao`, `rms_broadband_tempo`, `rms_low`, `rms_mid`, `rms_high`,
   `rms_full_espectro` — este último é a versão "toda a banda, via
   espectro", útil como conferência cruzada contra `rms_broadband_tempo`)
   e `Etapas/PSD/{sensor}/{canal}.parquet` (colunas: `freq_hz` +
   `{condicao}_psd`) — pra a futura etapa 10 (relatório) reaproveitar.

- `--f1`/`--f2`: mesmos limites de faixa do resto do pipeline.
- `--freq-resolucao`: só afeta o grid de interpolação da PSD sobreposta
  (padrão 0.5 Hz); o RMS por banda sempre usa a resolução nativa de cada
  condição, sem interpolar.
- `--cmap` (padrão `tab10`, qualitativo — uma cor por condição na PSD
  sobreposta).