# TraceRAG

Status: piloto M004 (retrieval lexical rastreável e mensurável) e comparação M006 (retrieval semântico multilíngue local × BM25), executados localmente. É retrieval com avaliação; não há geração de respostas. Não é um produto pronto.

## Objetivo

O TraceRAG pretende oferecer consulta técnica controlada, reproduzível e verificável sobre documentação pública de metrologia, laboratório e Qualidade. O diferencial buscado é o controle do corpus: cada trecho recuperado indica documento, seção, versão capturada e origem, e a qualidade da recuperação é medida antes de qualquer geração de resposta.

O projeto também é um laboratório de aprendizado e um portfólio de AI Engineering. Executar localmente é uma escolha de arquitetura: corpus, modelos, índices e avaliação ficam sob controle de quem consulta.

## O que existe nesta versão

- Corpus controlado: três seções HTML da adaptação web da NIST Technical Note 1297, cadastradas em `corpus_manifest.yaml` com análise de uso registrada.
- Ingestão (M004): captura HTTPS das três páginas como snapshots locais, identificados pelo SHA-256 dos bytes brutos, em `data/m004/`, fora do Git.
- Extração e chunks (M004): o corpo técnico de cada página é dividido por subseção (2.1 a 2.7, 3, 4.1 a 4.7); cada chunk guarda documento, seção, locator, URL, hash do snapshot e notas de extração.
- Retrieval lexical (M004): SQLite FTS5 em memória, tokenizer `unicode61`, ranking por `bm25()`.
- Retrieval semântico (M006): embeddings do modelo `intfloat/multilingual-e5-small` (revisão fixa), em CPU, a partir dos mesmos quinze chunks, com proveniência completa de modelo, vetores e tokens.
- Avaliação: dez consultas fixas em inglês da M004 (`evaluation/golden_set.jsonl`) e 24 consultas da M006 em doze pares português/inglês (`evaluation/golden_set_m006.jsonl`), julgadas e congeladas antes das medições, com Recall@3, MRR@3 e comparação pareada entre os métodos.

Não há LLM gerador, resposta com grounding, recusa por evidência insuficiente, busca híbrida, API ou serviço.

## Como executar — M004 (lexical)

Requisitos: Python 3.11 ou posterior (exercitado em 3.12.10), com `sqlite3` compilado com FTS5. A única dependência é PyYAML 6.0.3 (`requirements.txt`). Exemplo no Windows com Git Bash, a partir da raiz do repositório:

```bash
python -m venv data/.venv-m004
data/.venv-m004/Scripts/python.exe -m pip install -r requirements.txt
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1
PY=data/.venv-m004/Scripts/python.exe

$PY -m tracerag ingest      # captura (ou confere) os três snapshots; único comando que usa a rede
$PY -m tracerag build       # extrai e grava data/m004/chunks.jsonl, ou compara com o arquivo existente
$PY -m tracerag search --query "ANOVA" --top-k 3
$PY -m tracerag evaluate    # avaliação offline; --output cria um novo JSON em data/m004/
```

`build`, `search` e `evaluate` trabalham offline sobre os snapshots e chunks locais, nunca importam torch/transformers/numpy, e nenhum comando sobrescreve arquivo existente. O HTML do site muda entre acessos: uma nova captura gera outros hashes e outros IDs de chunk, então os golden sets congelados valem para os snapshots registrados na captura que eles referenciam.

## Como executar — M006 (semântico, offline)

Plataforma exercitada: CPython 3.12.10 Windows AMD64, CPU, um ambiente próprio em `data/.venv-m006` (o lock de dependências é específico dessa plataforma). A aquisição é a única etapa com rede; ela baixa somente os 26 wheels e os nove arquivos do modelo fixados nos locks, de hosts HTTPS listados, e confere tamanho e SHA-256 antes de promover cada arquivo:

```bash
PYTHON312=/c/Users/<usuário>/AppData/Local/Programs/Python/Python312/python.exe   # Python base, caminho completo
$PYTHON312 -m venv data/.venv-m006
PY=data/.venv-m006/Scripts/python.exe
$PY tools/acquire_m006.py --downloads config/m006_downloads.json --model-lock config/m006_model.json --output-dir data/m006
$PY -m pip --isolated install --no-index --find-links data/m006/wheels --require-hashes --only-binary=:all: --no-cache-dir -r requirements-m006-win-cpu.lock
$PY -m pip check
```

Depois da aquisição tudo roda offline. Em cada chamada: `PYTHONPATH=src`, `PYTHONDONTWRITEBYTECODE=1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`, `DO_NOT_TRACK=1`, `HF_HUB_DISABLE_XET=1`, `TOKENIZERS_PARALLELISM=false`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, e `TMP`/`TEMP`/`HF_HOME`/`HF_HUB_CACHE`/`XDG_CACHE_HOME`/`TORCH_HOME` em subdiretórios de `data/m006`. Os comandos que carregam o modelo também fixam esses valores, bloqueiam a rede do próprio processo (audit hook de socket/HTTP) e recusam começar com menos de 2 GiB de memória física disponível (menor de três amostras):

```bash
$PY -m tracerag freeze --execution-revision <revisão> --packet-sha256 <sha> --pre-sha256 <sha> --semantic-t0 "<instante>"
$PY -m tracerag semantic-build --output-dir data/m006/index
$PY -m tracerag semantic-search --query "Qual é a diferença entre uma avaliação de incerteza Tipo A e uma avaliação Tipo B?" --top-k 3
$PY -m tracerag compare --output data/m006/results/comparison.json
$PY -m tracerag observe-costs --output data/m006/results/performance.json
TRACERAG_M006_REAL=1 $PY -m unittest discover -s tests -v
```

`freeze` registra os inputs, locks, parâmetros e o contexto de execução antes de qualquer busca ou encoding; os demais comandos conferem o freeze antes de agir. Saídas só são criadas dentro de `data/m006/` e nunca substituem um caminho existente; reproduções usam diretórios novos (por exemplo, `semantic-build --output-dir data/m006/reproduction/index --reference-index data/m006/index` e `compare --index-dir data/m006/reproduction/index --reference data/m006/results/comparison.json --output data/m006/reproduction/comparison.json`). Sem `TRACERAG_M006_REAL=1`, a suíte roda os testes sintéticos (sem modelo e sem busca no corpus) e os 30 testes da M004.

## Corpus e atribuição

| ID | Seção | URL |
|---|---|---|
| `tn1297-s2` | NIST TN 1297: 2. Classification of Components of Uncertainty | https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-2-classification-components-uncertainty |
| `tn1297-s3` | NIST TN 1297: 3. Type A Evaluation of Standard Uncertainty | https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-3-type-evaluation-standard-uncertainty |
| `tn1297-s4` | NIST TN 1297: 4. Type B Evaluation of Standard Uncertainty | https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-4-type-b-evaluation-standard-uncertainty |

Fonte: Taylor, B.N. and Kuyatt, C.E. (1994), *Guidelines for Evaluating and Expressing the Uncertainty of NIST Measurement Results*, NIST Technical Note 1297 (1994 Edition), versão web adaptada da Technical Note, National Institute of Standards and Technology. Republished courtesy of the National Institute of Standards and Technology.

As condições de uso foram analisadas para estudo e experimento local, com atribuição e sem redistribuição (`usage_review` no manifesto). Os textos ficam apenas nos snapshots e derivados locais ignorados pelo Git. A análise não se estende a outras fontes e não define a licença do projeto, que continua uma decisão pendente.

## Modelo de embeddings (M006)

`intfloat/multilingual-e5-small`, revisão `614241f622f53c4eeff9890bdc4f31cfecc418b3`: encoder de 12 camadas, vetores de 384 dimensões, até 512 tokens, 117.653.760 parâmetros em float32. Autoria: Wang, L.; Yang, N.; Huang, X.; Yang, L.; Majumder, R.; Wei, F., *Multilingual E5 Text Embeddings: A Technical Report*, arXiv:2402.05672 (2024). O model card dessa revisão declara licença MIT; o uso aqui é somente inferência local para comparação diagnóstica, e os arquivos do modelo não são redistribuídos (ficam em `data/m006/model`, ignorados pelo Git, com o card integral preservado). Essa declaração não é a licença do TraceRAG nem do conteúdo NIST. A análise de uso registrada em `config/m006_model.json` é técnica, não um parecer jurídico.

## Avaliação

Recall@3 é a fração dos chunks relevantes julgados (R) que aparece entre os três primeiros resultados. RR@3 vale 1 dividido pela posição do primeiro relevante, se ele estiver entre os três primeiros, e 0 caso contrário. MRR@3 é a média de RR@3 nas consultas com evidência julgada; consultas sem evidência no corpus ficam fora dos denominadores (N/A) e são relatadas à parte.

### M004 — BM25, dez consultas em inglês

Primeira medição (15 chunks, um relevante por consulta): Recall@3 médio 1,0 e MRR@3 = 11/12 ≈ 0,917. Um ranking aleatório teria Recall@3 esperado de 0,2 e RR@3 esperado de 11/90 ≈ 0,122; essa é uma expectativa analítica, não uma medição. Na consulta sobre métodos estatísticos da avaliação Tipo A, o trecho relevante ficou em terceiro lugar. As duas consultas sem evidência retornaram candidatos lexicais, o que mostra que match lexical não comprova suporte. A M006 reproduziu esse resultado com IDs, posições e scores idênticos.

### M006 — BM25 × semântico (medição de 08/10/2026)

Os dois conjuntos são relatados separadamente: o golden da M004 mantém seus rótulos legados (um relevante por consulta), e o da M006 usa a rubrica `direct-contribution-v2` (um chunk é relevante quando contribui diretamente para algum aspecto do escopo da pergunta), com R = 7, 4, 7, 1, 1, 1, 1, 1 nos pares positivos 01, 02, 03, 05, 06, 07, 08 e 09. Como R chega a 7 e só três posições contam, o teto de Recall@3 médio é 185/224 ≈ 0,826.

| Golden M006, positivas | BM25 Recall@3 | BM25 MRR@3 | Semântico Recall@3 | Semântico MRR@3 | Teto Recall@3 |
|---|---|---|---|---|---|
| Todas (16) | 27/64 ≈ 0,422 | 53/96 ≈ 0,552 | 37/56 ≈ 0,661 | 77/96 ≈ 0,802 | 185/224 ≈ 0,826 |
| Português (8) | 11/56 ≈ 0,196 | 11/48 ≈ 0,229 | 61/112 ≈ 0,545 | 2/3 ≈ 0,667 | 185/224 |
| Inglês (8) | 145/224 ≈ 0,647 | 7/8 = 0,875 | 87/112 ≈ 0,777 | 15/16 = 0,9375 | 185/224 |

- Consultas com pelo menos um relevante no top-3: BM25 10/16 (PT 3, EN 7); semântico 14/16 (PT 6, EN 8).
- Comparação pareada em RR@3 (primária): semântico melhor em 6, igual em 10, BM25 melhor em 0. Em Recall@3: semântico melhor em 7, igual em 8, BM25 melhor em 1 (`m006-en-02`, em que o BM25 trouxe três relevantes e o semântico dois).
- Golden M004 (oito positivas, rótulos legados): os dois métodos têm Recall@3 1,0 e MRR@3 11/12; o semântico pôs o relevante de `q04` em primeiro (BM25: terceiro) e o de `q01` em terceiro (BM25: primeiro).
- Referência ao acaso com o R real de cada pergunta (analítica, N = 15, k = 3): Recall@3 esperado 1/5, probabilidade de pelo menos um acerto 1543/3640 ≈ 0,424 e RR@3 esperado 19099/65520 ≈ 0,292 na média das 16 positivas.
- Negativas: o semântico sempre devolve três candidatos, e o BM25 devolve candidatos sempre que algum termo coincide. Nas oito negativas da M006, o cosseno do primeiro candidato vai de 0,779 a 0,882; nas 16 positivas, de 0,824 a 0,908. As faixas se sobrepõem, e um score alto não prova suporte nem falha.

A diferença está quase toda no português: o BM25 só casa termos escritos igual. Nas doze consultas em português, os únicos termos que existem no texto inglês são letras isoladas (`a`, `b`, `e`, inclusive `é`, que o tokenizer `unicode61` iguala a `e` ao remover acentos) e coincidências como `as`, `for` e `no`; nenhum conceito casa. No inglês a diferença entre os métodos é menor. Os pares de tradução não são observações independentes, e as perguntas foram escritas conhecendo o corpus: os números são diagnóstico, não estimativa de generalização.

### Exemplo rastreável

Pergunta original de Lucas (`m006-pt-01`): "Qual é a diferença entre uma avaliação de incerteza Tipo A e uma avaliação Tipo B?" (R = 7: 2.2, 2.3, 2.5, 2.6, 3, 4.1, 4.7).

- Semântico: 4.1 (0,8577), 2.6 (0,8355), 2.5 (0,8347), os três relevantes: Recall@3 3/7 (o teto) e RR@3 1.
- BM25: 4.4, 4.5 e 4.1, só 4.1 relevante: Recall@3 1/7 e RR@3 1/3.

Negativa (`m006-pt-04`, também de Lucas): "Como as contribuições de incerteza Tipo A e Tipo B são combinadas para obter a incerteza padrão combinada?" As seções preservadas não trazem esse procedimento. O semântico devolveu 4.2 (0,8442), 4.1 (0,8401) e 4.3 (0,8383), e o BM25 devolveu 4.4, 4.5 e 4.3: candidatos com tema próximo, nenhum com a combinação pedida. Cada trecho pode ser conferido pelo `chunk_id`, pelo locator e pela URL acima.

### Custo local (mesma máquina, um thread, CPU)

| Medida | BM25 | Semântico |
|---|---|---|
| Mediana por consulta (7 amostras, 24 consultas) | ≈ 0,099 ms | ≈ 53,0 ms (52,9 ms codificando a consulta; 0,05 ms ordenando) |
| Varredura das 24 consultas (mediana de 7) | ≈ 2,5 ms | ≈ 1,30 s |
| Preparação | índice FTS5 0,7 ms | imports 2,0 s; carga do modelo 4,1 s; vetores persistidos 10 ms |
| Pico do processo (PeakWorkingSetSize) | 30.121.984 B (≈ 29 MiB) | 936.046.592 B (≈ 893 MiB) |

Construir o índice dos 15 chunks, incluindo a carga do modelo, levou 16,8 s; nenhum chunk passou de 512 tokens (o maior, 4.6, tem 443). Espaço: wheels 153.773.204 B, arquivos do modelo 493.292.441 B, ambiente `data/.venv-m006` 762.621.360 B, índice 44.397 B. São observações locais, não benchmark. A reprodução offline em diretórios novos gerou vetores com bytes idênticos e os mesmos rankings e métricas.

## O que a comparação ensina

- Uma pergunta vira vetor assim: prefixo `query: ` (o texto do chunk recebe `passage: `), tokenização, um vetor por token na última camada, média só dos tokens reais indicados pela máscara de atenção e normalização para comprimento 1. O score é o produto escalar de dois vetores unitários, isto é, o cosseno do ângulo entre eles.
- O prefixo segue o treino do modelo; a média pela máscara impede que tokens de preenchimento entrem no vetor; a normalização faz o score depender da direção, não do tamanho do texto.
- Recuperar um trecho relevante significa que ele contribui para a pergunta; um acerto não cobre todos os aspectos quando R é maior que três.
- Score alto não é resposta: neste modelo quase todos os cossenos ficam entre 0,7 e 1,0, e as negativas recebem candidatos com scores na mesma faixa das positivas.
- O BM25 vence quando a pergunta usa as palavras do texto (inglês) e perde quando o idioma muda; o semântico custa cerca de 500 vezes mais por consulta e quase 900 MiB de memória, ainda assim abaixo de 0,1 s por consulta nesta máquina.

## Limitações

- Três seções HTML de uma publicação, em inglês, quinze chunks; perguntas em português são comparadas com texto inglês, sem tradução.
- Golden sets pequenos, escritos conhecendo o corpus; os 24 casos da M006 são doze pares de tradução. Os rótulos são propostas do Architect conferidas pelo Auditor; cinco perguntas em português são originais de Lucas.
- Fórmulas, imagens e tabelas não são interpretadas; subscritos e sobrescritos são mantidos com os delimitadores `_{...}` e `^{...}`.
- O modelo pré-treinado pode ter visto material NIST no treino.
- Os scores `bm25()` e de cosseno são valores de ranking de naturezas diferentes, não probabilidades de suporte; nenhum limiar foi calibrado.
- Não há geração de resposta nem recusa por evidência insuficiente.

Os próximos passos possíveis estão em `docs/ARCHITECTURE.md`; nenhum deles está aprovado por esta versão.
