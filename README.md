# TraceRAG

Status: piloto M004 — retrieval lexical rastreável e mensurável, executado localmente. Não é um produto pronto.

## Objetivo

O TraceRAG pretende oferecer consulta técnica controlada, reproduzível e verificável sobre documentação pública de metrologia, laboratório e Qualidade. O diferencial buscado é o controle do corpus: cada trecho recuperado indica documento, seção, versão capturada e origem, e a qualidade da recuperação é medida antes de qualquer geração de resposta.

O projeto também é um laboratório de aprendizado e um portfólio de AI Engineering. Executar localmente é uma escolha de arquitetura: corpus, índice e avaliação ficam sob controle de quem consulta.

## O que existe nesta versão

- Corpus controlado: três seções HTML da adaptação web da NIST Technical Note 1297, cadastradas em `corpus_manifest.yaml` com análise de uso registrada.
- Ingestão: captura HTTPS das três páginas como snapshots locais, identificados pelo SHA-256 dos bytes brutos, em `data/m004/`, fora do Git.
- Extração e chunks: o corpo técnico de cada página é dividido por subseção (2.1 a 2.7, 3, 4.1 a 4.7); cada chunk guarda documento, seção, locator, URL, hash do snapshot e notas de extração.
- Retrieval lexical: SQLite FTS5 em memória, tokenizer `unicode61`, ranking por `bm25()`.
- Avaliação: dez consultas fixas em inglês (`evaluation/golden_set.jsonl`), julgadas e congeladas antes da primeira busca, com Recall@3 e MRR@3.

Não há LLM, embeddings, geração de respostas, API ou serviço.

## Como executar

Requisitos: Python 3.11 ou posterior, com `sqlite3` compilado com FTS5. A única dependência é PyYAML 6.0.3 (`requirements.txt`). Exemplo no Windows com Git Bash, a partir da raiz do repositório:

```bash
python -m venv data/.venv-m004
data/.venv-m004/Scripts/python.exe -m pip install -r requirements.txt
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1
PY=data/.venv-m004/Scripts/python.exe

$PY -m tracerag ingest      # captura (ou confere) os três snapshots; único comando que usa a rede
$PY -m tracerag build       # extrai e grava data/m004/chunks.jsonl, ou compara com o arquivo existente
$PY -m tracerag search --query "ANOVA" --top-k 3
$PY -m tracerag evaluate    # avaliação offline; --output cria um novo JSON em data/m004/
$PY -m unittest discover -s tests -v
```

`build`, `search` e `evaluate` trabalham offline sobre os snapshots e chunks locais, e nenhum comando sobrescreve arquivo existente. O HTML do site muda entre acessos: uma nova captura gera outros hashes e outros IDs de chunk, então o golden set congelado vale para os snapshots registrados na captura que ele referencia.

## Corpus e atribuição

| ID | Seção | URL |
|---|---|---|
| `tn1297-s2` | NIST TN 1297: 2. Classification of Components of Uncertainty | https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-2-classification-components-uncertainty |
| `tn1297-s3` | NIST TN 1297: 3. Type A Evaluation of Standard Uncertainty | https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-3-type-evaluation-standard-uncertainty |
| `tn1297-s4` | NIST TN 1297: 4. Type B Evaluation of Standard Uncertainty | https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-4-type-b-evaluation-standard-uncertainty |

Fonte: Taylor, B.N. and Kuyatt, C.E. (1994), *Guidelines for Evaluating and Expressing the Uncertainty of NIST Measurement Results*, NIST Technical Note 1297 (1994 Edition), versão web adaptada da Technical Note, National Institute of Standards and Technology. Republished courtesy of the National Institute of Standards and Technology.

As condições de uso foram analisadas para estudo e experimento local, com atribuição e sem redistribuição (`usage_review` no manifesto). Os textos ficam apenas nos snapshots e derivados locais ignorados pelo Git. A análise não se estende a outras fontes e não define a licença do projeto, que continua uma decisão pendente.

## Avaliação

Recall@3 é a fração dos chunks relevantes julgados que aparece entre os três primeiros resultados. RR@3 vale 1 dividido pela posição do primeiro relevante, se ele estiver entre os três primeiros, e 0 caso contrário. MRR@3 é a média de RR@3 nas oito consultas com evidência julgada; as duas consultas sem evidência no corpus são relatadas à parte.

Primeira medição (15 chunks, um relevante por consulta): Recall@3 médio 1,0 e MRR@3 = 11/12 ≈ 0,917. Um ranking aleatório teria Recall@3 esperado de 0,2 e RR@3 esperado de 11/90 ≈ 0,122; essa é uma expectativa analítica, não uma medição. Na consulta sobre métodos estatísticos da avaliação Tipo A, o trecho relevante ficou em terceiro lugar. As duas consultas sem evidência retornaram candidatos lexicais, o que mostra que match lexical não comprova suporte.

Esse conjunto pequeno foi escrito junto com o corpus: serve para diagnóstico e para detectar regressões, não para medir generalização.

## Limitações

- Apenas três seções HTML de uma publicação, em inglês; consultas em português não são avaliadas.
- Fórmulas, imagens e tabelas não são interpretadas; subscritos e sobrescritos são mantidos com os delimitadores `_{...}` e `^{...}`.
- O score `bm25()` é um valor de ranking, não uma probabilidade de suporte.
- Não há geração de resposta nem recusa por evidência insuficiente.

Os próximos passos possíveis estão em `docs/ARCHITECTURE.md`; nenhum deles está aprovado por esta versão.
