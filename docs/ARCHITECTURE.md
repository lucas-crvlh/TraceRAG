# Arquitetura do TraceRAG

Estado: piloto M004 implementado (retrieval lexical rastreável e mensurável) e comparação M006 implementada (retrieval semântico multilíngue local, medido contra o BM25 nos mesmos chunks). O sistema recupera e avalia trechos; não gera respostas. O restante do fluxo continua planejado.

## Objetivo e princípios

O objetivo é oferecer consulta técnica local-first sobre documentação pública de metrologia, laboratório e Qualidade, com respostas futuras fundamentadas em evidências e referências às fontes.

- Persona: profissionais de metrologia, laboratório e Qualidade que precisam localizar e conferir o trecho exato de uma fonte técnica, inclusive perguntando em português sobre textos em inglês.
- Controle do corpus: só entram fontes públicas cadastradas no manifesto depois da análise das condições de uso.
- Rastreabilidade: cada trecho carrega documento, seção, locator, URL e o hash do snapshot capturado; cada vetor carrega modelo, revisão, hashes de arquivos, tokens vistos e código produtor.
- Avaliação antecipada: a recuperação é medida com julgamentos congelados antes de cada medição, e antes de acrescentar busca híbrida ou LLM.
- Local-first: a execução local é uma escolha de arquitetura; corpus, modelo, índices e avaliação ficam sob controle de quem consulta, sem serviço externo de inferência.
- Aprendizado: cada componente é simples o bastante para ser lido, explicado e substituído com medição.

## Fluxo implementado (M004, lexical)

```text
corpus_manifest.yaml -> ingest -> snapshots HTML (data/m004/raw) + snapshots.json
snapshots -> build -> unidades -> data/m004/chunks.jsonl (proveniência)
chunks congelados -> índice SQLite FTS5 em memória (bm25) -> search
golden set congelado + índice -> evaluate -> métricas por consulta e agregadas
```

1. `ingest` valida o manifesto (`yaml.safe_load`, campos, tipos, IDs únicos, `local_path` e `usage_review`) e captura cada página uma única vez: HTTPS, timeout de 30 s, limite de 2 MiB, identificação descritiva do cliente e redirecionamento só para HTTPS no mesmo host. Status, tipo de conteúdo, H1 e corpo técnico são conferidos antes de gravar. Snapshots existentes são conferidos pelo hash e reutilizados, nunca baixados de novo nem sobrescritos.
2. `build` extrai somente o conteúdo de `div.text-with-summary`, até o fechamento desse elemento, com `html.parser` da biblioteca padrão; menu, contato, rodapé e scripts ficam de fora. Uma unidade começa apenas no parágrafo cujo texto inicia com o número da subseção; números citados no meio de frases, listas e notas permanecem na unidade corrente. Cada unidade vira um chunk de até 400 palavras, dividido em partes consecutivas sem sobreposição se passar disso.
3. `search` reconstrói o índice SQLite FTS5 em memória a partir dos chunks congelados (somente o texto, tokenizer `unicode61`, sem stemming nem stoplist), transforma a consulta em termos alfanuméricos entre aspas unidos por OR, passados como parâmetro SQL, e ordena por `bm25()` crescente, com desempate por `chunk_id`.
4. `evaluate` roda as dez consultas, calcula Recall@3, RR@3 e MRR@3 e registra hashes de entradas e código, ambiente, parâmetros, resultados individuais e a referência analítica ao acaso.

## Fluxo implementado (M006, semântico)

```text
config/m006_downloads.json + config/m006_model.json -> tools/acquire_m006.py -> data/m006/wheels + data/m006/model (única etapa com rede)
requirements-m006-win-cpu.lock + wheels -> pip offline --require-hashes -> data/.venv-m006
inputs, locks, parâmetros, autoria e contexto -> freeze -> data/m006/freeze.json
chunks congelados + modelo local -> semantic-build -> data/m006/index (embeddings.npy + index.json)
consulta + modelo + índice -> semantic-search (top-3 por cosseno, texto integral e proveniência)
dois golden sets + BM25 + semântico -> compare -> data/m006/results/comparison.json
processos filhos BM25 e semântico em sequência -> observe-costs -> data/m006/results/performance.json
```

1. `tools/acquire_m006.py` usa só a biblioteca padrão. Valida os dois locks, aceita apenas HTTPS nos quatro hosts listados (`files.pythonhosted.org`, `download.pytorch.org`, `huggingface.co`, `us.aws.cdn.hf.co`) e confere cada redirecionamento. Baixa cada arquivo por streaming para um temporário novo em `data/m006/tmp`, mede tamanho e SHA-256 e só então o promove ao destino final, que nunca é sobrescrito. Uma falha só de transporte pode ser repetida uma vez; divergência de tamanho ou hash não. O log guarda origem fixa, hosts dos redirecionamentos, status, bytes e hash, sem URLs assinadas.
2. A instalação é offline no ambiente próprio (`--no-index --find-links data/m006/wheels --require-hashes --only-binary=:all:`), sem `--no-deps`, seguida de `pip check`. São 26 pacotes fixados, entre eles torch 2.9.1+cpu, transformers 4.57.3, tokenizers 0.22.1, numpy 2.3.5, safetensors 0.7.0 e huggingface-hub 0.36.0. O hf-xet fica instalado, mas a rede Xet permanece desligada.
3. `freeze` grava, por criação exclusiva, os hashes dos dois golden sets, chunks, snapshots, manifesto, avaliação M004 e dos três locks. Registra também parâmetros fixos, autoria, limitações, contexto de execução e a proveniência dos recursos copiados byte a byte. Os comandos seguintes recusam agir se algum input mudou.
4. Todo processo que carrega o modelo segue a mesma ordem:
   - confere a guarda de memória: a menor de três amostras de `GlobalMemoryStatusEx`, com 1 s de intervalo, precisa chegar a 2 GiB disponíveis;
   - fixa as variáveis offline e de um thread;
   - instala um audit hook que rejeita eventos de socket/HTTP;
   - só então importa numpy, torch e transformers;
   - fixa um thread intra/inter-op, seed 0 e algoritmos determinísticos;
   - carrega o modelo apenas do diretório local verificado (exatamente os nove arquivos do lock, com tamanho e hash).
5. `semantic-build` codifica os quinze chunks em ordem de `chunk_id`, um por vez (`passage: ` + texto original, máximo de 512 tokens com truncamento, média pela máscara de atenção, normalização L2). Salva `embeddings.npy` (15 × 384, float32, sem pickle) e `index.json`. O `index.json` traz a ordem dos IDs, os tokens antes e depois do truncamento por chunk, a norma de cada vetor, os hashes de chunks, locks, arquivos do modelo e código, versões, threads, memória e o hash do `.npy`.
6. `semantic-search` e `compare` validam o índice antes de consultar:
   - bytes e hash do `.npy` iguais ao registro;
   - forma, valores finitos e norma 1 (tolerância 1e-5);
   - IDs iguais aos chunks congelados;
   - inputs e arquivos do modelo iguais aos atuais.

   A consulta recebe `query: ` e o mesmo protocolo. O score é o produto escalar dos vetores unitários, acumulado em float64 e não multiplicado por 100. Só empates exatos são ordenados por `chunk_id`. A busca devolve o texto integral do trecho e informa quantos tokens o encoder viu.
7. `compare` primeiro reproduz o BM25 histórico da M004 e exige os mesmos IDs, posições, scores e métricas. Depois roda os dois métodos nos dois golden sets e só então aplica os rótulos. Calcula Recall@3, RR@3, teto de Recall@3, acerto e referência analítica ao acaso com o R real de cada pergunta. As médias são macro, nas positivas, por conjunto e por idioma. A comparação pareada conta melhor, igual e pior em RR@3 (primária) e em Recall@3. As negativas ficam como N/A, com seus candidatos.
8. `observe-costs` mantém o processo pai leve e roda um filho por backend, em sequência; o filho BM25 nunca importa torch. Cada filho mede imports, carga e preparação à parte, faz duas varreduras de aquecimento e sete medidas das 24 consultas com `perf_counter_ns` e registra o `PeakWorkingSetSize` do processo inteiro. O pai repete a guarda de memória antes de iniciar o filho semântico e soma o espaço lógico de `data/m006` e `data/.venv-m006` por categoria.

## Contratos

### Manifesto (`corpus_manifest.yaml`, schema 2)

Cada entrada tem `id`, `title`, `source_url`, `local_path` e `usage_review`. O piloto aplica as regras abaixo na ingestão, com validação mínima de campos e tipos; não é um validador geral nem JSON Schema.

#### Caminho local

`local_path` é relativo à raiz do repositório, nunca ao próprio diretório `data/`. Deve começar por `data/`, usar `/` como separador e identificar um arquivo dentro desse diretório. Não são aceitos caminhos absolutos, prefixos de unidade, barras invertidas, segmentos vazios, `.` ou `..`, barra final ou o caminho `data/.gitkeep`.

Ao acessar o arquivo, o caminho resolvido deve permanecer dentro de `data/`, inclusive quando houver links simbólicos ou junções no sistema de arquivos. A ingestão do piloto exige, além disso, que os snapshots fiquem em `data/m004/raw/`.

#### Registro das condições de uso

Cada entrada contém o objeto obrigatório `usage_review`:

- `reviewed_on`: data válida da análise realizada, como texto `"YYYY-MM-DD"` entre aspas. Uma data sem aspas, que o YAML converte em objeto de data, é rejeitada com explicação.
- `evidence_urls`: lista não vazia de URLs públicas das condições consultadas, que fundamentam a análise; não basta repetir a URL de origem sem que ela contenha essa evidência.
- `intended_use`: texto não vazio que descreve o uso pretendido no projeto.
- `decision`: texto `approved`, registrado somente após análise favorável para o uso pretendido.
- `notes`: texto não vazio com as condições identificadas, o fundamento da decisão, os requisitos de atribuição e as restrições aplicáveis ao uso pretendido.

Uma entrada só pode ser cadastrada, e o documento incorporado ao corpus, depois dessa análise aprovada. Condições incertas ou incompatíveis impedem ambos. A disponibilidade pública, isoladamente, não substitui a análise. Mudanças no uso pretendido ou nas condições aplicáveis exigem nova análise e atualização do registro antes de prosseguir com esse uso; a ingestão recusa reutilizar um snapshot cujo registro mudou desde a captura.

A escolha da licença do projeto é uma decisão separada das condições de uso dos documentos de terceiros e continua pendente. O campo `decision` registra uma análise; não cria licença.

### Snapshots (`data/m004/snapshots.json`)

Por documento: URL solicitada e final, título observado, seção, instante da captura com fuso, bytes, SHA-256, tipo de conteúdo, atribuição e a cópia do `usage_review`. A versão é `snapshot:<SHA-256>`, não uma edição editorial presumida.

### Chunks (`data/m004/chunks.jsonl`)

Uma linha JSON por chunk, com `chunk_id`, `document_id`, `publication_id`, `title`, `source_url`, `source_sha256`, `source_version`, `snapshot_path`, `retrieved_at`, `section`, `unit_locator`, `part`, `text`, `text_sha256`, `word_count`, `extraction_notes` e `attribution`. O `chunk_id` combina documento, hash completo do snapshot, locator e parte; os mesmos snapshots e parâmetros reproduzem os mesmos bytes.

### Golden set M004 (`evaluation/golden_set.jsonl`)

Uma linha JSON por consulta, com `id`, `query`, `language`, `answerable`, `relevant_chunk_ids`, `evidence_locators` e `judgment_notes`. Consultas com evidência têm pelo menos um chunk relevante; consultas sem evidência têm lista vazia e ficam fora dos denominadores. Os julgamentos foram feitos por leitura dos chunks e congelados antes da primeira busca. Na M006 esses rótulos legados são mantidos como estão.

### Golden set M006 (`evaluation/golden_set_m006.jsonl`)

24 linhas, doze pares `pt`/`en` (`pair_id`), 16 positivas e 8 negativas. Cada linha tem exatamente estes campos:

- `id`, `query`, `language`, `answerable`, `pair_id`;
- `relevant_chunk_ids` e `evidence_locators`, com documento, locator, parte, hash e URL de cada relevante;
- `relevance_policy` (`direct-contribution-v2`) e `question_scope`;
- `relevance_basis`, com uma razão por chunk relevante, na mesma ordem;
- `non_relevant_notes`, `judgment_notes`, `query_provenance` e `judgment_provenance`.

O validador exige que cada locator corresponda ao chunk e que os dois idiomas de um par tenham os mesmos rótulos. A busca nunca lê esses campos. Há 46 referências de relevância e 13 chunks distintos, e R vale 7, 4, 7, 1, 1, 1, 1, 1 nos pares positivos.

### Locks M006

- `requirements-m006-win-cpu.lock`: 26 pacotes `nome==versão --hash=sha256:...` para CPython 3.12.10 Windows AMD64, somente wheels, sem extras.
- `config/m006_downloads.json`: para cada wheel e arquivo do modelo, nome, tamanho, SHA-256 e URL fixa; mais a lista de hosts permitidos e o total esperado (647.065.645 B).
- `config/m006_model.json`: modelo, revisão, protocolo (prefixos, pooling, normalização, dimensão, 512 tokens), CPU float32, batch 1, um thread, flags de carregamento, tolerâncias numéricas, análise de uso e os nove arquivos com tamanho e hash.

Os três locks e o golden set M006 são cópias byte a byte dos recursos congelados na revisão M006-SEMANTIC-v2; o campo `mission` desses arquivos registra essa proveniência.

### Freeze, índice e resultados (`data/m006/`, ignorados pelo Git)

- `freeze.json`: execução (revisão, hashes do packet e do PRE, instante de referência), instante do congelamento, inputs com caminho/bytes/hash, proveniência dos recursos, resumo dos golden sets, parâmetros e limitações.
- `index/embeddings.npy` + `index/index.json`: vetores unitários float32 e metadados descritos no fluxo; lidos com `allow_pickle=False`.
- `results/comparison.json`:
  - inputs e código com hash, modelo e ambiente;
  - reprodução do BM25 histórico;
  - por pergunta, top-3 dos dois métodos com scores, relevantes recuperados, primeiro relevante, Recall@3, RR@3, diferença, desfecho pareado, teto e acaso;
  - agregados por conjunto e idioma;
  - o bloco `run`, com horário, guarda de memória, memória e rede, fora das comparações de reprodução.
- `results/performance.json`: método, ambiente, as sete amostras por consulta e as medianas, totais por varredura, memória de cada processo, guarda antes do filho semântico e espaço por categoria.
- Reproduções ficam em diretórios novos, como `data/m006/reproduction/`, e nunca substituem a primeira medição.

## Dados locais

Snapshots, chunks, avaliações, wheels, modelo, índices, resultados e os ambientes virtuais ficam sob `data/`, dentro do working tree, mas fora do Git (`/data/*` no `.gitignore`, com exceção de `data/.gitkeep`). A regra de ignore não é backup nem garantia contra cópias feitas por outras ferramentas. Os snapshots e a primeira medição devem ser preservados, porque são a base para reproduzir chunks, vetores e avaliações. A M006 ocupa cerca de 617 MiB de downloads (wheels e arquivos do modelo) e 727 MiB de ambiente instalado.

## Decisões e alternativas

| Componente | Escolha | Alternativa | Trade-off |
|---|---|---|---|
| Formato da fonte | HTML das seções web | PDF da Technical Note | HTML tem estrutura e texto diretos; PDF exigiria parser e tratamento de layout, deixados para depois |
| Extração | `html.parser` restrito ao corpo técnico | biblioteca de parsing genérica | sem dependência extra, mas acoplado à estrutura destas páginas; mudança estrutural bloqueia |
| Chunk | uma unidade por subseção, até 400 palavras | janela fixa com sobreposição | preserva a referência da fonte; unidades longas precisam ser divididas |
| Retrieval lexical | `bm25()` do FTS5 | BM25 próprio ou serviço de busca | algoritmo testado, sem servidor; não trata sinônimos nem outro idioma |
| Retrieval semântico | multilingual-e5-small local, CPU, float32 | modelo maior, API de embeddings, ONNX | roda offline em ~0,9 GiB de pico; qualidade menor que modelos grandes e custo ~500× o BM25 por consulta |
| Protocolo | prefixos `query:`/`passage:`, média pela máscara, L2 | CLS, sem prefixo, max pooling | segue o treino descrito no model card; outro protocolo exigiria nova medição |
| Execução | batch 1, um thread, algoritmos determinísticos | lotes maiores, vários threads | reprodutível e econômico em memória; mais lento do que o hardware permitiria |
| Índice vetorial | matriz numpy persistida, busca exaustiva | banco vetorial, ANN | exato e auditável para 15 vetores; não escala para corpus grande |
| Isolamento de rede | flags offline + audit hook no processo | só variáveis de ambiente | bloqueia rede Python no processo e registra tentativas; não prova isolamento do computador |
| Avaliação | golden sets congelados, Recall@3/MRR@3, comparação pareada | inspeção ad hoc | mede mudanças com o mesmo critério; conjuntos pequenos e não cegos |

## Planejado, não implementado

Sequência recomendada, sem aprovação de missões futuras:

1. Discutir os resultados da M006 e os casos de fronteira (negativas com candidatos de tema próximo; quase-empates de cosseno) antes de decidir o próximo passo.
2. Ampliar corpus e avaliação, inclusive PDF, mais perguntas originais em português e julgamentos independentes, para medir além deste diagnóstico.
3. Comparar busca híbrida (lexical + semântica) com os dois baselines e manter apenas ganhos demonstrados.
4. Acrescentar LLM local com respostas fundamentadas, citações e recusa por evidência insuficiente, avaliando retrieval e resposta separadamente, inclusive com perguntas fora do corpus.
5. Conforme necessidade real: API, persistência de índices em escala, Docker, CI e observabilidade.

## Estado atual e limitações

O sistema cobre três seções HTML em inglês, quinze chunks, dez consultas de diagnóstico da M004 e 24 consultas bilíngues da M006. Fórmulas, imagens e tabelas não são interpretadas. Os scores `bm25()` e de cosseno não são probabilidades de suporte, e ter um candidato bem pontuado não significa ter evidência suficiente. O ambiente semântico foi exercitado apenas em CPython 3.12.10 Windows AMD64. Não há geração de resposta nem abstention. Interface de chat, agentes, múltiplos serviços e implantação não fazem parte desta etapa.
