# Arquitetura do TraceRAG

Estado: piloto M004 implementado — retrieval lexical rastreável e mensurável. O restante do fluxo continua planejado.

## Objetivo e princípios

O objetivo é oferecer consulta técnica local-first sobre documentação pública de metrologia, laboratório e Qualidade, com respostas futuras fundamentadas em evidências e referências às fontes.

- Persona: profissionais de metrologia, laboratório e Qualidade que precisam localizar e conferir o trecho exato de uma fonte técnica.
- Controle do corpus: só entram fontes públicas cadastradas no manifesto depois da análise das condições de uso.
- Rastreabilidade: cada trecho carrega documento, seção, locator, URL e o hash do snapshot capturado.
- Avaliação antecipada: a recuperação é medida com julgamentos congelados antes de acrescentar LLM ou busca semântica.
- Local-first: a execução local é uma escolha de arquitetura; corpus, índice e avaliação ficam sob controle de quem consulta.
- Aprendizado: cada componente é simples o bastante para ser lido, explicado e substituído com medição.

## Fluxo implementado (M004)

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

### Golden set (`evaluation/golden_set.jsonl`)

Uma linha JSON por consulta, com `id`, `query`, `language`, `answerable`, `relevant_chunk_ids`, `evidence_locators` e `judgment_notes`. Consultas com evidência têm pelo menos um chunk relevante; consultas sem evidência têm lista vazia e ficam fora dos denominadores. Os julgamentos foram feitos por leitura dos chunks e congelados antes da primeira busca.

## Dados locais

Snapshots, chunks, avaliações e o ambiente virtual ficam sob `data/`, dentro do working tree, mas fora do Git (`/data/*` no `.gitignore`, com exceção de `data/.gitkeep`). A regra de ignore não é backup nem garantia contra cópias feitas por outras ferramentas. Os snapshots devem ser preservados, porque são a base para reproduzir chunks e avaliações.

## Decisões e alternativas

| Componente | Escolha | Alternativa | Trade-off |
|---|---|---|---|
| Formato da fonte | HTML das seções web | PDF da Technical Note | HTML tem estrutura e texto diretos; PDF exigiria parser e tratamento de layout, deixados para depois |
| Extração | `html.parser` restrito ao corpo técnico | biblioteca de parsing genérica | sem dependência extra, mas acoplado à estrutura destas páginas; mudança estrutural bloqueia |
| Chunk | uma unidade por subseção, até 400 palavras | janela fixa com sobreposição | preserva a referência da fonte; unidades longas precisam ser divididas |
| Retrieval | lexical, `bm25()` do FTS5 | embeddings | baseline barato e explicável; não trata sinônimos nem outro idioma |
| Motor | SQLite FTS5 | BM25 próprio ou serviço de busca | algoritmo testado, sem servidor; detalhes do BM25 ficam no SQLite |
| Índice | em memória, reconstruído a cada execução | índice persistido | nenhum estado a sincronizar; não escala para corpus grande |
| Avaliação | golden set congelado, Recall@3 e MRR@3 | inspeção ad hoc | mede mudanças com o mesmo critério; conjunto pequeno e não cego |

## Planejado, não implementado

Sequência recomendada, sem aprovação de missões futuras:

1. Inspecionar erros de parsing, chunking e ranking deste piloto.
2. Ampliar corpus e avaliação, inclusive PDF e consultas em português, e comparar retrieval lexical com embeddings sobre o mesmo material e os mesmos julgamentos, com medições separadas por idioma.
3. Comparar busca híbrida com os baselines e manter apenas ganhos demonstrados.
4. Acrescentar LLM local com respostas fundamentadas, citações e recusa por evidência insuficiente, avaliando retrieval e resposta separadamente, inclusive com perguntas fora do corpus.
5. Conforme necessidade real: API, persistência de índices, Docker, CI e observabilidade.

## Estado atual e limitações

O piloto cobre três seções HTML em inglês e dez consultas de diagnóstico. Fórmulas, imagens e tabelas não são interpretadas. O score `bm25()` não é probabilidade de suporte, e ter match lexical não significa ter evidência suficiente. Não há geração de resposta nem abstention. Interface de chat, agentes, múltiplos serviços e implantação não fazem parte desta etapa.
