# Arquitetura do TraceRAG

Arquitetura planejada — não implementada.

## Objetivo e princípios

O objetivo futuro é oferecer um sistema RAG local-first para documentação pública de metrologia, com respostas fundamentadas em evidências e referências às fontes.

A execução local é uma orientação de projeto. Somente fontes públicas poderão ser consideradas, com suas condições de uso verificadas antes de qualquer inclusão.

## Fluxo conceitual futuro

Documentos públicos → preparação local → consulta às evidências → resposta com referências às fontes.

Esse fluxo descreve uma intenção de arquitetura. Nenhuma dessas etapas está implementada nesta versão.

## Organização inicial

- `corpus_manifest.yaml`: contrato documental para futuras entradas do corpus; contém uma versão de esquema e uma lista vazia de documentos.
- `data/`: diretório dentro do working tree reservado aos arquivos locais do corpus, que devem ficar fora do versionamento Git; a exceção prevista no `.gitignore` é `data/.gitkeep`.
- `evaluation/golden_set.jsonl`: arquivo vazio reservado a um futuro conjunto de avaliação; o contrato das perguntas ainda não foi definido.

## Contratos e dados locais

Os contratos documentais são versionados no repositório. Os arquivos locais do corpus ficam sob `data/`, dentro do working tree, mas não devem ser adicionados ao índice nem incluídos em commits. A regra de ignore não representa garantia sobre capturas ou cópias feitas por outras ferramentas.

### Caminho local

`local_path` é relativo à raiz do repositório, nunca ao próprio diretório `data/`. Deve começar por `data/`, usar `/` como separador e identificar um arquivo dentro desse diretório. Não são aceitos caminhos absolutos, prefixos de unidade, barras invertidas, segmentos vazios, `.` ou `..`, barra final ou o caminho `data/.gitkeep`.

Ao acessar o arquivo, o caminho resolvido deverá permanecer dentro de `data/`, inclusive quando houver links simbólicos ou junções no sistema de arquivos. Essas condições são requisitos documentados para etapas futuras; não existe verificação automática implementada nesta versão.

### Registro das condições de uso

Além de `id`, `title`, `source_url` e `local_path`, cada entrada futura deverá conter o objeto obrigatório `usage_review`:

- `reviewed_on`: data válida da análise realizada, como texto no formato `YYYY-MM-DD`.
- `evidence_urls`: lista não vazia de URLs públicas das condições consultadas, que fundamentam a análise; não basta repetir a URL de origem sem que ela contenha essa evidência.
- `intended_use`: texto não vazio que descreve o uso pretendido no projeto.
- `decision`: texto `approved`, registrado somente após análise favorável para o uso pretendido.
- `notes`: texto não vazio com as condições identificadas, o fundamento da decisão, os requisitos de atribuição e as restrições aplicáveis ao uso pretendido.

Uma entrada só poderá ser cadastrada e o documento incorporado ao corpus depois dessa análise aprovada. Condições incertas ou incompatíveis impedem ambos. A disponibilidade pública, isoladamente, não substitui a análise. Mudanças no uso pretendido ou nas condições aplicáveis exigem nova análise e atualização do registro antes de prosseguir com esse uso.

A escolha da licença do projeto é uma decisão separada das condições de uso dos documentos de terceiros. Esta versão não escolhe licença, não analisa nenhuma fonte e não concede autorização de uso por meio do campo `decision`.

### Versão e aplicação do contrato

`schema_version: 2` identifica a revisão documental que define a base de `local_path` e acrescenta `usage_review` aos campos obrigatórios. A versão anterior continha `documents: []`; nenhuma entrada foi migrada ou criada nesta revisão.

Os comentários do manifesto e esta documentação definem os requisitos das futuras entradas. Não existe validador automático implementado. O manifesto continua vazio, com `documents: []`.

## Estado atual

Existe apenas a fundação documental. O manifesto não cadastra documentos e o conjunto de avaliação não contém perguntas.

As escolhas técnicas e a implementação serão tratadas em missões posteriores.
