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
- `data/`: local reservado a arquivos de corpus mantidos fora do versionamento; somente `.gitkeep` será versionável.
- `evaluation/golden_set.jsonl`: arquivo vazio reservado a um futuro conjunto de avaliação; o contrato das perguntas ainda não foi definido.

## Contratos e dados locais

Os contratos documentais pertencem ao repositório. Os arquivos locais do corpus devem permanecer fora dele.

Os comentários do manifesto descrevem os campos obrigatórios das futuras entradas. Não existe validador automático implementado.

## Estado atual

Existe apenas a fundação documental. O manifesto não cadastra documentos e o conjunto de avaliação não contém perguntas.

As escolhas técnicas e a implementação serão tratadas em missões posteriores.
