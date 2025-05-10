# Word2Vec Streamlit Clustering App

Este projeto explora o uso de Word2Vec nas arquiteturas CBOW e Skip-gram para agrupar frases semelhantes em tempo real, usando uma interface web interativa criada com Streamlit. O gerenciamento de dependências é feito com Poetry, garantindo um ambiente isolado e reproduzível.

## Funcionalidades

- **Testes em Jupyter Notebook**: Experimentos e visualizações em notebooks para validação dos modelos CBOW e Skip-gram.
- **Aplicação Web com Streamlit**: Interface onde o usuário digita frases e visualiza a evolução dos clusters conforme novas entradas.
- **Escolha de Arquitetura**: Alternância entre CBOW e Skip-gram diretamente na interface.
- **Plot Dinâmico**: Representação gráfica interativa dos vetores de frases agrupados em clusters.

## Pré-requisitos

- Git
- Python 3.11 ou superior
- [Poetry](https://python-poetry.org) para gerenciamento de dependências

## Passos para instalação

1. **Clonar o repositório**:

   ```bash
   git clone https://github.com/AndersonIvanildo/exploring-word2vec.git
   cd exploring-word2vec
````

2. **Instalar dependências sem instalar o pacote raiz**:

   ```bash
   poetry install --no-root
   ```

3. **Ativar o ambiente Poetry** (opcional, mas recomendado para desenvolvimento):

   ```bash
   poetry shell
   ```

## Como executar a aplicação

Utilize o comando abaixo para iniciar o servidor Streamlit:

```bash
poetry run streamlit run main.py
```

> **Dica de uso**: ao abrir o navegador, escolha entre **CBOW** e **Skip-gram** no menu lateral. Digite frases no campo de texto e observe como os pontos no gráfico se agrupam dinamicamente.

## Estrutura do projeto

```
├── tools/
│   ├── __init__.py
│   └── process_text.py
├── main.py
├── word2vec-exploration.ipynb
├── pyproject.toml
└── README.md
```

* `word2vec-exploration.ipynb`: Jupyter notebooks com testes e visualizações.
* `main.py`: Código principal da interface Streamlit.
* `pyproject.toml`: Definição de dependências e configurações do Poetry.