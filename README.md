# Previsão da Taxa de Sobrevivência em Células Hepáticas

O **Previsor de Criopreservação** é um projeto da Fundação Oswaldo Cruz que utiliza técnicas de Machine Learning para prever a taxa de sobrevivência em células hepáticas submetidas a criopreservantes. O objetivo é desenvolver um modelo preditivo que possa ajudar a otimizar processos de criopreservação em aplicações biomédicas e tornar essa tecnologia de fácil acesso por meio de uma interface web.

## Bibliotecas Utilizadas

O projeto é desenvolvido em Python e utiliza as seguintes bibliotecas:

- **ipywidgets**: Para criar interfaces interativas no Jupyter Notebook.
- **jupyter e jupyterlab**: Ambientes de desenvolvimento para notebooks.
- **lazypredict**: Para facilitar a experimentação com diversos modelos de machine learning.
- **lightgbm**: Um algoritmo de boosting que é eficiente e rápido.
- **matplotlib**: Para visualização de dados.
- **numpy**: Para operações numéricas e manipulação de arrays.
- **pandas**: Para manipulação e análise de dados.
- **pyeasyga**: Para algoritmos genéticos.
- **scikit-learn**: Biblioteca de aprendizado de máquina para modelos e validação.
- **scipy**: Para computação científica e manipulação de dados.
- **shap**: Para interpretação de modelos preditivos.
- **xgboost**: Algoritmo de boosting que melhora a performance de modelos.
- **tabulate**: Para exibir dados em formato tabular.
- **flask**: Para criar uma aplicação web que possa interagir com o modelo.

## Instalação

Para instalar as dependências necessárias, você pode usar o `pip`. Execute o seguinte comando:

```bash
pip install -r requirements.txt

## Minhas Modificações

### Pontos importantes

- *Base de Dados:* Passei a utilizar somente o arquivo "hepg2.csv" como a fonte principal de dados, misturando os dados das células hepáticas de animais e humanos, com colunas como % DMSO, % ANTES DO CONGELAMENTO, % APÓS O DESCONGELAMENTO, etc.
- *Modelo:* O algoritmo Random Forest substituiu o algoritmo genético.
- *Ajustar de Filtro*: Filtrei apenas as colunas que são relevantes para o DMSO.
- *Treinamento de um modelo de Random Forest:* Usei a classe Model existente para implementar o Random Forest. E removi o código do Algoritmo Genético
- *Ajustes nos scripts* **feature_engineering.py:** Garanti que apenas as colunas relevantes ao DMSO sejam consideradas. **model.py:** Confirmei que o RandomForestRegressor está corretamente implementado e priorizado para treinamento. **Hepatocytes Cryopreservation Solution Optimization.ipynb:** Removi o código relacionado ao algoritmo genético e atualizei para usar o RandomForestRegressor.







