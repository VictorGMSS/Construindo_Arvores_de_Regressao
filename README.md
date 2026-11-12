# 📦 Precificação de Entrega de Esculturas com Árvores de Regressão

Projeto prático desenvolvido durante o curso **"Regressão: Construindo Árvores de Regressão"** (formação *Machine Learning & Deep Learning*). O objetivo deste repositório é aplicar técnicas de preparação de dados, engenharia de atributos, otimização de hiperparâmetros e validação cruzada para construir modelos preditivos baseados em árvores (`DecisionTreeRegressor` e `RandomForestRegressor`) capazes de estimar o custo de entrega (`custo`) de esculturas.

## 🎯 Objetivos do Projeto
- **Preparação e Engenharia de Atributos:** Conversão de variáveis temporais, cálculo da diferença de dias entre entrega agendada e realizada, e aplicação de codificação *One-Hot* (`pd.get_dummies`) para variáveis categóricas.
- **Modelagem Baseada em Árvores:** Construção e comparação entre árvores de decisão individuais (`DecisionTreeRegressor`) e métodos *ensemble* (`RandomForestRegressor`).
- **Diagnóstico e Mitigação de Overfitting:** Análise de discrepância entre treino e teste em árvores irrestritas e controle de complexidade por meio de poda de hiperparâmetros.
- **Otimização de Hiperparâmetros:** Busca sistemática das melhores combinações de parâmetros (`max_depth`, `min_samples_leaf`, `min_samples_split`, `max_leaf_nodes`) utilizando `GridSearchCV`.
- **Validação Cruzada e Generalização:** Avaliação da estabilidade do modelo com validação cruzada (`KFold` e `cross_validate`) e geração de previsões em dados inéditos.

## 🗂️ Estrutura do Repositório
- `Arvores_de_Regressao_Esculturas.ipynb`: Notebook principal contendo todo o pipeline de pré-processamento, treinamento, visualização de árvores/importância de atributos, otimização com `GridSearchCV` e validação cruzada.
- `entregas.csv`: Base de dados principal para treino e validação (4.462 registros e 19 colunas originais, contendo dimensões físicas, peso, valor da obra, reputação do artista, características logísticas e o alvo `custo`).
- `teste_entregas.csv`: Base de dados de teste sem a variável alvo, utilizada para gerar previsões finais com o melhor modelo treinado.
- `Precos_entregas.csv`: Arquivo de saída gerado pelo pipeline contendo as estimativas finais de custo de envio.

## 🛠️ Tecnologias e Bibliotecas Utilizadas
- **Python 3**
- **Pandas & NumPy:** Manipulação, tratamento de datas e codificação de variáveis categóricas.
- **Matplotlib & Seaborn:** Visualização de dados, estrutura das árvores (`plot_tree`) e importância das variáveis (*Feature Importance*).
- **Scikit-Learn:** Modelagem preditiva (`DecisionTreeRegressor`, `RandomForestRegressor`), otimização (`GridSearchCV`), validação cruzada (`KFold`, `cross_validate`) e métricas de avaliação ($R^2$, MAE, RMSE).

## 🚀 Etapas Desenvolvidas no Notebook
1. **Preparação e Engenharia de Atributos:** Carregamento do dataset `entregas.csv` (4.462 registros), extração de dia, mês e ano das datas, criação da variável `diferenca_dias_entrega`, remoção de colunas identificadoras (`id_cliente`, `nome_artista`) e transformação *One-Hot Encoding*, resultando em 29 variáveis preditoras.
2. **Árvore de Regressão Baseline:** Divisão dos dados em treino (80%) e teste (20%) e treinamento de uma `DecisionTreeRegressor` sem restrições, identificando *overfitting* severo ($R^2 = 1.00$ no treino vs. $0.66$ no teste) e analisando as variáveis mais relevantes (`preco_escultura`, `reputacao_artista` e `preco_base_envio`).
3. **Otimização da Árvore de Regressão:** Ajuste de hiperparâmetros via `GridSearchCV` (`max_depth=10`, `min_samples_leaf=10`, `min_samples_split=2`), reduzindo o sobreajuste e elevando o $R^2$ de teste para `0.7539` (RMSE de `1315.20`).
4. **Floresta Aleatória (Random Forest):** Treinamento do `RandomForestRegressor` com 100 estimadores e avaliação *Out-of-Bag* (`oob_score=True` de `0.8284`), atingindo $R^2$ de teste de `0.8254` e RMSE de `1108.02`.
5. **Refinamento Final e Validação:** Otimização da Floresta Aleatória com `GridSearchCV` (`max_leaf_nodes=550`, `min_samples_leaf=2`, `min_samples_split=2`), alcançando o melhor desempenho geral ($R^2$ de teste de `0.8259`, RMSE de `1106.16` e RMSE médio na validação cruzada de `1143.54`), seguido pela predição e exportação do arquivo `Precos_entregas.csv`.

## 💻 Como Executar
1. **Via Google Colab (Recomendado):** Clique no botão *"Open in Colab"* no topo deste README para abrir e executar o notebook diretamente no navegador sem precisar configurar ambiente local.
2. **Localmente:**
   ```bash
   git clone [https://github.com/VictorGMSS/Precificacao_Entrega_Esculturas.git](https://github.com/VictorGMSS/Precificacao_Entrega_Esculturas.git)
   cd Precificacao_Entrega_Esculturas
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   jupyter notebook Arvores_de_Regressao_Esculturas.ipynb
---

## 👨‍💻 Informações pessoais

Victor Gabriel Martins e Souza

* 💼 LinkedIn: [https://www.linkedin.com/in/victorgms](https://www.linkedin.com/in/victorgms)
* 🌐 Portfólio: [https://verxstech.vercel.app](https://verxstech.vercel.app)
