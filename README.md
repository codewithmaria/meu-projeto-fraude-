# Detecção de Fraude com Redes Neurais

Este projeto utiliza **Deep Learning** para identificar transações financeiras fraudulentas. O modelo foi desenvolvido em Python, utilizando a biblioteca **TensorFlow/Keras**, com o objetivo de automatizar a segurança em operações digitais e minimizar riscos financeiros.

## Sobre o Projeto
O sistema analisa padrões de comportamento em transações (como valores de transferência e saldos de contas de origem e destino) para classificar se uma operação é legítima ou uma tentativa de fraude. 

Este projeto foi estruturado de forma **modular**, seguindo boas práticas de engenharia de software. Isso permite que o processamento de dados e a arquitetura da IA sejam mantidos e atualizados de forma independente.

## Estrutura do Repositório
A organização das pastas reflete uma estrutura profissional de projeto de Data Science:

*   **`main.py`**: O "maestro" do projeto. Coordena o fluxo, executa o treinamento e gera os relatórios de acurácia.
*   **`src/data_loader.py`**: Responsável pelo consumo de dados via API do Kaggle, limpeza e codificação de variáveis categóricas.
*   **`src/model_setup.py`**: Contém a arquitetura da Rede Neural Sequencial (Dense Layers).
*   **`requirements.txt`**: Arquivo com todas as dependências necessárias para replicar o ambiente.

## 🛠️ Tecnologias Utilizadas
*   **Linguagem**: Python
*   **IA/ML**: TensorFlow, Keras, Scikit-learn
*   **Análise de Dados**: Pandas, Numpy
*   **Visualização**: Matplotlib, Seaborn, Mlxtend

## 🚀 Como Executar
1. Clone este repositório em sua máquina:
   ```bash
   git clone [https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO.git](https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO.git)
