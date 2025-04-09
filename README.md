KNN Customizado e Comparação com Scikit-Learn

Este repositório apresenta uma implementação do algoritmo **KNN** (K-Nearest Neighbors) usando uma abordagem customizada (do zero) e sua comparação com a implementação disponível na biblioteca **Scikit-Learn**, utilizando a base de dados *Iris*. O código realiza a preparação dos dados, a divisão em conjuntos de treino (70%) e teste (30%), a classificação para diferentes valores de *k* (1, 3, 5, 7) e a exibição de métricas de desempenho e gráficos comparativos (acurácia, precisão, revocação e tempo de execução).

## Tabela de Conteúdo

- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Execução do Código](#execução-do-código)
- [Estrutura de Diretórios](#estrutura-de-diretórios)
- [Observações Finais](#observações-finais)

## Pré-requisitos

- **Python 3.6 ou superior**  
- As seguintes bibliotecas Python:
  - [pandas](https://pandas.pydata.org/)
  - [numpy](https://numpy.org/)
  - [matplotlib](https://matplotlib.org/)
  - [scikit-learn](https://scikit-learn.org/)
- O arquivo `Iris.csv` deve estar presente no diretório do script (ou o caminho para o arquivo deverá ser ajustado no código).

## Instalação

1. **Clonando o Repositório**  
   Utilize o comando abaixo para clonar o repositório:
   ```bash
   git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git

	Criando um Ambiente Virtual (Opcional, mas Recomendado)
No Linux ou macOS:

python3 -m venv venv
source venv/bin/activate

python -m venv venv
venv\Scripts\activate

3.	Instalando as Dependências
Com o ambiente virtual ativado (ou não), instale as bibliotecas necessárias executando:

pip install pandas numpy matplotlib scikit-learn


Caso prefira utilizar um arquivo de requisitos, crie um arquivo chamado requirements.txt com o seguinte conteúdo:

pandas
numpy
matplotlib
scikit-learn

E Excute:
pip install -r requirements.txt

Execução do Código
	1.	Verifique o arquivo Iris.csv:
Certifique-se de que o arquivo Iris.csv esteja no mesmo diretório do script ou ajuste o caminho dentro do código conforme necessário.
	2.	Execute o script:
No terminal, execute o arquivo Python (substitua nome_do_script.py pelo nome do seu arquivo):

python nome_do_script.py

Durante a execução, o script:
	•	Carrega e prepara os dados da base Iris.
	•	Divide o conjunto de dados em treino e teste.
	•	Implementa o algoritmo KNN de forma customizada para os valores de k definidos e exibe suas métricas e matrizes de confusão.
	•	Utiliza a implementação do KNN disponível no Scikit-Learn e realiza a comparação.
	•	Gera gráficos comparativos para as métricas de desempenho e tempo de execução entre as duas abordagens.

Estrutura de Diretórios
├── README.md
├── Iris.csv
└── nome_do_script.py

Observações Finais
	•	Visualização dos Gráficos:
Os gráficos serão exibidos utilizando a biblioteca matplotlib. Certifique-se de que sua janela de visualização esteja habilitada para a exibição de gráficos.
