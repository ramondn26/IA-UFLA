# KNN Customizado e Comparação com Scikit-Learn

Este repositório apresenta uma implementação do algoritmo **KNN** (K-Nearest Neighbors) usando uma abordagem customizada (do zero) e sua comparação com a implementação disponível na biblioteca **Scikit-Learn**, utilizando a base de dados *Iris*.

O código realiza:
- Preparação dos dados
- Divisão em conjuntos de treino (70%) e teste (30%)
- Classificação para diferentes valores de *k* (1, 3, 5, 7)
- Exibição de métricas de desempenho e gráficos comparativos (acurácia, precisão, revocação e tempo de execução)

---

## ✅ Pré-requisitos

- **Python 3.6 ou superior**
- As seguintes bibliotecas Python:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`

> ⚠️ O arquivo `Iris.csv` deve estar presente no diretório do script (ou o caminho para o arquivo deverá ser ajustado no código).

---

## 🛠️ Instalação

### Clonando o Repositório

```bash
git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
```

### Criando um Ambiente Virtual (Opcional, mas Recomendado)

**Linux ou macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

### Instalando as Dependências

```bash
pip install pandas numpy matplotlib scikit-learn
```

Ou crie um arquivo `requirements.txt` com o seguinte conteúdo:

```
pandas
numpy
matplotlib
scikit-learn
```

E execute:

```bash
pip install -r requirements.txt
```

---

## ▶️ Execução do Código

1. **Verifique o arquivo `Iris.csv`:**  
   Certifique-se de que o arquivo `Iris.csv` esteja no mesmo diretório do script ou ajuste o caminho no código.

2. **Execute o script:**

```bash
python nome_do_script.py
```

Durante a execução, o script:
- Carrega e prepara os dados da base *Iris*.
- Divide o conjunto de dados em treino e teste.
- Implementa o algoritmo KNN de forma customizada para os valores de `k` definidos e exibe suas métricas e matrizes de confusão.
- Utiliza a implementação do KNN disponível no **Scikit-Learn** e realiza a comparação.
- Gera gráficos comparativos para as métricas de desempenho e tempo de execução entre as duas abordagens.

---

## 📁 Estrutura de Diretórios

```
├── README.md
├── Iris.csv
└── nome_do_script.py
```

---

## 📝 Observações Finais

**Visualização dos Gráficos:**  
Os gráficos serão exibidos utilizando a biblioteca `matplotlib`. Certifique-se de que sua janela de visualização esteja habilitada para exibir gráficos.
