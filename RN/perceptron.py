import numpy as np
from sklearn.model_selection import train_test_split

def stepFunction(x):
    return 1 if x >= 0 else 0

def treinoPerceptron(X, y, taxaAprendizado, epocas):
    numCaracteristicas = X.shape[1]
    numAmostras = X.shape[0]
    peso = np.random.rand(numCaracteristicas)
    vies = np.random.rand()
    
    for _ in range(epocas):
        for i in range(numAmostras):
            # Calcular a saída do perceptron
            z = np.dot(X[i], peso) + vies
            output = stepFunction(z)
            
            # Atualizar pesos e vies
            erro = y[i] - output
            peso += taxaAprendizado * erro * X[i]
            vies += taxaAprendizado * erro
    
    return peso, vies

def classificaAmostra(X, peso, vies):
    numAmostras = X.shape[0]
    previsao = []
    for i in range(numAmostras):
        z = np.dot(X[i], peso) + vies
        output = stepFunction(z)
        previsao.append(output)
    return previsao

# Características: comprimento das sépalas, largura das sépalas, comprimento das pétalas, largura das pétalas
X = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [4.7, 3.2, 1.3, 0.2],
    [4.6, 3.1, 1.5, 0.2],
    [5.0, 3.6, 1.4, 0.2],
    [5.4, 3.9, 1.7, 0.4],
    [4.6, 3.4, 1.4, 0.3],
    [5.0, 3.4, 1.5, 0.2],
    [4.4, 2.9, 1.4, 0.2],
    [4.9, 3.1, 1.5, 0.1],
    [5.4, 3.7, 1.5, 0.2],
    [4.8, 3.4, 1.6, 0.2],
    [4.8, 3.0, 1.4, 0.1],
    [4.3, 3.0, 1.1, 0.1],
    [5.8, 4.0, 1.2, 0.2],
    [5.7, 4.4, 1.5, 0.4],
    [5.4, 3.9, 1.3, 0.4],
    [5.1, 3.5, 1.4, 0.3],
    [5.7, 3.8, 1.7, 0.3],
    [5.1, 3.8, 1.5, 0.3],
    [5.4, 3.4, 1.7, 0.2],
    [5.1, 3.7, 1.5, 0.4],
    [4.6, 3.6, 1.0, 0.2],
    [5.1, 3.3, 1.7, 0.5],
    [4.8, 3.4, 1.9, 0.2],
    [5.0, 3.0, 1.6, 0.2],
    [5.0, 3.4, 1.6, 0.4],
    [5.2, 3.5, 1.5, 0.2],
    [5.2, 3.4, 1.4, 0.2],
    [4.7, 3.2, 1.6, 0.2],
    [7.0, 3.2, 4.7, 1.4],
    [6.4, 3.2, 4.5, 1.5],
    [6.9, 3.1, 4.9, 1.5],
    [5.5, 2.3, 4.0, 1.3],
    [6.5, 2.8, 4.6, 1.5],
    [5.7, 2.8, 4.5, 1.3],
    [6.3, 3.3, 4.7, 1.6],
    [4.9, 2.4, 3.3, 1.0],
    [6.6, 2.9, 4.6, 1.3],
    [5.2, 2.7, 3.9, 1.4],
    [5.0, 2.0, 3.5, 1.0],
    [5.9, 3.0, 4.2, 1.5],
    [6.0, 2.2, 4.0, 1.0],
    [6.1, 2.9, 4.7, 1.4],
    [5.6, 2.9, 3.6, 1.3],
    [6.7, 3.1, 4.4, 1.4],
    [5.6, 3.0, 4.5, 1.5],
    [5.8, 2.7, 4.1, 1.0],
    [6.2, 2.2, 4.5, 1.5],
    [5.6, 2.5, 3.9, 1.1],
    [5.9, 3.2, 4.8, 1.8],
    [6.1, 2.8, 4.0, 1.3],
    [6.3, 2.5, 4.9, 1.5],
    [6.1, 2.8, 4.7, 1.2],
    [6.4, 2.9, 4.3, 1.3],
    [6.6, 3.0, 4.4, 1.4],
    [6.8, 2.8, 4.8, 1.4],
    [6.7, 3.0, 5.0, 1.7],
    [6.0, 2.9, 4.5, 1.5],
    [5.7, 2.6, 3.5, 1.0],
    [5.5, 2.4, 3.8, 1.1]
])

# Classes (0 para Iris-setosa, 1 para Iris-versicolor)
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

"""
taxaAprendizado = 0.1
epocas = 100

 # Treinar o Perceptron
pesoTreinado, viesTreinado = treinoPerceptron(X, y, taxaAprendizado, epocas)

# Dados para classificar
X_teste = np.array([ [5.9,3.0,5.1,1.8],
                     [6.2,3.4,5.4,2.3],
                     [6.5,3.0,5.2,2.0],
                   ])

# Classificar os dados de teste
previsao = classificaAmostra(X_teste, pesoTreinado, viesTreinado)

# Previsões (0 para Iris-setosa, 1 para Iris-versicolor)
print(previsao) """

 # Taxas de aprendizado e épocas para testar
taxaAprendizado = [0.01, 0.1, 0.2]
epocas = [10, 50, 100]
tamTeste = 0.3

print(f"Tamanho do conjunto de teste: {tamTeste*100}%")
# Dividir o conjunto de dados em treinamento e teste
for taxa in taxaAprendizado:
    for num_epocas in epocas:
        print(f"Taxa de Aprendizado: {taxa}, Epocas: {num_epocas}")
        for _ in range(5):  # Realize 5 testes diferentes
            X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=tamTeste)

            peso, vies = treinoPerceptron(X_treino, y_treino, taxa, num_epocas)
            previsoes = classificaAmostra(X_teste, peso, vies)

            # Avaliar o desempenho do Perceptron
            acuracia = np.mean(previsoes == y_teste)
            print(f"Acuracia: {acuracia:.2f}")
