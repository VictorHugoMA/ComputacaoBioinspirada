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


# Exemplo de uso:
tamanhoPopulacao = 10
taxaMutacao = 0.1
numGeracoes = 50
tamTeste = 0.3

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=tamTeste)



def algoritmoGenetico(tamanhoPopulacao, taxaMutacao, numGeracoes, X_treino, y_treino):
    # Inicialize a população de pesos como cromossomos
    populacao = np.random.rand(tamanhoPopulacao, X_treino.shape[1] + 1)

    for geracao in range(numGeracoes):
        fitness = []

        # Avalie o fitness para cada indivíduo na população
        for i in range(tamanhoPopulacao):
            peso = populacao[i, :-1]
            vies = populacao[i, -1]

            # Avalie o fitness usando o Perceptron
            previsoes = classificaAmostra(X_treino, peso, vies)
            fitness.append(np.mean(previsoes == y_treino))

        # Aplique seleção para escolher os melhores indivíduos
        melhores_indices = np.argsort(fitness)[-tamanhoPopulacao // 2:]
        melhores_populacao = populacao[melhores_indices]

        # Aplique recombinação (crossover)
        nova_populacao = []
        for i in range(tamanhoPopulacao // 2):
            pai1 = melhores_populacao[np.random.randint(0, len(melhores_populacao))]
            pai2 = melhores_populacao[np.random.randint(0, len(melhores_populacao))]

            ponto_corte = np.random.randint(0, len(pai1))
            filho1 = np.concatenate((pai1[:ponto_corte], pai2[ponto_corte:]))
            filho2 = np.concatenate((pai2[:ponto_corte], pai1[ponto_corte:]))

            nova_populacao.append(filho1)
            nova_populacao.append(filho2)

        # Aplique mutação
        for i in range(tamanhoPopulacao):
            if np.random.rand() < taxaMutacao:
                gene_mutante = np.random.randint(0, X_treino.shape[1] + 1)
                nova_populacao[i, gene_mutante] += np.random.normal(0, 0.1)

        # Substitua a antiga população pela nova
        populacao = np.array(nova_populacao)

    # Retorne o melhor indivíduo (pesos otimizados)
    melhor_individuo = populacao[np.argmax(fitness)]

    return melhor_individuo

# Exemplo de uso:
tamanhoPopulacao = 10
taxaMutacao = 0.1
numGeracoes = 50

# Treino Perceptron com o Algoritmo Genético
pesos_otimizados = algoritmoGenetico(tamanhoPopulacao, taxaMutacao, numGeracoes, X_treino, y_treino)

# Utilize os pesos otimizados para classificar outros dados
previsoes = classificaAmostra(X_teste, pesos_otimizados[:-1], pesos_otimizados[-1])

print(previsoes)


            
