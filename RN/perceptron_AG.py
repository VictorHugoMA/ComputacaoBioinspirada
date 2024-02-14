import numpy as np
from sklearn.model_selection import train_test_split

def stepFunction(x):
    return 1 if x >= 0 else 0

def treinoPerceptron(X, y, peso, vies, taxaAprendizado, epocas):
    numAmostras = X.shape[0]

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

def avaliar_cromossomo(cromossomo, X_treino, y_treino):
    peso_inicial = cromossomo[:-1]
    vies_inicial = cromossomo[-1]

    # Treinar o Perceptron com os pesos iniciais

    # Avaliar o desempenho do Perceptron treinado
    previsoes = classificaAmostra(X_treino, peso_inicial, vies_inicial)
    acuracia = np.mean(previsoes == y_treino)

    return acuracia


def treinoPerceptronComAG(X_treino, y_treino, tamanhoPopulacao, taxaMutacao, numGeracoes):
    tc = X_treino.shape[1] + 1  # Tamanho do cromossomo
    ng = numGeracoes

    # Função para avaliar a aptidão de um cromossomo
    def fitness(cromossomo):
        return avaliar_cromossomo(cromossomo, X_treino, y_treino)

    # Algoritmo evolutivo
    def algoritmo_evolutivo(tp, tc, ng, taxamut, taxacross):
        # Cria uma população de indivíduos com valores reais entre 0 e 1
        pop = np.random.rand(tp, tc)

        # Calcula a aptidão de cada indivíduo
        fitness_values = np.apply_along_axis(fitness, axis=1, arr=pop)

        # Inicia o processo evolutivo
        for i in range(ng):
            # Seleciona dois indivíduos para reprodução
            reprodutor = np.random.choice(tp, size=2, replace=False)

            # Gera um número aleatório e só faz a reprodução se for maior que a taxa de crossover
            if np.random.rand() > taxacross:
                # Aplica o crossover, gerando dois filhos
                alpha = np.random.rand()
                f1 = alpha * pop[reprodutor[0], :] + (1 - alpha) * pop[reprodutor[1], :]
                f2 = alpha * pop[reprodutor[1], :] + (1 - alpha) * pop[reprodutor[0], :]

                # Gera um número aleatório e só faz a mutação se for maior que a taxa de mutação
                if np.random.rand() > taxamut:
                    # Aplica a mutação em cada gene do filho 1
                    f1 += np.random.normal(scale=0.1, size=tc)

                    # Aplica a mutação em cada gene do filho 2
                    f2 += np.random.normal(scale=0.1, size=tc)

                # Garante que os valores permaneçam no intervalo [0, 1]
                f1 = np.clip(f1, 0, 1)
                f2 = np.clip(f2, 0, 1)

                # Calcula o fitness de cada filho
                fitness_f1 = fitness(f1)
                fitness_f2 = fitness(f2)

                # Encontra os dois indivíduos de menor fitness na população
                pos = np.argsort(fitness_values)[:2]

                # Se os filhos têm melhor fitness que os dois piores indivíduos, então realiza uma substituição
                if fitness_f1 > fitness_values[pos[0]]:
                    pop[pos[0], :] = f1
                    fitness_values[pos[0]] = fitness_f1
                if fitness_f2 > fitness_values[pos[1]]:
                    pop[pos[1], :] = f2
                    fitness_values[pos[1]] = fitness_f2

        # Retorna o melhor indivíduo (pesos otimizados)
        melhor_individuo = pop[np.argmax(fitness_values)]

        return melhor_individuo

    # Chama o método de algoritmo evolutivo
    melhor_individuo = algoritmo_evolutivo(tamanhoPopulacao, tc, ng, taxaMutacao, 1.0)

    # Obtém os pesos otimizados do melhor indivíduo
    pesos_otimizados = melhor_individuo[:-1]
    vies_otimizado = melhor_individuo[-1]

    return pesos_otimizados, vies_otimizado

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

# Dividir o conjunto de dados em treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=tamTeste)

# Treinar o Perceptron com AG
pesos_otimizados, vies_otimizado = treinoPerceptronComAG(X_treino, y_treino, tamanhoPopulacao, taxaMutacao, numGeracoes)

# Classificar os dados de teste
previsoes = classificaAmostra(X_teste, pesos_otimizados, vies_otimizado)

# Avaliar o desempenho do Perceptron treinado
acuracia = np.mean(previsoes == y_teste)
print(f"Acuracia do Perceptron treinado com AG: {acuracia:.2f}")
