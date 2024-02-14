import numpy as np
from sklearn.model_selection import train_test_split

def stepFunction(x):
    return 1 if x >= 0 else 0

# Função para avaliar a aptidão de um cromossomo (precisão do Perceptron)
def avaliar_cromossomo(cromossomo, X, y):
    peso_inicial = cromossomo[:-1]
    vies_inicial = cromossomo[-1]

    numAmostras = X.shape[0]
    previsao = np.zeros(numAmostras)

    for i in range(numAmostras):
        z = np.dot(X[i], peso_inicial) + vies_inicial
        output = stepFunction(z)
        previsao[i] = output

    # Calcular a precisão (acurácia) do Perceptron
    precisao = np.mean(previsao == y)
    return precisao

# Algoritmo evolutivo para otimizar os pesos e viés do Perceptron
def algoritmo_genetico(X_treino, y_treino, tp, tc, ng, taxamut, taxacross):
    # Função de aptidão (fitness) para o algoritmo genético
    def fitness(cromossomo):
        return avaliar_cromossomo(cromossomo, X_treino, y_treino)

    # Cria uma população de indivíduos com valores reais entre 0 e 1
    pop = np.random.rand(tp, tc)

    # Calcula a aptidão de cada indivíduo
    fitness_pop = np.apply_along_axis(fitness, axis=1, arr=pop)

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
            pos = np.argsort(fitness_pop)[:2]

            # Se os filhos têm melhor fitness que os dois piores indivíduos, então realiza uma substituição
            if fitness_f1 > fitness_pop[pos[0]]:
                pop[pos[0], :] = f1
                fitness_pop[pos[0]] = fitness_f1
            if fitness_f2 > fitness_pop[pos[1]]:
                pop[pos[1], :] = f2
                fitness_pop[pos[1]] = fitness_f2

    # Encontrar o melhor indivíduo na população final
    melhor_individuo = pop[np.argmax(fitness_pop)]

    return melhor_individuo

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

# Dividir o conjunto de dados em treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3)

# Parâmetros para o algoritmo genético
tp = 10  # Tamanho da população
tc = X.shape[1] + 1  # Tamanho do cromossomo (número de características + 1 para o viés)
ng = 1000  # Número de gerações
taxamut = 0.1  # Taxa de mutação
taxacross = 0.8  # Taxa de crossover

# Executar o algoritmo genético para obter os pesos e viés otimizados
melhor_individuo = algoritmo_genetico(X_treino, y_treino, tp, tc, ng, taxamut, taxacross)

# Extrair os pesos e viés do melhor indivíduo
peso_otimizado = melhor_individuo[:-1]
vies_otimizado = melhor_individuo[-1]

# Treinar o Perceptron com os pesos e viés otimizados
def treinoPerceptronOtimizado(X, y, peso, vies, taxaAprendizado, epocas):
    numCaracteristicas = X.shape[1]
    numAmostras = X.shape[0]

    for _ in range(epocas):
        for i in range(numAmostras):
            # Calcular a saída do perceptron
            z = np.dot(X[i], peso) + vies
            output = stepFunction(z)
            
            # Atualizar pesos e viés
            erro = y[i] - output
            peso += taxaAprendizado * erro * X[i]
            vies += taxaAprendizado * erro

# Parâmetros para o treinamento do Perceptron
taxaAprendizado = 0.1
epocas = 100

# Treinar o Perceptron com os pesos e viés otimizados
treinoPerceptronOtimizado(X_treino, y_treino, peso_otimizado, vies_otimizado, taxaAprendizado, epocas)

# Classificar os dados de teste
previsoes = classificaAmostra(X_teste, peso_otimizado, vies_otimizado)

# Avaliar o desempenho do Perceptron otimizado
acuracia_otimizado = np.mean(previsoes == y_teste)
print(f"Acurácia do Perceptron otimizado: {acuracia_otimizado:.2f}")
