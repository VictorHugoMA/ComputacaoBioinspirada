import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

np.random.seed(12020)

def stepFunction(x):
    return 1 if x >= 0 else 0

def classificaAmostra(X, peso):
    numAmostras = X.shape[0]
    previsao = []
    for i in range(numAmostras):
        z = np.dot(X[i], peso)
        output = stepFunction(z)
        previsao.append(output)
    return previsao

def avaliar_cromossomo(cromossomo, X_treino, y_treino):
    peso_inicial = cromossomo[:-1]

    # Avaliar o desempenho do Perceptron treinado
    previsoes = classificaAmostra(X_treino, peso_inicial)
    acuracia = np.mean(previsoes == y_treino)

    return acuracia


def treinoPerceptronComAG(X_treino, y_treino, tamanhoPopulacao, taxaMutacao, numGeracoes,taxacross):
    tc = X_treino.shape[1] + 1  # Tamanho do cromossomo
    ng = numGeracoes

    # Função para avaliar a aptidão de um cromossomo
    def fitness(cromossomo):
        return avaliar_cromossomo(cromossomo, X_treino, y_treino)

    # Algoritmo evolutivo
    def algoritmo_evolutivo(tp, tc, ng, taxamut, taxacross):
        # Cria uma população de indivíduos com valores reais entre 0 e 1
        pop = np.random.uniform(low=-1, high=1, size=(tp, tc))
        # Calcula a aptidão de cada indivíduo
        fitness_values = np.apply_along_axis(fitness, axis=1, arr=pop)
        print(f"{'Geração': <10}|{'Melhor': <10}|{'Média': <10}|{'Pior': <10}|")
        print(
          f"{0: <10}|{np.max(fitness_values):.9f}|{np.mean(fitness_values):.9f}|{np.min(fitness_values):.9f}|"
      )
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
                f1 = np.clip(f1, -1, 1)
                f2 = np.clip(f2, -1, 1)

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

            print(
              f"{i+1: <10}|{np.max(fitness_values):.9f}|{np.mean(fitness_values):.9f}|{np.min(fitness_values):.9f}|"
          )
        # Retorna o melhor indivíduo (pesos otimizados)
        melhor_individuo = pop[np.argmax(fitness_values)]

        return melhor_individuo

    # Chama o método de algoritmo evolutivo
    melhor_individuo = algoritmo_evolutivo(tamanhoPopulacao, tc, ng, taxaMutacao, taxacross)

    # Obtém os pesos otimizados do melhor indivíduo
    pesos_otimizados = melhor_individuo[:-1]

    return pesos_otimizados
# Características: comprimento das sépalas, largura das sépalas, comprimento das pétalas, largura das pétalas
tamanhoPopulacao = 10
taxaMutacao = 0.1
numGeracoes = 100
taxaCross = 0.2
tamTeste = 0.3

iris = load_iris()
X = iris.data
y = iris.target  # Vetor original de classes (0, 1, 2)

# Filtrar apenas as classes Iris-setosa (0) e Iris-versicolor (1)
indices_setosa_versicolor = np.where((y == 0) | (y == 1))[0]
X = X[indices_setosa_versicolor]
y = y[indices_setosa_versicolor]

# Transformar as classes para 0 (Iris-setosa) e 1 (Iris-versicolor)
y = (y == 1).astype(int)

# Dividir o conjunto de dados em treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=tamTeste)

# Treinar o Perceptron com AG
pesos_otimizados = treinoPerceptronComAG(
    X_treino, y_treino, tamanhoPopulacao, taxaMutacao, numGeracoes,taxaCross)

# Classificar os dados de teste
previsoes = classificaAmostra(X_teste, pesos_otimizados)

# Avaliar o desempenho do Perceptron treinado
acuracia = np.mean(previsoes == y_teste)
print(f"Acuracia do Perceptron treinado com AG: {acuracia:.2f}")