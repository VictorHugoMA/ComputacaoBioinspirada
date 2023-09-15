import random
import time

itens = [(5094, 3485), (6506, 326), (416, 5248), (4992, 2421), (4649, 322), (5237, 795), (1457, 3043), (4815, 845), (4446, 4955), (5422, 2252), (2791, 2009), (3359, 6901), (3667, 6122), (1598, 5094), (3007, 738), (3544, 4574), (6334, 3715), (766, 5882), (3994, 5367), (1893, 1984)]
limiteMochila = 7001
tamanhoPopulacao = 50
taxaMutacao = 0.3
numGeracoes = 100

def criarIndividuo():
    individuo = [random.randint(0, 1) for _ in range(len(itens))]
    while calcularPeso(individuo) > limiteMochila:
        individuo = [random.randint(0, 1) for _ in range(len(itens))]
    return individuo

def calcularValor(individuo):
    return sum(individuo[i] * itens[i][1] for i in range(len(individuo)))

def calcularPeso(individuo):
    return sum(individuo[i] * itens[i][0] for i in range(len(individuo)))

def selecao(populacao, k=3):
    if len(populacao) < k:
        return None
    torneio = random.sample(populacao, k)
    return max(torneio, key=calcularValor)

def cruzamento(pai1, pai2):
    pontoCorte1 = random.randint(1, len(pai1) - 1)
    pontoCorte2 = random.randint(1, len(pai1) - 1)

    if pontoCorte1 > pontoCorte2:
        pontoCorte1, pontoCorte2 = pontoCorte2, pontoCorte1

    filho1 = pai1[:pontoCorte1] + pai2[pontoCorte1:pontoCorte2] + pai1[pontoCorte2:]
    filho2 = pai2[:pontoCorte1] + pai1[pontoCorte1:pontoCorte2] + pai2[pontoCorte2:]

    if calcularPeso(filho1) > limiteMochila:
        for i in range(len(filho1)):
            if calcularPeso(filho1) > limiteMochila and filho1[i] == 1:
                filho1[i] = 0
    if calcularPeso(filho2) > limiteMochila:
        for i in range(len(filho2)):
            if calcularPeso(filho2) > limiteMochila and filho2[i] == 1:
                filho2[i] = 0

    return filho1, filho2

def mutacao(individuo):
    for i in range(len(individuo)):
        if random.random() < taxaMutacao:
            individuo[i] = 1 - individuo[i]

def removerExcesso(populacao):
    return [individuo for individuo in populacao if calcularPeso(individuo) <= limiteMochila]

def elitismo(populacao, num_elitismo):
    sorted_populacao = sorted(populacao, key=calcularValor, reverse=True)
    return sorted_populacao[:num_elitismo]

def main():
    if tamanhoPopulacao < 2:
        print("Tamanho da população muito pequeno. Deve ser pelo menos 2.")
        return

    populacao = [criarIndividuo() for _ in range(tamanhoPopulacao)]

    for geracao in range(numGeracoes):
        novaPopulacao = []

        elite = elitismo(populacao, num_elitismo=10)
        novaPopulacao.extend(elite)

        while len(novaPopulacao) < tamanhoPopulacao - len(elite):
            pai1 = selecao(populacao)
            if pai1 is None:
                print("Não há indivíduos suficientes para formar o torneio. Encerrando o algoritmo.")
                return
            pai2 = selecao(populacao)
            if pai2 is None:
                print("Não há indivíduos suficientes para formar o torneio. Encerrando o algoritmo.")
                return
            filho1, filho2 = cruzamento(pai1, pai2)
            mutacao(filho1)
            mutacao(filho2)
            novaPopulacao.extend([filho1, filho2])

        populacao = removerExcesso(novaPopulacao)

    melhorIndividuo = max(populacao, key=calcularValor)
    valorMelhor = calcularValor(melhorIndividuo)
    print("Valor total:", valorMelhor)

start_time = time.time()
main()
execution_time = time.time() - start_time
print(f"AG - Execution time: {execution_time} seconds")
