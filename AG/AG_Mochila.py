import random

itens = [(2, 5), (3, 8), (5, 13), (7, 15), (9, 24)]
limiteMochila = 20
tamanhoPopulacao = 50
taxaMutacao = 0.1
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

populacao = [criarIndividuo() for _ in range(tamanhoPopulacao)]

for geracao in range(numGeracoes):
    novaPopulacao = []

    while len(novaPopulacao) < tamanhoPopulacao:
        pai1 = selecao(populacao)
        pai2 = selecao(populacao)
        filho1, filho2 = cruzamento(pai1, pai2)
        mutacao(filho1)
        mutacao(filho2)
        novaPopulacao.extend([filho1, filho2])

    populacao = removerExcesso(novaPopulacao)

melhorIndividuo = max(populacao, key=calcularValor)
valorMelhor = calcularValor(melhorIndividuo)
pesoMelhor = calcularPeso(melhorIndividuo)

print("Melhor solução encontrada:")
print("Cromossomo:", melhorIndividuo)
print("Valor total:", valorMelhor)
print("Peso total:", pesoMelhor)
