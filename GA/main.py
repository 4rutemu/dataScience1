import random

import networkx as nx
from loguru import logger
from networkx import Graph

random.seed(57)


def calculate_distance(graph, route):
    distance = graph.edges[route[-1], route[0]]["weight"]
    for i in range(len(route) - 1):
        weight = graph.edges[route[i], route[i + 1]]["weight"]
        distance += weight
    return distance


class NaturalSelector:
    """
    Класс для проведения турнирного отбора
    """
    def __init__(self, tournament_size):
        self.tournament_size = tournament_size

    def select_parents(self, population, graph):
        tournament = random.sample(population, self.tournament_size)
        return min(
            tournament,
            key=lambda route: calculate_distance(graph, route),
        )


class OrderCrossover:
    """
    Задача решается порядковым методом
    Кроссинговер двухточечный
    """
    @staticmethod
    def crossover(X, Y):
        start, end = sorted(random.sample(range(len(X)), 2))

        # Подмассив между выбранными позициями от первого родителя к дочернему элементу.
        child = [None] * len(X)
        X_subset = X[start:end]
        child[start:end] = X_subset

        # Заполнение оставшихся позиций в дочернем элементе элементами второго родителя.
        idx = end
        Y_head = Y[end:]
        Y_tail = Y[:end]
        Y_subset = Y_head + Y_tail
        for city in Y_subset:
            if None in child:
                if city not in child:
                    child_idx = idx % len(X)
                    child[child_idx] = city
                    idx += 1

        return child


class MutationOperator:
    """
    В качестве мутации происходит смена двух городов местами
    """
    @staticmethod
    def mutate(individ):
        idx1, idx2 = random.sample(range(len(individ)), 2)
        individ[idx1], individ[idx2] = individ[idx2], individ[idx1]
        return individ


class GeneticAlgorithm:
    def __init__(
        self,
        selector,
        crossover_operator,
        mutation_operator,
    ):
        self.selector = selector
        self.crossover_operator = crossover_operator
        self.mutation_operator = mutation_operator

    def run(self, pop_size, generations, crossover_rate, mutation_rate, graph):
        population = []

        # Инициализация первой популяции
        for _ in range(pop_size):
            route = list(graph.nodes)
            random.shuffle(route)
            population.append(route)

        for g in range(generations):
            new_population = []

            for p in range(pop_size // 2):
                X = self.selector.select_parents(population, graph)
                Y = self.selector.select_parents(population, graph)

                # Выполнение кроссинговера с некоторой вероятностью
                XY = (
                    self.crossover_operator.crossover(X, Y)
                    if random.random() < crossover_rate
                    else X
                )
                YX = (
                    self.crossover_operator.crossover(Y, X)
                    if random.random() < crossover_rate
                    else Y
                )

                # Выполнение мутации с некоторой вероятностью
                for child in [XY, YX]:
                    if random.random() < mutation_rate:
                        new_population.extend([self.mutation_operator.mutate(child)])
                    else:
                        new_population.extend([child])
            population = new_population
            best_route = min(
                population,
                key=lambda route: calculate_distance(G, route),
            )
            logger.info(f"Generation: {g}. Best route: {best_route}")

        return population


if __name__ == "__main__":
    cities_count = 20
    G: Graph = nx.complete_graph(range(cities_count))
    for edge in G.edges():
        G.edges[edge]["weight"] = random.randint(1, cities_count)

    natural_selector = NaturalSelector(
        tournament_size=3,
    )

    genetic_algorithm = GeneticAlgorithm(
        selector=natural_selector,
        crossover_operator=OrderCrossover(),
        mutation_operator=MutationOperator(),
    )

    population = genetic_algorithm.run(
        pop_size=100,
        generations=1000,
        crossover_rate=0.9,
        mutation_rate=0.1,
        graph=G,
    )

    best_route = min(
        population,
        key=lambda route: calculate_distance(G, route),
    )
    best_route_total_distance = calculate_distance(G, best_route)

    logger.info(f"Edges: {G.edges}")
    logger.info(f"Best Route: {best_route}")
    logger.info(f"Total Distance: {best_route_total_distance}")