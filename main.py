import numpy as np

np.random.seed(57)


class ArtificialBee:
    def __init__(
            self,
            func,
            colony_size,
            employed_ratio,
            onlooker_ratio,
            scout_ratio,
            dim,
            li,
            ui,
            max_cycles,
            abandonment_limit,
    ):
        """
        [limit] используется для отслеживания количества последовательных циклов,
        в течение которых решение конкретной пчелы не улучшилось.
        Помогает предотвратить застревание алгоритма в локальных оптимумах,
        поощряя исследование новых решений, когда определенное решение
        не улучшается в течение определенного числа циклов `abandonment_limit`
        :param func: фитнесс функция
        :param colony_size: размер популяции
        :param dim: Количество переменных в векторе xm→
        :param li: Верхняя граница
        :param ui: Нижняя границв
        :param max_cycles: количество повроторений
        :param abandonment_limit: сколько последовательных циклов решение может оставаться без улучшений перед тем, как оно будет заменено.
        """
        self.func = func
        self.colony_size = colony_size
        self.employed_num = int(colony_size * employed_ratio)
        self.onlooker_num = int(colony_size * onlooker_ratio)
        self.scout_num = int(colony_size * scout_ratio)
        self.dim = dim
        self.li = li
        self.ui = ui
        self.max_cycles = max_cycles
        self.abandonment_limit = abandonment_limit
        self.population = self.initialize_population()
        self.best_solution = None
        self.best_fitness = float("inf")
        self.limit = np.zeros(self.colony_size)

    def initialize_population(self):
        """
        Все векторы популяции источников питания, xm→’s, инициализируются (m=1...SN, SN: размер популяции)
        пчелами-разведчиками и задаются контрольные параметры.
        Поскольку каждый источник пищи, xm→, является вектором решения задачи оптимизации,
        каждый вектор xm→ содержит n переменных (xmi, i=1...n),
        которые необходимо оптимизировать так, чтобы минимизировать целевую функцию.

        Xmi=li+rand(0,1)∗(ui−li)
        """
        Xmi = self.li + np.random.rand(self.colony_size, self.dim) * (self.ui - self.li)
        return Xmi

    def calculate_fitness(self, solution):
        value = self.func(solution)
        return 1 / (1 + abs(value)) if value >= 0 else 1 + abs(value)

    def employed_bees_phase(self):
        """
        Работающие пчелы ищут новые источники пищи (υm→), имеющие в своей памяти больше нектара
        поблизости от источника пищи (xm→).
        Они находят соседний источник пищи и затем оценивают его рентабельность (пригодность).
        Например, они могут определить соседний источник пищи υm→, используя формулу, представленную уравнением (6):

        υmi=xmi+ϕmi(xmi−xki)

        """
        for i in range(self.colony_size):
            employed_bee = self.population[i, :]
            neighbour_index = np.random.randint(0, self.colony_size)
            phi = np.random.uniform(-1, 1)
            neighbour_bee = employed_bee + phi * (
                    employed_bee - self.population[neighbour_index, :]
            )
            neighbour_bee_fitness = self.calculate_fitness(neighbour_bee)
            if neighbour_bee_fitness < self.calculate_fitness(employed_bee):
                self.population[i, :] = neighbour_bee
                self.limit[i] = 0
            else:
                self.limit[i] += 1

    def onlooker_bees_phase(self):
        """
        Работающие пчелы делятся информацией о своих источниках пищи с пчелами-наблюдателями, ожидающими в улье,
        а затем пчелы-наблюдатели вероятностно выбирают источники пищи в зависимости от этой информации.

        В ABC пчела-наблюдатель выбирает источник пищи в зависимости от значений вероятности,
        рассчитанных с использованием значений приспособленности, предоставленных рабочими пчелами.

        Для этой цели можно использовать метод отбора на основе приспособленности,
        такой как метод выбора колеса рулетки (Goldberg, 1989).

        Значение вероятности pm, с которым пчела-наблюдатель выбирает xm→,
        можно рассчитать с помощью выражения, приведенного в уравнении (8):

        pm=fitm(xm→)/∑m=1SNfitm(xm→)
        """
        fitness_values = np.array(
            [self.calculate_fitness(bee) for bee in self.population]
        )
        probabilities = fitness_values / np.sum(fitness_values)

        for i in range(self.colony_size):
            chosen_index = np.random.choice(
                np.arange(self.colony_size), p=probabilities
            )
            onlooker_bee = self.population[chosen_index, :]
            phi = np.random.uniform(-1, 1)
            neighbour_index = np.random.randint(0, self.colony_size)
            neighbour_bee = onlooker_bee + phi * (
                    onlooker_bee - self.population[neighbour_index, :]
            )
            neighbour_bee_fitness = self.calculate_fitness(neighbour_bee)

            if neighbour_bee_fitness < self.calculate_fitness(onlooker_bee):
                self.population[chosen_index, :] = neighbour_bee
                self.limit[chosen_index] = 0
            else:
                self.limit[chosen_index] += 1

    def scout_bees_phase(self):
        """
        Если limit для пчелы превышает abandonment_limit, это означает, что решение,
        связанное с этой пчелой, не улучшалось в течение определенного количества последовательных циклов.
        В таком случае пчела "обнуляется" и становится разведчиком,
        и для нее генерируется новое случайное решение.
        """
        for i in range(self.colony_size):
            if self.limit[i] >= self.abandonment_limit:
                self.population[i, :] = self.li + np.random.rand(self.dim) * (
                        self.ui - self.li
                )
                self.limit[i] = 0
            else:
                self.limit[i] += 1

    def memorize_best_solution(self):
        fitness_values = [self.calculate_fitness(bee) for bee in self.population]
        min_index = np.argmin(fitness_values)
        current_best_fitness = fitness_values[min_index]

        if current_best_fitness < self.best_fitness:
            self.best_solution = np.copy(self.population[min_index, :])
            self.best_fitness = current_best_fitness

    def run(self):
        cycle = 0

        while cycle < self.max_cycles:
            self.employed_bees_phase()
            self.onlooker_bees_phase()
            self.scout_bees_phase()
            self.memorize_best_solution()

            print(
                f"Cycle {cycle + 1}: Best Solution = {self.best_solution}, Best Fitness = {self.best_fitness}"
            )

            cycle += 1


def rastrigin_function(x, A=10):
    """
    Глобальный минимум в нулевой точке x=(0,0,…,0) с минимальным значением f(x)=0
    """
    return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


if __name__ == "__main__":
    abc = ArtificialBee(
        func=rastrigin_function,
        colony_size=20,
        employed_ratio=0.5,
        onlooker_ratio=0.5,
        scout_ratio=0.2,
        dim=1,
        li=-600,
        ui=600,
        max_cycles=15,
        abandonment_limit=4,
    )
    abc.run()
