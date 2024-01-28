import numpy as np

np.random.seed(57)


class ArtificialBeeColony:
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

        self.func = func # фитнесс функция
        self.colony_size = colony_size # размер популяции
        self.employed_num = int(colony_size * employed_ratio)
        self.onlooker_num = int(colony_size * onlooker_ratio)
        self.scout_num = int(colony_size * scout_ratio)
        self.dim = dim # Количество переменных в векторе xm→
        self.li = li # Верхняя граница
        self.ui = ui # Нижняя границa
        self.max_cycles = max_cycles # количество повроторений
        self.abandonment_limit = abandonment_limit # сколько последовательных циклов решение может оставаться без улучшений перед тем, как оно будет заменено.
        self.population = self.initialize_population()
        self.best_solution = None
        self.best_fitness = float("inf")
        self.limit = np.zeros(self.colony_size)

    def initialize_population(self):
        """
        При инициализации колонии рабочим пчелам присуждается случайная точка в пределах ui-li
        Xmi=li+rand(0,1)∗(ui−li)
        """
        Xmi = self.li + np.random.rand(self.colony_size, self.dim) * (self.ui - self.li)
        return Xmi

    def calculate_fitness(self, solution):
        value = self.func(solution)
        return 1 / (1 + abs(value)) if value >= 0 else 1 + abs(value)

    def employed_bees_phase(self):
        """
        Фаза рабочих пчел: Рабочая пчела сравнивает свою точку
         и случайную точку в пределах точки следующей рабочей пчелы
         Если новая точка содержит больше "нектара" рабочая пчела
         перемещается туда, иначе остается на месте
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
        Пчелы наблюдатели вероятностно выдвигаются к точкам рабочих пчел чтобы произвести анализ
        подобно рабочим пчелам, вероятность движения к точке вычисляется по формуле:
        pm=fitm(xm→)/∑m=1SNfitm(xm→)

        В случае нахождения точки с большим кол-вом "нектара" рабочая пчела закрепленная за точкой перемещается
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
        В случае если рабочая пчела не сменила свою локацию в течение кол-ва циклов равном abandonment_limit
        источник признается "заброшеным" и рабочая пчела переводится в ранг разветчика
        Для разведчика происходит повторная инициализация с присвоением новой случайной точки
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
        """
        Запись промежуточных результатов между циклами
        """
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
    Функция по которой ведется вычисления фитнесс функции
    """
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))


abc = ArtificialBeeColony(
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