from utils import check_random_state
from algorithem.Metric import _is_better


class Hillclimbing():#todo ABC META

    def __init__(self, metric, seed, neighbourhood_size, stopping_criterion):
        self.metric = metric
        self.elite = None
        self.random_state = check_random_state(seed)
        self.neighborhood = list()
        self.neighborhood_size = neighbourhood_size
        self.stopping_criterion = stopping_criterion
        self.current_generation = 0

    def predict(self, data):
        return self.elite.predict(data)

    def _verbose_reporter(self):
        print(f"Generation: {self.current_generation}, {str(self.elite)}")
        if self.current_generation % 5 == 0:
            print("====================   ===================")
            print(f"Generation: {self.current_generation}, {str(self.elite)}, "
                  f"Validation Fitness: {self.elite.evaluate()}")

    def _get_elite(self):

        self.neighborhood.sort(key=lambda x: x.fitness, reverse=self.metric.greater_is_better)
        if _is_better(self.neighborhood[0].fitness, self.elite.fitness, self.metric):
            new_elite = self.neighborhood[0]
        else:
            new_elite = self.elite
        return new_elite








