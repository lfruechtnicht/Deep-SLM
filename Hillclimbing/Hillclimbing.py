from utils import check_random_state
from algorithem.Metric import Metric


class Hillclimbing():#todo ABC META

    def __init__(self, seed, neighbourhood_size, stopping_criterion):
        self.seed = check_random_state(seed)
        self.elite = None
        self.random_state = check_random_state(seed)
        self.neighborhood = list()
        self.neighborhood_size = neighbourhood_size
        self.stopping_criterion = stopping_criterion
        self.current_generation = 0


    def fit(self, x_train, y_train, metric):
        pass

    def predict(self, data):
        return self.elite.predict(data)

    def _get_elite(self):

        return

    def _generate_neigbourhood(self):
        pass





