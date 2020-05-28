from utils import check_random_state
from algorithem.Metric import _is_better
from sklearn.metrics import accuracy_score

class Hillclimbing():  # todo ABC META

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
        if self.current_generation % 5 == 0 and self.current_generation != 0:
            print("====================   ===================")
            print(f"Generation: {self.current_generation}, {str(self.elite)}, "
                  f"Validation Fitness: {self.elite.evaluate()}")

    def _get_elite(self):

        self.neighborhood.sort(key=lambda x: x.fitness,
                               reverse=self.metric.greater_is_better)  # could do argmax or arg min

        """ TODO: -Lukas _get_elite """
        if self.neighborhood[0].fitness > self.elite.fitness:
            print("Could not decrease loss, best child = %.5f, parent = %.5f" % (
            self.neighborhood[0].fitness, self.elite.fitness))
            child_y_pred_as_labels = self.neighborhood[0].semantics.argmax(axis=1)
            parent_y_pred_as_labels = self.elite.semantics.argmax(axis=1)
            y_train_as_labels = self.neighborhood[0].y_train.argmax(axis=1)
            child_accuracy = accuracy_score(y_train_as_labels, child_y_pred_as_labels) * 100
            parent_accuracy = accuracy_score(y_train_as_labels, parent_y_pred_as_labels) * 100
            print("Accuracy, best child = %.5f, parent = %.5f" % (child_accuracy, parent_accuracy))
            print()

        """ TODO: -Lukas _get_elite, the best child is always kept """
        new_elite = self.neighborhood[0]  # todo - IVO why would you change the elite if the child is acutally worse?

        # =======================================================================
        # if _is_better(self.neighborhood[0].fitness, self.elite.fitness, self.metric):
        #     new_elite = self.neighborhood[0]
        # else:
        #     new_elite = self.elite
        # =======================================================================
        return new_elite
