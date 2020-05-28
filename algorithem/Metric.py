from numpy import sqrt, mean, square, empty, where
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()


class Metric():
    greater_is_better = None

    @staticmethod
    def evaluate(prediction, target):
        pass


class RootMeanSquaredError(Metric):
    greater_is_better = False
    name = "RMSE"
    type = "classification"

    @staticmethod
    def evaluate(prediction, target):
        # =======================================================================
        # value = 0
        # for i in range(prediction.shape[0]):
        #     dif = prediction[i] - target[i]
        #     value += dif * dif
        # value /= prediction.shape[0]
        # value = sqrt(value)[0]
        # return value
        # =======================================================================
        return sqrt(mean(square(prediction - target)))


class WeightedRootMeanSquaredError(Metric):
    greater_is_better = False

    def __init__(self, weight_vector):
        self.weight_vector = weight_vector

    def evaluate(self, prediction, target):
        """Calculates RMSE taking into account a weight vector"""
        return sqrt(mean(square((prediction - target) * self.weight_vector)))


class Accuracy(Metric):
    greater_is_better = True

    @staticmethod
    def evaluate(prediction, target):
        return accuracy_score(target, prediction, normalize=True)

        # =======================================================================
        # return accuracy_score(target, where(prediction >= 0.5, True, False))
        # =======================================================================
        # =======================================================================
        # return accuracy_score(where(target >= 0.5, True, False), where(prediction >= 0.5, True, False))
        # =======================================================================


class AUROC(Metric):
    greater_is_better = True

    @staticmethod
    def evaluate(prediction, target, bound=True):
        if bound:
            auroc_y_score = empty(prediction.shape)
            for i in range(prediction.shape[0]):
                if prediction[i] < 0:
                    auroc_y_score[i] = 0
                elif prediction[i] > 1:
                    auroc_y_score[i] = 1
                else:
                    auroc_y_score[i] = prediction[i]
            return roc_auc_score(target, auroc_y_score)
        else:
            return roc_auc_score(target, prediction)


class AUROC_2(Metric):
    greater_is_better = True

    @staticmethod
    def evaluate(prediction, target):
        return AUROC.evaluate(prediction, target, bound=False)


class BinaryCrossEntropy(Metric):
    greater_is_better = False

    @staticmethod
    def evaluate(prediction, target):
        y_pred_prob = empty(prediction.shape)
        for i in range(prediction.shape[0]):
            if prediction[i] < 0:
                y_pred_prob[i] = 0
            elif prediction[i] > 1:
                y_pred_prob[i] = 1
            else:
                y_pred_prob[i] = prediction[i]

        return log_loss(target, y_pred_prob)


class CCE(Metric):
    greater_is_better = False
    name = "CCE"
    type = "classification"

    def __repr__(self):
        return str("LogLoss")

    def __str__(self):
        return str("LogLoss")

    @staticmethod
    def evaluate(prediction, target, epsilon=1e-12, ):
        """
        Computes cross entropy between targets (encoded as one-hot vectors)
        and predictions.
        Input: predictions (N, k) ndarray
               targets (N, k) ndarray
        Returns: scalar
        """
        predictions = np.clip(prediction, epsilon, 1. - epsilon)
        N = predictions.shape[0]
        ce = -np.sum(target * np.log(predictions + 1e-9)) / N
        return ce


class MSE(Metric):
    greater_is_better = False
    type = "regression"
    name = "MSE"

    @staticmethod
    def evaluate(prediction, target):
        # =======================================================================
        # value = 0
        # for i in range(prediction.shape[0]):
        #     dif = prediction[i] - target[i]
        #     value += dif * dif
        # value /= prediction.shape[0]
        # value = sqrt(value)[0]
        # return value
        # =======================================================================
        return mean(square(prediction - target))


class Accurarcy(Metric):
    greater_is_better = False
    name = "CCE"
    type = "classification"

    def __repr__(self):
        return str("Acc")

    def __str__(self):
        return str("Acc")

    @staticmethod
    def evaluate(prediction, target):
        m = tf.keras.metrics.CategoricalAccuracy()
        m.update_state(y_true=target, y_pred=prediction)
        return m.result().numpy()


def _is_better(value_1, value_2, metric):
    # ===========================================================================
    # print('[DEBUG] value 1 = %.5f, value 2 = %.5f\n' % (value_1, value_2))
    # ===========================================================================
    if metric.greater_is_better:
        return value_1 > value_2
    else:
        return value_1 < value_2


def new_elite(value_1, value_2):
    if True:
        if value_1.fitness > value_2.fitness:
            return value_2
        else:
            return value_1
    else:
        if value_1.fitness > value_2.fitness:
            return value_1
        else:
            return value_2
