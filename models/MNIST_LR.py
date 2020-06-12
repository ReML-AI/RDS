from sklearn.linear_model import LogisticRegression
import numpy as np


class MNIST_LR:
    def __init__(self):
        pass

    def run(self, state, action):
        data_x, data_y = state
        train_x, train_y, test_x, test_y = data_x[action == 1], \
                                           data_y[action == 1], \
                                           data_x[action == 0], \
                                           data_y[action == 0]

        y_train_integer = np.argmax(train_y, axis=1)
        y_test_integer = np.argmax(test_y, axis=1)
        model = LogisticRegression(solver="liblinear",random_state=123)
        model.fit(train_x, y_train_integer) 

        return model.predict_proba(test_x)

