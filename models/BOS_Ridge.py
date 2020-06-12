import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


class BOS_Ridge:
    def __init__(self):
        pass

    def run(self, state, action):
        data_x, data_y = state

        train_x, train_y, test_x, test_y = (
            data_x[action == 1],
            data_y[action == 1].ravel(),
            data_x[action == 0],
            data_y[action == 0].ravel(),
        )
        model = Ridge(solver="saga", random_state=123)
        model.fit(train_x, train_y)

        return np.expand_dims(model.predict(test_x), axis=1)
