import numpy as np
from sklearn.svm import SVR

class BOS_SVM:
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
        
        np.random.seed(123)
        model = SVR(kernel='linear')
        model.fit(train_x, train_y)
        return np.expand_dims(model.predict(test_x), axis=1)
    