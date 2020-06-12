import numpy as np
from sklearn.ensemble import RandomForestClassifier

class MDL_RF:
    def __init__(self):
        pass
    
    def run(self, state, action):   
        clf = RandomForestClassifier(n_estimators=128, random_state=123, bootstrap=True, n_jobs=-1)
        data_x, data_y = state
        train_x, train_y, test_x = data_x[action == 1], data_y[action == 1, 1], data_x[action == 0]

        clf.fit(train_x, train_y)
        test_y_pred = clf.predict_proba(test_x)

        return test_y_pred
