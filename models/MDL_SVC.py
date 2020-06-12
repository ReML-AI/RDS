from sklearn.svm import SVC

class MDL_SVC:
    def __init__(self):
        pass
    
    def run(self, state, action):
        data_x, data_y = state
        train_x, train_y, test_x = data_x[action == 1], data_y[action == 1, 1], data_x[action == 0]
        self.clf = SVC(kernel='rbf',probability=True,coef0=1)
        self.clf.fit(train_x, train_y)
        test_y_pred = self.clf.predict_proba(test_x)
        return test_y_pred
    
    def get_model(self):
        return self.clf