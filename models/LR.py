from sklearn.linear_model import LogisticRegression

class LR:

    def run(self, state, action):
        clf = LogisticRegression(solver='liblinear', random_state=123)
        data_x, data_y = state
        
        train_x, train_y, test_x = data_x[action == 1], data_y[action == 1, 1], data_x[action == 0]
        
        clf.fit(train_x, train_y)
        test_y_pred = clf.predict_proba(test_x)
        
        return test_y_pred