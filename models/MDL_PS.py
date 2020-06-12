from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

# Public Solution for MDL
class MDL_PS:
    def __init__(self):
        pass
        
    def run(self, state, action):
        data_x, data_y = state
        data_x = data_x[:, [48, 64, 105, 128, 241, 323, 336, 338, 378, 442, 453, 472, 475]]
        train_x, train_y, test_x = data_x[action == 1], data_y[action == 1, 1], data_x[action == 0]

        interactions = PolynomialFeatures(degree=4, interaction_only=True)
        train_x = interactions.fit_transform(train_x)
        test_x = interactions.fit_transform(test_x)
        
        self.clf = LogisticRegression(solver='liblinear', max_iter=1000)
        self.clf.fit(train_x, train_y)
        test_y_pred = self.clf.predict_proba(test_x)
        
        return test_y_pred
    
    def get_model(self):
        return self.clf