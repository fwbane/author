import pandas as pd
class Model:
    def preprocess(self):
        filename = 'train.csv'
        df = pd.read_csv(filename, index_col='id')
        return df

    def vectorize(self, dataset):
        pass

    def create(self):
        pass

    def train(self, model, X_train, Y_train, X_dev, Y_dev):
        pass

    def save(self, model, save_weights=False):
        pass

    def load(self):
        pass

    def predict(self, query):
        pass
