import pandas as pd
class Model:
    def preprocess(self):
        filename = 'train.csv'
        df = pd.read_csv(filename, index_col='id')
        return df

    def vectorize(self):
        pass

    def create(self):
        pass

    def train(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def predict(self, query):
        pass
