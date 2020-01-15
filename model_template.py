import pandas as pd

MODEL_DIR = "/media/D/data/models/"

class Model:
    def __init__(self):
        self.embedding = None
        self.model = None
        self.dir = MODEL_DIR

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

    def phi(self):
        print("Hi")