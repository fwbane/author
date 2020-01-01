from datetime import datetime
import numpy as np
import math
from os import path

class Model:
    def __init__(self, name):
        self.name = name
        self.description = "Model description here"
        self.fname = name + ".model"
        self.confidence = None

    def fit(self):
        pass

    def transform(self):
        pass

    def fit_transform(self):
        self.fit(self)
        self.transform(self)

    def save(self, fname=None):
        if not fname:
            fname = self.fname
        pass


def load_data(train_data_path, test_data_path):
    train_data = []
    test_data = []
    print("Loading data")
    return train_data, test_data

def preprocess_data(data):
    clean_data = []
    print("Preprocessing data")
    return clean_data

def create_features(data, features):
    feature_array = [] # numpy array of size (examples,features)?
    for feature in features:
        feature_vector = get_feature_vector(data, feature)
        feature_array.append(feature_vector)
    return feature_array

def get_feature_vector(data, feature):
    return feature(data) #assuming here that feature is a function passed in

def get_DataFrame(data, features, feature_array):
    for n, feature in enumerate(features):
        pass
#         df[feature] = feature_array[n]? [:,n]? [n,:]?

def train_model(data, model):
    print("Training model {}".format(model.description))
    trained_model = []
    return trained_model

def get_model_predictions(data, models):
    predictions = []
    for model in models:
        prediction = model.predict(data)
        predictions.append(prediction)
    return predictions

def model_mixture(models, predictions):
    final_answer = []
    for prediction in predictions:
        final_answer.append(0)
    return final_answer

def prepare_submission(output, fname):
    with open(fname, 'w') as f:
        for line in output:
            f.write(line)


def main():
    save_models = False
    train_data_path, test_data_path = "train.csv", "test.csv"
    data = load_data(train_data_path, test_data_path)
    data = preprocess_data(data)
    features = [] #TODO: Logic for how feature list is instantiated
    feature_array = create_features(features)
    df = get_DataFrame(data, features, feature_array)
#     TODO: Logic for how DF is passed to model creator
    models = [] #TODO: Logic for how model list is instantiated
    trained_models = []
    for model in models:
        trained_models.append(train_model(data, model))
    if save_models:
        for model in trained_models:
            model.save(model.fname)
    predictions = get_model_predictions(data, trained_models)
    output = model_mixture(trained_models, predictions)
    outfile_dir = ""
    outfile_name = path.join(outfile_dir, "submission_{}.csv".format(datetime.now().strftime("%Y-%m-%d-%H:%M")))
    submission = prepare_submission(output, outfile_name)
    print("Submission file saved to {}".format(outfile_name))



if __name__ == '__main__':
    main()
