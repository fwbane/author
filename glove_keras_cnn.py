from nltk.tokenize import TreebankWordTokenizer
import os
import gensim
from sklearn.model_selection import train_test_split
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, Dropout, GlobalMaxPooling1D
from keras.optimizers import SGD
import gensim
from keras.models import model_from_json


from model_template import Model

class glove_keras_cnn(Model):
    def preprocess(self):
        df = Model.preprocess(self)
        authors = list(df.author.unique())
        lookup = {a: _ for _, a in enumerate(authors)}
        df['y'] = [lookup[i] for i in df.author]
        return df

    def vectorize(self, dataset):
        print("vectorizing")
        t = time.time()
        GLOVE_DIR = "/media/D/data/glove/"
        GLOVE_W2V_FILE = "glove.840B.300d.w2vformat.txt"
        GLOVE_W2V_PATH = os.path.join(GLOVE_DIR, GLOVE_W2V_FILE)
        glove_model = gensim.models.KeyedVectors.load_word2vec_format(GLOVE_W2V_PATH)
        print("time taken loading glove: {}".format(time.time()-t))
        wv = glove_model.wv
        tokenizer = TreebankWordTokenizer()
        vectorized_data = []
        for sentence in dataset:
            sample_vecs = []
            for token in tokenizer.tokenize(sentence):
                try:
                    sample_vecs.append(wv[token])
                except KeyError:
                    # print(token, "not in wv")
                    pass
            vectorized_data.append(sample_vecs)
        return vectorized_data

    def create(self):
        print('Build model...')
        model = Sequential()
        model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1,
                         input_shape=(maxlen, embedding_dims)))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(hidden_dims))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(num_classes))
        model.add(Activation('sigmoid'))

        return model





    def train(self, model):
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_dev, Y_dev))
        return model

    def save(self, model, save_weights=False):
        model_structure = model.to_json()
        with open("cnn_model.json", "w") as json_file:
            json_file.write(model_structure)
        if save_weights:
            model.save_weights("cnn_weights.h5")

    def load(self):
        with open("cnn_model.json", "r") as json_file:
            json_string = json_file.read()
        model = model_from_json(json_string)
        model.load_weights('cnn_weights.h5')
        return model

    def predict(self, query):
        pass

def pad_trunc(data, maxlen):
    new_data = []
    # Create a vector of 0s the length of our word vectors
    zero_vector = []
    for _ in range(len(data[0][0])):
        zero_vector.append(0.0)
    for sample in data:
        if len(sample) > maxlen:
            temp = sample[:maxlen]
        elif len(sample) < maxlen:
            temp = sample
            # Append the appropriate number 0 vectors to the list
            additional_elems = maxlen - len(sample)
            for _ in range(additional_elems):
                temp.append(zero_vector)
        else:
            temp = sample
        new_data.append(temp)
    return new_data

if __name__ == "__main__":
    load = False # Eventually parse as command line argument
    maxlen = 100
    batch_size = 32
    embedding_dims = 300
    filters = 250
    kernel_size = 3
    hidden_dims = 250
    epochs = 2

    cnn = glove_keras_cnn()
    if load:
        model = cnn.load()
    else:
        t = time.time()
        print("hi\t{}".format(t))
        df = cnn.preprocess()
        num_classes = len(list(df.author.unique()))
        print(df.shape)
        X, y = df['text'], df['y']
        X = cnn.vectorize(X)
        X_train, X_dev, Y_train, Y_dev = train_test_split(X, y, test_size=0.2, random_state=707)
        print("padding data", time.time() - )
        X_train = pad_trunc(X_train, maxlen)
        X_dev = pad_trunc(X_dev, maxlen)
        X_train = np.reshape(X_train, (len(X_train), maxlen, embedding_dims))
        Y_train = np.array(Y_train)
        X_dev = np.reshape(X_dev, (len(X_dev), maxlen, embedding_dims))
        Y_dev = np.array(Y_dev)
        print(X_train.shape, Y_train.shape, X_dev.shape, Y_dev.shape)
        print("creating model", time.time()-t)
        model = cnn.create()
        print("training model", time.time() - t)
        model = cnn.train(model)
        cnn.save(model, save_weights=True)








