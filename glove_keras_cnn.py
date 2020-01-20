from nltk.tokenize import TreebankWordTokenizer
import os
import gensim
from sklearn.model_selection import train_test_split
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, Dropout, GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D
from keras.optimizers import SGD, Adam
import gensim
from keras.models import model_from_json
import pickle

from model_template import Model

load = True  # Eventually parse as command line argument
maxlen = 100
batch_size = 32
embedding_dims = 300
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 30
num_classes = 3

GLOVE_DIR = "/media/D/data/glove/"
GLOVE_W2V_FILE = "glove.840B.300d.w2vformat.txt"
GLOVE_W2V_PATH = os.path.join(GLOVE_DIR, GLOVE_W2V_FILE)

# BASE CLASS FOR KERAS CNN WITH GLOVE
class GloveKerasCnn(Model):
    def __init__(self):
        Model.__init__(self)
        self.mname = "glove_keras_cnn_model.json"
        self.wname = "glove_keras_cnn_weights.h5"

    def preprocess(self):
        df = Model.preprocess(self)
        authors = list(df.author.unique())
        lookup = {a: _ for _, a in enumerate(authors)}
        y_numbers = [lookup[i] for i in df.author]
        y_vecs = []
        for y in y_numbers:
            base_vec = np.zeros(num_classes, dtype='int')
            base_vec[y] = 1
            y_vecs.append(base_vec)
        df['y'] = y_vecs
        return df

    def vectorize(self, dataset):
        print("vectorizing")
        if not self.embedding:
            GLOVE_DIR = "/media/D/data/glove/"
            GLOVE_W2V_FILE = "glove.840B.300d.w2vformat.txt"
            GLOVE_W2V_PATH = os.path.join(GLOVE_DIR, GLOVE_W2V_FILE)
            glove_model = gensim.models.KeyedVectors.load_word2vec_format(GLOVE_W2V_PATH)
            # print("time taken loading glove: {}".format(time.time()-t))
            self.embedding = glove_model.wv
        wv = self.embedding
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
        self.model = model
        return model

    def train(self, model, X_train, Y_train, X_dev, Y_dev):
        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, clipnorm=1.)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_dev, Y_dev))
        return model

    def save(self, model, save_weights=False):
        model_structure = model.to_json()
        outfile = os.path.join(self.dir, self.mname)
        with open(outfile, "w") as json_file:
            json_file.write(model_structure)
        if save_weights:
            outfile = os.path.join(self.dir, self.wname)
            model.save_weights(outfile)

    def load(self):
        model_path = os.path.join(self.dir, self.mname)
        weight_path = os.path.join(self.dir, self.wname)
        with open(model_path, "r") as json_file:
            json_string = json_file.read()
        model = model_from_json(json_string)
        model.load_weights(weight_path)
        self.model = model
        return model

    def predict(self, query, vectorize=False):
        if not self.model:
            print("No model available for prediction")
        if vectorize:
            if isinstance(query, str):
                query = [query]
            vectorized_query = self.vectorize(query)
            pickle.dump(vectorized_query, open("glove_vectorized_test_sentences", "wb"))
        else:
            vectorized_query = query
        vectorized_query = self.pad_trunc(vectorized_query, maxlen)
        vectorized_query = np.asarray(vectorized_query)
        vectorized_query = np.reshape(vectorized_query, (len(query), maxlen, embedding_dims)) # Should be redundant, to ensure compliance
        predictions = self.model.predict(vectorized_query)
        return predictions


    def pad_trunc(self, data, maxlen):
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



class VanillaGloveKerasCnn(GloveKerasCnn):
    def __init__(self, wv=None):
        GloveKerasCnn.__init__(self)
        if wv:
            self.embedding = wv
        self.mname = "vanilla_cnn_model.json"
        self.wname = "vanilla_cnn_weights.h5"

    def create(self):
        model = Sequential()
        model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1,
                         input_shape=(maxlen, embedding_dims)))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(hidden_dims))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(num_classes))
        model.add(Activation('sigmoid'))
        self.model = model
        return model

    def train(self, model, X_train, Y_train, X_dev, Y_dev):
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, clipnorm=1.)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_dev, Y_dev))
        return model

class GloveKerasDoubleCnn(GloveKerasCnn):
    def __init__(self, wv=None):
        GloveKerasCnn.__init__(self)
        if wv:
            self.embedding = wv
        self.mname = "double_cnn_model.json"
        self.wname = "double_cnn_weights.h5"

    def create(self):
        double_filters_1 = 128
        double_filters_2 = 256
        double_kernel_size_1 = 5
        double_kernel_size_2 = 3
        double_hidden_dims_1 = 128

        model = Sequential()
        model.add(Conv1D(double_filters_1, double_kernel_size_1, padding='valid', activation='relu', input_shape=(maxlen, embedding_dims)))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.2))
        model.add(Dense(double_hidden_dims_1))
        model.add(Activation('relu'))
        model.add(Conv1D(double_filters_2, double_kernel_size_2, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('sigmoid'))
        self.model = model
        return model

    def train(self, model, X_train, Y_train, X_dev, Y_dev):
        double_epochs = 30
        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, clipnorm=1.)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=double_epochs, validation_data=(X_dev, Y_dev))
        return model

class GloveKerasStackedCnn(GloveKerasCnn):
    def __init__(self, wv=None):
        GloveKerasCnn.__init__(self)
        if wv:
            self.embedding = wv
        self.mname = "stacked_cnn_model.json"
        self.wname = "stacked_cnn_weights.h5"

    def create(self):
        stacked_filters_1 = 64
        stacked_filters_2 = 128
        stacked_kernel_size_1 = 3
        stacked_kernel_size_2 = 3
        model = Sequential()
        model.add(Conv1D(stacked_filters_1, stacked_kernel_size_1, padding='valid', activation='relu', input_shape=(maxlen, embedding_dims)))
        model.add(Conv1D(stacked_filters_1, stacked_kernel_size_1, activation='relu'))
        model.add(MaxPooling1D(5))
        model.add(Conv1D(stacked_filters_2, stacked_kernel_size_2, activation='relu'))
        model.add(Conv1D(stacked_filters_2, stacked_kernel_size_2, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
        self.model = model
        return model

    def train(self, model, X_train, Y_train, X_dev, Y_dev):
        stacked_epochs = 30
        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, clipnorm=1.)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=stacked_epochs, validation_data=(X_dev, Y_dev))
        return model

def main():
    t = time.time()
    cnn = GloveKerasCnn()
    if load:
        X = pickle.load(open("X-glove-encoding", "rb"))
        y = pickle.load(open("y-glove-encoding", "rb"))
    else:
        t = time.time()
        print("hi\t{}".format(t))
        df = cnn.preprocess()
        print(df.shape)
        X = df['text']
        y = np.ndarray([len(df['y']), num_classes])
        for _, vec in enumerate(df['y']):
            y[_] = vec
        X = cnn.vectorize(X)
        pickle.dump(X, open("X-glove-encoding", "wb"))
        pickle.dump(y, open("y-glove-encoding", "wb"))

    cnns = [GloveKerasCnn(), GloveKerasStackedCnn(), GloveKerasDoubleCnn()]
    randoms = np.random.randint(0, 9999, size=len(cnns))
    for cnn, random_state in zip(cnns, randoms):
        print("Now training model: {}".format(cnn))
        X_train, X_dev, Y_train, Y_dev = train_test_split(X, y, test_size=0.2, random_state=random_state)
        X_train = cnn.pad_trunc(X_train, maxlen)
        X_dev = cnn.pad_trunc(X_dev, maxlen)
        X_train = np.reshape(X_train, (len(X_train), maxlen, embedding_dims))
        Y_train = np.reshape(Y_train, (len(Y_train), num_classes))
        X_dev = np.reshape(X_dev, (len(X_dev), maxlen, embedding_dims))
        Y_dev = np.reshape(Y_dev, (len(Y_dev), num_classes))
        # if load:
        #     model = cnn.load()
        # else:
        model = cnn.create()
        model = cnn.train(model, X_train, Y_train, X_dev, Y_dev)
        cnn.save(model, save_weights=True)

    EAP_test_string = """Once upon a midnight dreary, while I pondered, weak and weary, 
    Over many a quaint and curious volume of forgotten lore, 
    While I nodded, nearly napping, suddenly there came a tapping, 
    As of some one gently rapping, rapping at my chamber door.""".replace("\n", "")
    HPL_test_string = """In this luminous Company I was tolerated more because of my Years 
    than for my Wit or Learning; being no Match at all for the rest. My Friendship for the 
    celebrated Monsieur Voltaire was ever a Cause of Annoyance to the Doctor; who was deeply 
    orthodox, and who us'd to say of the French Philosopher.""".replace("\n", "")
    MWS_test_string = """A few seconds ago they had all been active and healthy beings, 
    so full of employment they could not afford to mend his calèche unless tempted by 
    some extraordinary reward; now the men declared themselves cripples and invalids, the 
    children were orphans, the women helpless widows, and they would all die of hunger if 
    his Eccellenza did not bestow a few grani.""".replace("\n", "")
    test_strings = [EAP_test_string, HPL_test_string, MWS_test_string]

    if load:
        vectorized_query = pickle.load(open("glove_vectorized_test_sentences", "rb"))
        predictions = []
        for cnn in cnns:
            predictions.append(cnn.predict(vectorized_query, vectorize=False))


    else:
        predictions = []
        for cnn in cnns:
            predictions.append(cnn.predict(test_strings, vectorize=True))

    for i, sentence in enumerate(test_strings):
            print(sentence)
            for n, cnn in enumerate(cnns):
                print(cnn)
                print(predictions[n][i])
                print(np.argmax(predictions[n][i]), i, i == np.argmax(predictions[n][i]))
                print()


if __name__ == "__main__":
    main()






