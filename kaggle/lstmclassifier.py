from gensim.models import KeyedVectors
import re
import numpy as np
# or if you prefer tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import  pandas as pd
from tensorflow import one_hot
from sklearn.model_selection import train_test_split
# or if you don't like sklearn. **Remember to shuffle your data before splitting.**
import numpy as np
# or if you don't like tensorflow
from sklearn.preprocessing import OneHotEncoder
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Input
w2v = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz',binary = True)
def preprocess(text):
    result = []
    text = text.lower()
    # text = re.sub('[^A-Za-z]+', ' ', text)
    # text = re.sub('[\t|\s]+', ' ', text)
    for w in text.split(' '):
        result.append(w) 
    return result
def to_embedding(tokens):
    ret = []
    for item in tokens:
        try:
            ret.append(w2v[item])
        except:
            pass
    return ret

def add_padding(embeddings, padding_width = None):
    embeddings = pad_sequences( embeddings, dtype='float32', padding='pre', value=0.0,maxlen=padding_width)
    return embeddings

def process_text(sentences, padding = None):
    result = [ preprocess(sentence) for sentence in sentences ]
    result = [ to_embedding(sentence) for sentence in result ]
    result = add_padding(result, padding)
    return result

if __name__ == '__main__':

    indice, first_X, first_Y = [], [], [] # sentence id of selected samples, selected sentences, detected labels
    with open('train.csv') as f:
        lines = f.readlines()[1:]
        i = 0 
        for l in lines:
            l = l.replace('\n','')
            tmp = l.split(',')
            indice.append(tmp[0])
            first_X.append(tmp[1])
            first_Y.append(tmp[2])
            i+=1
            if i >=500000:
                break

    tok = Tokenizer()
    tok.fit_on_texts(pd.DataFrame (first_X, columns = ['sent'])['sent'])
    vocab_size = len(tok.word_index) + 1
    print(vocab_size)

    w2v = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz',binary = True)
    # embedding_matrix = np.zeros((vocab_size, 300))
    # for w, index in tok.word_index.items():
    #     try:
    #         embedding_matrix[index, :] = w2v[w]
    #     except:
    #         pass

    first_X = process_text(first_X)
    first_X = np.array(first_X)
    first_Y = np.array(first_Y,dtype="int32")
    print(first_X.shape)

    first_Y = one_hot(first_Y,dtype="int32",depth=8,on_value=1,off_value=0).numpy()
    X_train, X_val, Y_train, Y_val = train_test_split(
        first_X, first_Y,
        test_size = None,   # [TODO] How much data you want to used as validation set
        shuffle = False
    )

    _, PADDING_WIDTH, EMBEDDING_DIM = X_train.shape
    OUTPUT_CATEGORY = 8

    print(PADDING_WIDTH, EMBEDDING_DIM, OUTPUT_CATEGORY)
    inputs = Input(name='inputs',shape=[PADDING_WIDTH,300])
    layer = LSTM(128)(inputs)
    layer = Dense(128,activation="relu",name="FC1")(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(OUTPUT_CATEGORY,activation="softmax",name="FC2")(layer)
    model = Model(inputs=inputs,outputs=layer)

    print(model.summary())

    model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=["accuracy"])
    model.fit(
        X_train, Y_train, 
        validation_data=(X_val, Y_val),
        epochs = 8         # [TODO] how many iterations you want to run
        # initial_epoch = ?    # set this if you're continuing previous training
    )
    model.save('lstm.bin')