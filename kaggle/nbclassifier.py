import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer

train_df = pd.read_csv('train.csv',encoding = "utf-8").head(1300000)
test_df = pd.read_csv('train.csv',encoding = "utf-8")[1300000:1360000]
real_test_df = pd.read_csv('test.csv',encoding = "utf-8")
# build analyzers (bag-of-words)
BOW_500 = CountVectorizer(max_features=500, tokenizer=nltk.word_tokenize) 
BOW_500.fit(train_df['text'])

X_train = BOW_500.transform(train_df['text'])
y_train = train_df['labels']
X_test = BOW_500.transform(test_df['text'])
y_test = test_df['labels']
X_real_test = BOW_500.transform(real_test_df['text'])
NB = MultinomialNB() # build model
NB = NB.fit(X_train, y_train) # training

y_train_pred = NB.predict(X_train) # predict training data
y_test_pred = NB.predict(X_test) # predict test data
real_test_pred = NB.predict(X_real_test)
print(real_test_pred[0])

# accuracy
acc_train = accuracy_score(y_true=y_train, y_pred=y_train_pred)
acc_test = accuracy_score(y_true=y_test, y_pred=y_test_pred)
{'sadness': 0, 'disgust': 1, 'anticipation': 2, 'joy': 3, 'trust': 4, 'anger': 5, 'fear': 6, 'surprise': 7}
label_list = ['sadness','disgust','anticipation','joy','trust','anger','fear','surprise']

print('training accuracy: {}'.format(round(acc_train, 2)))
print('test accuracy: {}'.format(round(acc_test, 2)))
with  open('ans.csv','w') as f:
    f.write('id,emotion\n')
    for i in range(0,len(real_test_pred)):
        f.write('{},{}\n'.format(real_test_df.loc[i]['ID'],label_list[real_test_pred[i]]))
    