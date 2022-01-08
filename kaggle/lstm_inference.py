import numpy as np
import tensorflow as tf
from tensorflow import keras
from lstmclassifier import  process_text

model = keras.models.load_model('lstm.bin')

with open('submissionLSTM.csv','w') as fo:
    label_list = ['sadness','disgust','anticipation','joy','trust','anger','fear','surprise']
    fo.write('id,emotion\n')
    with open('test.csv','r') as f:
        lines = f.readlines()[1:]
        test_ids = []
        testcases = []
        for l in lines:
            l = l.replace('\n','')
            tmp = l.split(',')
            test_ids.append(tmp[0])
            testcases.append(tmp[1])
        test_X = process_text(testcases,padding=35)
        predictions = model.predict(test_X)
        for idx, result in enumerate(predictions):
            predict_id = result.argmax()
            fo.write("{},{}\n".format(test_ids[idx],label_list[predict_id]))