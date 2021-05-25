import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn


data = pd.read_csv('data.csv')
data.head()

data.drop(['no'], axis='columns', inplace=True)
data['type'].replace({'Endolysins': 1,'Autolysins':0}, inplace=True)



codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def create_dict(codes):
  char_dict = {}
  for index, val in enumerate(codes):
    char_dict[val] = index+1

  return char_dict

char_dict = create_dict(codes)

#print(char_dict)
#print("Dict Length:", len(char_dict))


def integer_encoding(data):
  """
  - Encodes code sequence to integer values.
  - 20 common amino acids are taken into consideration
    and rest 4 are categorized as 0.
  """
  
  encode_list = []
  for row in data['sequence'].values:
    row_encode = []
    for code in row:
      row_encode.append(char_dict.get(code, 0))
    encode_list.append(np.array(row_encode))
  
  return encode_list

testing = integer_encoding(data)
#len(testing)


import tensorflow as tf
from tensorflow.keras.preprocessing import sequence as sequence

max_length = 100
train_pad = sequence.pad_sequences(testing, maxlen=max_length, padding='post', truncating='post')

#print(train_pad)
#print(train_pad.shape)

y = data['type']

classes = data['type'].value_counts()[:1000].index.tolist()
#len(classes)

train_sm = data['type']
#print('Train size :', len(train_sm))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_pad, train_sm, test_size = 0.10, random_state=10)


#print(X_train[4])
#print(X_test)
#print(y_train)
#print(y_test)


from sklearn.svm import SVC # "Support vector classifier"  
classifier = SVC(kernel='linear', random_state=10)  
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)


from sklearn.metrics import accuracy_score,confusion_matrix

print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))

import pickle
import os

if not os.path.exists('models'):
    os.makedirs('models')
    
MODEL_PATH = "models/clf.sav"
pickle.dump(classifier, open(MODEL_PATH, 'wb'))
