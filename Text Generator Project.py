#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
import sys
import nltk
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


# In[2]:


file = open("shakespeare.txt").read()


# In[4]:


def tokenize_words(input):
    input = input.lower()

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)

   
    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return " ".join(filtered)

processed_inputs = tokenize_words(file)


# In[5]:


chars = sorted(list(set(processed_inputs)))
char_to_num = dict((c, i) for i, c in enumerate(chars))


# In[6]:


input_len = len(processed_inputs)
vocab_len = len(chars)
print ("Total number of characters:", input_len)
print ("Total vocab:", vocab_len)


# In[7]:


seq_length = 100
x_data = []
y_data = []


# In[8]:


for i in range(0, input_len - seq_length, 1):
    in_seq = processed_inputs[i:i + seq_length]
    out_seq = processed_inputs[i + seq_length]
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])


# In[9]:


n_patterns = len(x_data)
print ("Total Patterns:", n_patterns)


# In[11]:


X = numpy.reshape(x_data, (n_patterns, seq_length, 1))
X = X/float(vocab_len)
y = np_utils.to_categorical(y_data)


# In[12]:


model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))


# In[13]:


model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[14]:


filepath = "model_weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
desired_callbacks = [checkpoint]


# In[15]:


model.fit(X, y, epochs=4, batch_size=256, callbacks=desired_callbacks)


# In[16]:


filename = "model_weights_saved.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[17]:


num_to_char = dict((i, c) for i, c in enumerate(chars))


# In[18]:


start = numpy.random.randint(0, len(x_data) - 1)
pattern = x_data[start]
print("Random Seed:")
print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")


# In[19]:


for i in range(1000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(vocab_len)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = num_to_char[index]

    sys.stdout.write(result)

    pattern.append(index)
    pattern = pattern[1:len(pattern)]


# In[ ]:




