#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import keras
from  keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
import pandas as pd
from keras.utils import to_categorical
import matplotlib.pyplot as plt


# In[2]:


df_train = pd.read_csv('train.txt', header =None, sep =';', names = ['Input','Sentiment'], encoding='utf-8')
df_test = pd.read_csv('test.txt', header = None, sep =';', names = ['Input','Sentiment'],encoding='utf-8')
df_val=pd.read_csv('val.txt',header=None,sep=';',names=['Input','Sentiment'],encoding='utf-8')


# In[3]:




df_train.Sentiment.value_counts()


# In[4]:


X=df_train['Input']


# In[5]:


lst=[]
for i in X:
  lst.append(len(i))


# In[6]:


len1=pd.DataFrame(lst)
len1.describe()


# In[7]:


cts=[]
for i in range(7,301):
   ct=0
   for k in lst:
     if k==i:
       ct+=1
   cts.append(ct)


# In[8]:


plt.bar(range(7,301),cts)
plt.show()


# In[9]:


tokenizer=Tokenizer(15212,lower=True,oov_token='UNK')
tokenizer.fit_on_texts(X)


# In[10]:


len(tokenizer.word_index)


# In[11]:


X_train=tokenizer.texts_to_sequences(X)
X_train_pad=pad_sequences(X_train,maxlen=80,padding='post')


# In[12]:


df_train['Sentiment']=df_train.Sentiment.replace({'joy':0,'anger':1,'love':2,'sadness':3,'fear':4,'surprise':5})


# In[13]:


Y_train=df_train['Sentiment'].values


# In[14]:




Y_train_f=to_categorical(Y_train)


# In[15]:


Y_train_f[:6]


# In[16]:


X_val=df_val['Input']
Y_val=df_val.Sentiment.replace({'joy':0,'anger':1,'love':2,'sadness':3,'fear':4,'surprise':5})


# In[17]:


X_val_f=tokenizer.texts_to_sequences(X_val)
X_val_pad=pad_sequences(X_val_f,maxlen=80,padding='post')


# In[18]:


Y_val_f=to_categorical(Y_val)


# In[19]:


Y_val_f[:6]


# In[20]:


from keras.models import Sequential
from keras.layers import LSTM,Bidirectional,Dense,Embedding,Dropout


# In[21]:


model=Sequential()
model.add(Embedding(15212,64,input_length=80))
model.add(Dropout(0.6))
model.add(Bidirectional(LSTM(80,return_sequences=True)))
model.add(Bidirectional(LSTM(160)))
model.add(Dense(6,activation='softmax'))
print(model.summary())


# In[22]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[23]:


hist=model.fit(X_train_pad,Y_train_f,epochs=12,validation_data=(X_val_pad,Y_val_f))


# In[24]:


plt.plot(hist.history['accuracy'],c='b',label='train')
plt.plot(hist.history['val_accuracy'],c='r',label='validation')
plt.legend(loc='lower right')
plt.show()


# In[25]:


plt.plot(hist.history['loss'],c='orange',label='train')
plt.plot(hist.history['val_loss'],c='g',label='validation')
plt.legend(loc='upper right')
plt.show()


# In[26]:


X_test=df_test['Input']
Y_test=df_test.Sentiment.replace({'joy':0,'anger':1,'love':2,'sadness':3,'fear':4,'surprise':5})


# In[27]:


X_test_f=tokenizer.texts_to_sequences(X_test)
X_test_pad=pad_sequences(X_test_f,maxlen=80,padding='post')


# In[28]:


Y_test_f=to_categorical(Y_test)


# In[29]:


X_test_pad.shape


# In[30]:


Y_test_f[:7]


# In[31]:


model.evaluate(X_test_pad,Y_test_f)


# In[32]:


def get_key(value):
    dictionary={'joy':0,'anger':1,'love':2,'sadness':3,'fear':4,'surprise':5}
    for key,val in dictionary.items():
          if (val==value):
            return key


# In[33]:


def predict(sentence):
  sentence_lst=[]
  sentence_lst.append(sentence)
  sentence_seq=tokenizer.texts_to_sequences(sentence_lst)
  sentence_padded=pad_sequences(sentence_seq,maxlen=80,padding='post')
  ans=get_key(model.predict_classes(sentence_padded))
  print("The emotion predicted is",ans)


# In[34]:


predict(str(input('Enter a sentence : ')))


# In[35]:


predict(str(input('Enter a sentence : ')))


# In[ ]:




