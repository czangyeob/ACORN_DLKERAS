# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers.core import Dense, Lambda
from keras.layers.embeddings import Embedding
import keras.backend as K

vocab_size = 5000
embed_size = 300
window_size = 1

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embed_size, 
                    init='glorot_uniform',
                    input_length=window_size*2))
model.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))
model.add(Dense(vocab_size, init='glorot_uniform',
                activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer="adadelta")

# 가중치 가져오기
weights = model.layers[0].get_weights()[0]