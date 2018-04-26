
import sys
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
import keras.backend as K
import keras.layers
import keras.layers.merge
from keras.layers import Input, Embedding, Reshape, Merge, Flatten, Activation, concatenate
from keras.models import Model
from keras.losses import mean_absolute_error, mean_squared_error, binary_crossentropy, categorical_crossentropy
from keras.models import load_model
from keras import regularizers
import numpy as np
import csv
import os
from random import randint
import pandas as pd
#--------------------------------------------------------

# X1 = []
# X2 =[]
# y = []

# count = 0
# while count <= 5000:
#     X1.append((randint(10,50)))
#     X2.append((randint(10,50)))
#     count+=1

# for i in range(0,len(X1)):
#     if X1[i] > 25 and X2[i] > 25:
#         y.append(1)
#     else:
#         y.append(0)

data = pd.read_csv("/Users/puchiha/Documents/GitHub/crime-data-analysis/517a-crime-vancouver/raw-data/nn_processed.csv").as_matrix()
#X = data[:, [0,1,2,3,4,5,6,7,9]]
X1 = data[:, 0]
X2 = data[:, 1]
X3 = data[:, 2]
X4 = data[:, 3]
X5 = data[:, 4]
X6 = data[:, 5]
X7 = data[:, 6]
X8 = data[:, 7]
X9 = data[:, 9]
y = data[:, 8]

def data_generator(batchsize):
    input1, input2, input3, input4, input5, input6, input7, input8, input9 = [], [], [], [], [], [], [], [], []
    output = []

    def package_batch(inputs, output):
        input1 = np.array(inputs[0])
        input2 = np.array(inputs[1])
        input3 = np.array(inputs[2])
        input4 = np.array(inputs[3])
        input5 = np.array(inputs[4])
        input6 = np.array(inputs[5])
        input7 = np.array(inputs[6])
        input8 = np.array(inputs[7])
        input9 = np.array(inputs[8])
        output = np.array(output)
        data = ([input1, input2, input3, input4, input5, input6, input7, input8, input9], output)
        return data

    while True:
        for i in range(0,len(X1)):
            input1.append([X1[i]])
            input2.append([X2[i]])
            input3.append([X3[i]])
            input4.append([X4[i]])
            input5.append([X5[i]])
            input6.append([X6[i]])
            input7.append([X7[i]])
            input8.append([X8[i]])
            input9.append([X9[i]])
            output.append([y[i]])
            if len(input1) == batchsize:
                yield package_batch([input1, input2, input3, input4, input5, input6, input7, input8, input9], output)
                input1, input2, input3, input4, input5, input6, input7, input8, input9 = [], [], [], [], [], [], [], [], []
                output = []
                #input1 = []; input2 = []; output = []
        yield package_batch([input1, input2, input3, input4, input5, input6, input7, input8, input9], output)
        input1, input2, input3, input4, input5, input6, input7, input8, input9 = [], [], [], [], [], [], [], [], []
        output = []
        #input1 = []; input2 = []; output = []
#---------------------------------------------------------------------------------------MODEL---------------------------------------------------------------------------

x1 = Input(shape = (1,), dtype='float32', name = 'x1')
x2 = Input(shape = (1,), dtype='float32', name = 'x2')
x3 = Input(shape = (1,), dtype='float32', name = 'x3')
x4 = Input(shape = (1,), dtype='float32', name = 'x4')
x5 = Input(shape = (1,), dtype='float32', name = 'x5')
x6 = Input(shape = (1,), dtype='float32', name = 'x6')
x7 = Input(shape = (1,), dtype='float32', name = 'x7')
x8 = Input(shape = (1,), dtype='float32', name = 'x8')
x9 = Input(shape = (1,), dtype='float32', name = 'x9')

x = concatenate([x1,x2,x3,x4,x5,x6,x7,x8,x9])



h1 = Dense(10, activation = 'relu')(x)

out = Dense(1, activation = 'relu', kernel_regularizer=regularizers.l2(0.01))(h1)

model = Model(inputs=[x1, x2, x3, x4, x5, x6, x7, x8, x9], outputs=[out])
model.compile(loss=mean_squared_error, optimizer='adagrad', metrics = ['accuracy'])

model.fit_generator(data_generator(100), steps_per_epoch=50, epochs=100)

#--------------------------------------------------------------------------------------PREDICT--------------------------------------------------------------------------
preds = []
for i in range(0,len(X1)):
    a, b, c, d, e, f, g, h, j = [], [], [], [], [], [], [], [], []
    #b = []
    a.append(X1[i])
    b.append(X2[i])
    c.append(X3[i])
    d.append(X4[i])
    e.append(X5[i])
    f.append(X6[i])
    g.append(X7[i])
    h.append(X8[i])
    j.append(X9[i])
    
    
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    d = np.array(d)
    e = np.array(e)
    f = np.array(f)
    g = np.array(g)
    h = np.array(h)
    j = np.array(j)
    
    ip = ([a, b, c, d, e, f, g, h, j])
    prediction=model.predict(ip , verbose = 0)
    preds.append(prediction[0][0])

with open('output.tsv', 'wb') as tsvfile:
    tsvwriter = csv.writer(tsvfile, delimiter="\t")
    for i in range(0,len(y)):
        tsvwriter.writerow([y[i],preds[i]])
