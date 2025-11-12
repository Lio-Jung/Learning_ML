#imports
import pandas as pd
import tensorflow as tf

#read data
ref = "https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv"
iris = pd.read_csv(ref)
print(iris.shape)
print(iris.columns) #as a output : '품종'
print(iris.head)

#!!!One Hot Encoding!!!
iris = pd.get_dummies(iris) #whole STR -> number
print(iris.columns) #as outputs : '품종_setosa', '품종_versicolor','품종_virginica'

#slice data
ind_v = iris[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
d_v = iris[['품종_setosa', '품종_versicolor','품종_virginica']]

#model
X = tf.keras.layers.Input(shape=[4])
Y = tf.keras.layers.Dense(3, activation='softmax')(X) #added : activatin='softmax'
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy')

#fit
model.fit(ind_v, d_v, epochs=100)

#use
model.predict(ind_v[-5:])   #read 5 rows from at the end