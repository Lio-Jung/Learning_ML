#imports
import pandas as pd
import tensorflow as tf

#read data
ref = "https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv"
boston = pd.read_csv(ref)
print(boston.shape) #for checking, wether it worked
print(boston.head) #to see first 5 rows
print(boston.columns) #for checking, which !!!VARIABLE!!!

#slice data
ind_v = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
       'ptratio', 'b', 'lstat']] #conditions for price
d_v = boston[['medv']]  #price
print(ind_v.shape, d_v.shape) #to check, whether it worked good

#modeling
X = tf.keras.layers.Input(shape=[13]) #13 contidions
Y = tf.keras.layers.Dense(1)(X) # 1 output
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse') #'mse(Mean Squared Error)' is a method for !!!REGRESSION!!!

#fit
model.fit(ind_v, d_v, epochs=100) #epochs = times of trying

#use
model.predict([ind_v[0:5]]) #[0:5] = sliceing... means from 0 til 5th row