import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as pt # is used to plot the data 
import pandas as pd   #is used to load the data 
import tensorflow as tf 

#data=pd.read_csv('dataset/train.csv')
#(x_train,y_train),(x_test,y_test)=data
mnist= tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=tf.keras.utils.normalize(x_train, axis=1) #normalize the data from 0 to 1 
x_test=tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential() # a basic neural network layer 
model.add(tf.keras.layers.Flatten(input_shape(28,28))) #input is an image of 28*28 pixels
#the 1st hidden layer composed of 128 units & its activation fn is relu 
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
#the 2nd hidden layer composed of 128 units & its activation fn is relu 
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
#the output layer composed of 10 units & its activation fn is softmax activation function  
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=["accuracy"])
#epochs => means it will go through the training loop n times  
model.fit(x_train,y_train,epochs=3)

loss, accuracy= model.evaluate(x_train,y_train)

print("accuracy is: " + accuarcy)
print("loss is: "+ loss)
