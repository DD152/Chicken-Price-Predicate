import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, LSTM
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


lr=0.001
batchSize=5
epochs=50
checkpoint_save_path = "./checkpoint/LSTM_chicken.ckpt"

chickenPrice=pd.read_csv('./Meat_prices.csv')
trainingSet=chickenPrice.iloc[0:300,1:2].values
testSet=chickenPrice.iloc[300:361,1:2].values

sc=MinMaxScaler(feature_range=(0, 1))
trainingSet = sc.fit_transform(trainingSet)
testSet=sc.transform(testSet)

xTrain = []
yTrain = []
xTest = []
yTest = []

for i in range(5,len(trainingSet)):
    xTrain.append(trainingSet[i-5:i,0])
    yTrain.append(trainingSet[i,0])

for i in range(5,len(testSet)):
    xTest.append(testSet[i-5:i,0])
    yTest.append(testSet[i,0])

np.random.seed(1412)
np.random.shuffle(xTrain)
np.random.seed(1412)
np.random.shuffle(yTrain)

xTrain,yTrain,xTest,yTest =np.array(xTrain),np.array(yTrain),np.array(xTest),np.array(yTest)

xTrain=np.reshape(xTrain,(xTrain.shape[0],5,1))
xTest=np.reshape(xTest,(xTest.shape[0],5,1))

model=tf.keras.Sequential([
    LSTM(50,return_sequences=True),
    Dropout(0.01),
    LSTM(80),
    Dropout(0.01),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr),loss=tf.keras.losses.MeanSquaredError())

CheakpointCallbacks=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_save_path,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss'
)

history=model.fit(xTrain,yTrain,batchSize,epochs,validation_data=(xTest,yTest),validation_freq=1,callbacks=[CheakpointCallbacks])

model.summary()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

predictedPrice = model.predict(xTest)
# 对预测数据还原---从（0，1）反归一化到原始范围
predictedPrice = sc.inverse_transform(predictedPrice)
# 对真实数据还原---从（0，1）反归一化到原始范围
realPrice = sc.inverse_transform(testSet[5:])
# 画出真实数据和预测数据的对比曲线
plt.plot(realPrice, color='red', label='ChickenPrice')
plt.plot(predictedPrice, color='blue', label='Predicted Chicken Price')
plt.title('Chicken Price Prediction')
plt.xlabel('Time')
plt.ylabel('Chicken Price')
plt.legend()
plt.show()

##########evaluate##############
# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse = mean_squared_error(predictedPrice, realPrice)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = math.sqrt(mean_squared_error(predictedPrice, realPrice))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae = mean_absolute_error(predictedPrice, realPrice)
print('均方误差: %.6f' % mse)
print('均方根误差: %.6f' % rmse)
print('平均绝对误差: %.6f' % mae)