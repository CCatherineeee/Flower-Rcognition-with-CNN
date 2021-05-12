from skimage import io,transform   
import glob                        #查找目录和文件模块
import os                          #操作文件夹模块
import tensorflow as tf            #tens框架
import numpy as np                 #数组函数包 
import time                        #时间模块
from tensorflow import keras
import matplotlib.pyplot as plt
from collections import Counter
#数据集地址
path='E:/神经网络/flower_photos/'
#模型保存地址
model_path='E:/神经网络/model/'
 


#将所有的图片resize成100*100
w=100       #宽度
h=100       #高度
c=3         #深度

def read_one_image(path):
    img = io.imread(path,plugin='matplotlib')
    img=transform.resize(img,(w,h))
    np.asarray(img)
    img = np.expand_dims(img,0)
    print(img.shape)
    #img=np.expand_dims(img,axis=0)
    #print(img.shape)
    return img

path1="E:/神经网络/test/1.jpg"
path2="E:/神经网络/test/2.jpg"
path3="E:/神经网络/test/3.jpg"
path4="E:/神经网络/test/4.jpg"
path5="E:/神经网络/test/5.jpg"
path6="E:/神经网络/test/6.jpg"
path7="E:/神经网络/test/7.jpg"
path8="E:/神经网络/test/8.jpg"
path9="E:/神经网络/test/9.jpg"
path10="E:/神经网络/test/10.jpg"

data = []
data1 = read_one_image(path1)
data2 = read_one_image(path2)
data3 = read_one_image(path3)
data4 = read_one_image(path4)
data5 = read_one_image(path5)
data6 = read_one_image(path6)
data7 = read_one_image(path7)
data8 = read_one_image(path8)
data9 = read_one_image(path9)
data10 = read_one_image(path10)

data.append(data1)
data.append(data2)
data.append(data3)
data.append(data4)
data.append(data5)
data.append(data6)
data.append(data7)
data.append(data8)
data.append(data9)
data.append(data10)


flower_dict = {0:'dasiy',1:'dandelion',2:'roses',3:'sunflowers',4:'tulips'}
print(data[0].shape)
model=tf.keras.models.load_model(model_path)

model.summary()
#r=model.fit(x_train, y_train, batch_size=64,epochs=10, validation_data=(x_val, y_val))
i=1
for img in data:
    pred=model.predict(img)
    pred = np.argmax(pred, axis=1)
    for j in pred:
        print("第",i,"朵花是",flower_dict[j])
    i +=1