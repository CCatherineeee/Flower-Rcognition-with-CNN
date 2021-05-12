from skimage import io,transform   
import glob                        #查找目录和文件模块
import os                          #操作文件夹模块
import tensorflow as tf            #tens框架
import numpy as np                 #数组函数包 
import time                        #时间模块
from tensorflow import keras
import matplotlib.pyplot as plt
from collections import Counter
import random
from tensorflow.keras.callbacks import ReduceLROnPlateau
#数据集地址
path='E:/神经网络/flower_photos/'
#模型保存地址
model_path='E:/神经网络/model/'
 
#将所有的图片resize成100*100
w=100       #宽度
h=100       #高度
c=3         #深度
 
#读取图片
def read_img(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    imgs=[]
    labels=[]
    #转化为索引
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            print('reading the images:%s'%(im))
            img=io.imread(im)
            img=transform.resize(img,(w,h))
            imgs.append(img)
            labels.append(idx)
            p=random.random()
            if(p<0.5):
                img = transform.rotate(img, 15)
            imgs.append(img)
            labels.append(idx)
            print(idx)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)
data,label=read_img(path)
print("label")
#打乱顺序
num_example=data.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
label=label[arr]

#将所有数据分为训练集和验证集
ratio=0.8
s=np.int(num_example*ratio)
x_train=data[:s]
y_train=label[:s]
x_val=data[s:]
y_val=label[s:]

print(x_train)
print(x_train.shape)
print(x_train[0].shape)
print(y_train.shape)

y_train=tf.keras.utils.to_categorical(y_train,5)
y_val=tf.keras.utils.to_categorical(y_val,5)

#搭建
def network():
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32,(5,5),input_shape=(w,h,3),activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=6,activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128,activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))  
    model.add(tf.keras.layers.Dense(64,activation='relu')) 
    model.add(tf.keras.layers.Dropout(0.2)) 
    model.add(tf.keras.layers.Dense(5,activation='softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model


model=network()

print(y_val.shape)
print(x_val.shape)
print(x_train.shape)
print(y_train.shape)


reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
r=model.fit(x_train, y_train, batch_size=120, epochs=15, validation_data=(x_val, y_val), callbacks=[reduce_lr],shuffle=True)
#r=model.fit(x_train, y_train, batch_size=90,epochs=25, validation_data=(x_val, y_val))
test_loss, test_acc = model.evaluate(x_val, y_val, verbose=0)
print(f'测试集损失值: {test_loss}, 测试集准确率: {test_acc}')

print(r.history.keys()) 
accuracy = r.history['accuracy']
val_accuracy = r.history['val_accuracy']
loss = r.history['loss']
val_loss = r.history['val_loss']
epochs = range(len(accuracy))

plt.figure()
plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'bo', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('flower_accuracy_2.png', bbox_inches='tight', dpi=300)

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('flower_loss_2.png', bbox_inches='tight', dpi=300)

#model.save_weights(model_path)
model.save(model_path)
print("saved")