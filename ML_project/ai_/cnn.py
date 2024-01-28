# 풀링층 이미지에 필터를 적용하여 이미지를 줄이고, 대표되는 값을 가져오는 것 / 보폭 = stride 

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.optimizers import Adam

# MNIST 데이터셋을 읽고 신경망에 입력할 형태로 변환
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train.reshape(60000,28,28,1)
x_test=x_test.reshape(10000,28,28,1)
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0
y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)

###############################################
# 신경망 모델 설계
cnn=Sequential()  # <----- cnn 선언 
                
                #3*3 필터를 32번 사용한다. activation=‘relu’는 컨볼루션 결과에 ReLU 활성 함수를 적용하라는 뜻
                # input_shape=(28,28,1) 매개변수는 신경망에 (28*28*1) 텐서가 입력된다는 사실 알려줌 
                # (28*28) 대신 (28*28*1)을 사용하는 이유는 일반성 유지(RGB 영상의 경우 (28*28*3)으로 확장)  
cnn.add(Conv2D(32,(3,3),activation='relu', input_shape=(28,28,1)))
cnn.add(Conv2D(64,(3,3), activation='relu'))

# 블록 내의 원소들 중 최대값을 대표값으로 선택하는 Max Pooling
cnn.add(MaxPooling2D(pool_size=(2,2)))  # 필터링 

#Dropout    일정 비율의 가중치를 임의로 선택하여 불능으로 만들고 학습하는 규제기법
# 불능이 될 샘플마다 독립적으로 정하는데 난수를 통해 랜덤하게 선택함. 
# 즉, 랜덤한 노드를 빼버림 
cnn.add(Dropout(0.25))

#컴퓨터가 인식할 수 있게 1차원으로 만듦
cnn.add(Flatten())

#완전연결층
cnn.add(Dense(128, activation='relu'))

#Dropout
cnn.add(Dropout(0.5))

#완전연결층
cnn.add(Dense(10,activation='softmax'))


#신경망 모델 학습
cnn.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
hist = cnn.fit(x_train,y_train,batch_size=128, epochs=12, validation_data=(x_test,y_test), verbose=2)


# 신경망 모델 정확률 평가


res=cnn.evaluate(x_test,y_test,verbose=0)
print("정확률은",res[1]*100)

import matplotlib.pyplot as plt



# 정확률 그래프
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='best')
plt.grid()
plt.show()

# 손실 함수 그래프
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Modelloss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'],loc='best')
plt.grid()
plt.show()


#학습된 모델 저장
#   cnn.save("파일이름")    구조정보와 가중치를 저장해준다. 
# 불러오기
# cnn=tf.keras.model/load_model("파일명.h5")