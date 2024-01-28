import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam        # 최적화 방법 

# MNIST 데이터셋을 읽고 신경망에 입력할 형태로 변환 
(x_train, y_train), (x_test,y_test)=mnist.load_data()
x_train=x_train.reshape(60000,28,28,1)
x_test=x_test.reshape(10000,28,28,1)

x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0

y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)

#LeNet-5 산경망 모델 설계
cnn=Sequential()    # cnn

    # 5*5 필터를 6번 사용한다. ctivation=‘relu’는 컨볼루션 결과에 ReLU 활성 함수를 적용하라는 뜻
            # input_shape=(28,28,1) 매개변수는 신경망에 (28*28*1) 텐서가 입력된다는 사실 알려줌 
            # (28*28) 대신 (28*28*1)을 사용하는 이유는 일반성 유지(RGB 영상의 경우 (28*28*3)으로 확장)  

cnn.add(Conv2D(6,(5,5), padding="same", activation='relu',input_shape=(28,28,1)))
#풀링층
cnn.add(MaxPooling2D(pool_size=(2,2)))  # 2*2 풀링 

                # 5*5 필터를 16번 사용한다.
cnn.add(Conv2D(16,(5,5), padding="same", activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))  # 2*2 풀링 
                # 5*5 필터를 120번 사용한다.
cnn.add(Conv2D(120,(5,5), padding="same", activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))  # 2*2 풀링 

cnn.add(Flatten())  # 다차원 구조를 1차원(배열) 구조로 변환 

# Dense Layer를 사용하면 감소된 차원의 Feature Map들만 Input으로 
# 하여 Output과 완전연결 계층을 생성하여, 더 효율적인 학습이 가능해진다. 

# 모델.add(Dense(노드개수, 활성화함수))     활성화함수 계단 함수 
cnn.add(Dense(128,activation='relu'))       #128개의 노드 생성

#
cnn.add(Dense(10,activation='softmax')) # 10개의 노드 생성 

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